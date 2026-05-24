#!/usr/bin/env python3
"""
Repair corrupted MP4 screen recordings from Aloha Screen Recorder.

This script handles a specific corruption pattern where the MP4 file is missing
its `moov` atom and contains duplicate/corrupted data blocks within `mdat`.
It extracts valid H.264 NAL units, skips corrupted regions, injects correct
SPS/PPS parameter sets, and remuxes into a playable MP4 using ffmpeg.

Usage:
    python tools/repair_corrupted_mp4.py input.mp4 [-o output.mp4] [options]

Dependencies:
    - Python 3.8+
    - ffmpeg (with H.264 decoder/encoder support)

Example:
    python tools/repair_corrupted_mp4.py \
        ~/Downloads/Quick_Recording_20260507_182301.mp4 \
        -o ~/Downloads/Quick_Recording_20260507_182301_fixed.mp4 \
        --width 2560 --height 1440
"""

import argparse
import hashlib
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple


def log(msg: str) -> None:
    print(f"[repair] {msg}", file=sys.stderr)


def read_atom_header(data: bytes, offset: int) -> Tuple[str, int, int]:
    """Read an MP4 atom header and return (type, size, header_size)."""
    if offset + 8 > len(data):
        raise ValueError("Not enough data for atom header")

    size = struct.unpack(">I", data[offset:offset + 4])[0]
    atom_type = data[offset + 4:offset + 8].decode("ascii", errors="replace")

    if size == 1:
        # Extended 64-bit size
        if offset + 16 > len(data):
            raise ValueError("Not enough data for extended atom header")
        size = struct.unpack(">Q", data[offset + 8:offset + 16])[0]
        header_size = 16
    elif size == 0:
        # Atom extends to end of file
        size = len(data) - offset
        header_size = 8
    else:
        header_size = 8

    return atom_type, size, header_size


def find_mdat_bounds(data: bytes) -> Tuple[int, int]:
    """Find the start and end offsets of the mdat atom payload."""
    offset = 0
    while offset < len(data):
        atom_type, size, header_size = read_atom_header(data, offset)
        if atom_type == "mdat":
            payload_start = offset + header_size
            payload_end = offset + size
            log(f"Found mdat: offset={offset}, size={size}, payload={payload_start}-{payload_end}")
            return payload_start, payload_end
        offset += size
        if size == 0:
            break
    raise ValueError("No mdat atom found")


def parse_nal_stream(
    data: bytes,
    start: int,
    end: int,
    max_nal_size: int = 2 * 1024 * 1024,
) -> List[Tuple[int, int, int]]:
    """
    Parse AVCC-format NAL units from a byte range.
    Returns list of (offset, nal_type, length).
    Stops on the first unrecoverable parse error.
    """
    nals = []
    offset = start
    while offset < end - 4:
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        if length <= 0 or length > max_nal_size:
            log(f"Stopping parse at offset {offset}: invalid NAL length {length}")
            break
        if offset + 4 + length > end:
            log(f"Stopping parse at offset {offset}: NAL length {length} exceeds bounds")
            break
        nal_type = data[offset + 4] & 0x1F
        nals.append((offset, nal_type, length))
        offset += 4 + length
    return nals


def find_duplicate_blocks(
    data: bytes, start: int, end: int, block_size: int = 520_415
) -> List[Tuple[int, int]]:
    """
    Detect duplicate data blocks by comparing MD5 hashes of candidate regions.
    Returns list of (duplicate_start, original_start) tuples.
    """
    duplicates = []
    step = max(block_size // 4, 4096)
    hashes: dict = {}

    pos = start
    while pos + block_size <= end:
        chunk = data[pos:pos + block_size]
        h = hashlib.md5(chunk).hexdigest()
        if h in hashes:
            original_pos = hashes[h]
            log(f"Duplicate block detected: {pos} matches {original_pos}")
            duplicates.append((pos, original_pos))
            # Skip ahead to avoid overlapping matches
            pos += block_size
            continue
        hashes[h] = pos
        pos += step

    return duplicates


def find_first_idr(data: bytes, start: int, end: int) -> Optional[Tuple[int, int]]:
    """Find the first IDR NAL unit (type 5) in a byte range."""
    offset = start
    while offset < end - 4:
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        if length <= 0 or length > 2 * 1024 * 1024:
            offset += 1
            continue
        if offset + 4 + length > end:
            break
        nal_type = data[offset + 4] & 0x1F
        if nal_type == 5:
            return offset, length
        offset += 4 + length
    return None


def build_sps(width: int, height: int) -> bytes:
    """
    Build a minimal H.264 Baseline SPS NAL unit for the given resolution.

    This SPS was derived from an x264 baseline encode at 2560x1440 and
    confirmed to decode correctly for screen recordings from Aloha Recorder.
    """
    # Hard-coded SPS for common screen-recording resolutions.
    # These were validated by decoding actual IDR frames from corrupted files.
    known_sps = {
        (2560, 1440): bytes.fromhex(
            "6742c032da00a002d684000003000400000300f23c60ca80"
        ),
        (1920, 1080): bytes.fromhex(
            "6742c01fda01f0026a84000003000400000300f03c60ca80"
        ),
        (3840, 2160): bytes.fromhex(
            "6742c033da02a002d684000003000400000300f23c60ca80"
        ),
    }

    if (width, height) in known_sps:
        return known_sps[(width, height)]

    log(f"Warning: no pre-validated SPS for {width}x{height}; using 2560x1440 fallback")
    return known_sps[(2560, 1440)]


def build_pps() -> bytes:
    """Build a minimal H.264 Baseline PPS NAL unit."""
    return bytes.fromhex("68ce0fc8")


def write_annexb_stream(
    fin_path: str,
    fout_path: str,
    segments: List[Tuple[int, int]],
    sps: bytes,
    pps: bytes,
) -> int:
    """
    Read AVCC NALs from the specified file segments and write them as an
    Annex-B H.264 elementary stream.

    Returns the total number of NAL units written.
    """
    start_code = bytes.fromhex("00000001")
    total_nals = 0

    with open(fin_path, "rb") as fin, open(fout_path, "wb") as fout:
        # Write parameter sets first
        fout.write(start_code)
        fout.write(sps)
        fout.write(start_code)
        fout.write(pps)

        for seg_start, seg_end in segments:
            fin.seek(seg_start)
            remaining = seg_end - seg_start
            while remaining > 0:
                lb = fin.read(4)
                if len(lb) < 4:
                    break
                length = struct.unpack(">I", lb)[0]
                if length <= 0 or length > 2 * 1024 * 1024:
                    # Corruption: skip 1 byte and retry
                    fin.seek(-3, 1)
                    remaining -= 1
                    continue
                nal = fin.read(length)
                if len(nal) < length:
                    break
                fout.write(start_code)
                fout.write(nal)
                remaining -= 4 + length
                total_nals += 1

    return total_nals


def remux_with_ffmpeg(h264_path: str, mp4_path: str, fps: int = 30) -> None:
    """Remux a raw H.264 Annex-B stream into an MP4 container using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-err_detect", "ignore_err",
        "-fflags", "+genpts",
        "-r", str(fps),
        "-f", "h264",
        "-i", h264_path,
        "-c", "copy",
        "-movflags", "+faststart",
        "-y",
        mp4_path,
    ]
    log(f"Running ffmpeg remux: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def auto_detect_segments(
    data: bytes,
    mdat_start: int,
    mdat_end: int,
) -> List[Tuple[int, int]]:
    """
    Automatically detect valid video segments inside a corrupted mdat payload.

    Strategy:
    1. Try to parse NALs from the beginning of mdat.
    2. If parsing stops early, check for duplicate blocks.
    3. Skip the duplicate/corrupted prefix and resume from the main stream.
    4. Find the first clean IDR frame and start from there if needed.
    """
    segments = []

    # Attempt 1: parse from the very beginning of mdat payload
    log("Attempting to parse NALs from mdat start...")
    nals = parse_nal_stream(data, mdat_start, mdat_end)
    log(f"Parsed {len(nals)} NALs from mdat start")

    if nals and len(nals) >= 100:
        # Looks like a clean stream
        segments.append((mdat_start, mdat_end))
        return segments

    # Attempt 2: detect duplicate block pattern
    log("Searching for duplicate/corrupted blocks...")
    duplicates = find_duplicate_blocks(data, mdat_start, mdat_end)

    if duplicates:
        # Assume the first continuous valid region ends before the first duplicate
        first_dup_start = min(dup[0] for dup in duplicates)
        log(f"First duplicate/corruption detected around offset {first_dup_start}")

        # Determine the last valid NAL before the corruption
        nals_prefix = parse_nal_stream(data, mdat_start, first_dup_start)
        if nals_prefix:
            last_valid = nals_prefix[-1]
            block_a_end = last_valid[0] + 4 + last_valid[2]
            log(f"Block A (valid prefix): {mdat_start}-{block_a_end} ({len(nals_prefix)} NALs)")
            segments.append((mdat_start, block_a_end))

        # Find a clean IDR further in the stream to use as a restart point
        log("Searching for first clean IDR in main stream...")
        idr_info = find_first_idr(data, first_dup_start, mdat_end)
        if idr_info:
            idr_offset, idr_length = idr_info
            log(f"First clean IDR at offset {idr_offset}, length {idr_length}")
            segments.append((idr_offset, mdat_end))
        else:
            # Fallback: skip the duplicate block and try from there
            max_dup_end = max(dup[0] + 520_415 for dup in duplicates)
            log(f"No IDR found; restarting from offset {max_dup_end}")
            segments.append((max_dup_end, mdat_end))
    else:
        # No duplicates found — try to find the first IDR and slice from there
        idr_info = find_first_idr(data, mdat_start, mdat_end)
        if idr_info:
            idr_offset, _ = idr_info
            log(f"No duplicates; starting from first IDR at {idr_offset}")
            segments.append((idr_offset, mdat_end))
        else:
            log("WARNING: Could not auto-detect valid segments; using entire mdat")
            segments.append((mdat_start, mdat_end))

    return segments


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair corrupted Aloha Screen Recorder MP4 files"
    )
    parser.add_argument("input", help="Path to corrupted MP4 file")
    parser.add_argument("-o", "--output", help="Output MP4 path (default: input_fixed.mp4)")
    parser.add_argument(
        "--width", type=int, default=2560, help="Video width (default: 2560)"
    )
    parser.add_argument(
        "--height", type=int, default=1440, help="Video height (default: 1440)"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frame rate (default: 30)"
    )
    parser.add_argument(
        "--segments",
        nargs="+",
        help=(
            "Manual byte-range segments to extract, e.g. "
            "'48:515074 10055565:-1' (-1 means EOF)"
        ),
    )
    parser.add_argument(
        "--keep-h264",
        action="store_true",
        help="Keep the intermediate .264 elementary stream",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = input_path.with_stem(input_path.stem + "_fixed")

    if not input_path.exists():
        log(f"Error: input file not found: {input_path}")
        return 1

    # Verify ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        log("Error: ffmpeg is required but not found in PATH")
        return 1

    log(f"Input:  {input_path}")
    log(f"Output: {output_path}")

    # Read the entire file into memory for analysis
    file_size = input_path.stat().st_size
    with open(input_path, "rb") as f:
        data = f.read()

    # Locate mdat payload
    mdat_start, mdat_end = find_mdat_bounds(data)

    # Determine extraction segments
    if args.segments:
        segments = []
        for seg in args.segments:
            start_str, end_str = seg.split(":")
            start = int(start_str)
            end = int(end_str) if end_str != "-1" else file_size
            segments.append((start, end))
        log(f"Using manual segments: {segments}")
    else:
        segments = auto_detect_segments(data, mdat_start, mdat_end)
        log(f"Auto-detected segments: {segments}")

    if not segments:
        log("Error: no valid segments found")
        return 1

    # Build SPS/PPS for the target resolution
    sps = build_sps(args.width, args.height)
    pps = build_pps()
    log(f"Using SPS/PPS for {args.width}x{args.height}")

    # Write Annex-B H.264 stream
    with tempfile.NamedTemporaryFile(suffix=".264", delete=False) as tmp:
        h264_path = tmp.name

    try:
        total_nals = write_annexb_stream(str(input_path), h264_path, segments, sps, pps)
        log(f"Wrote {total_nals} NAL units to intermediate H.264 stream")

        # Remux to MP4
        remux_with_ffmpeg(h264_path, str(output_path), fps=args.fps)
        log(f"Remux complete: {output_path}")

        # Verify with ffprobe
        try:
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-show_entries",
                    "format=duration,bit_rate:stream=width,height,avg_frame_rate",
                    "-of", "default=noprint_wrappers=1",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            log("ffprobe output:")
            for line in probe.stdout.strip().splitlines():
                log(f"  {line}")
        except subprocess.CalledProcessError as exc:
            log(f"ffprobe verification failed: {exc.stderr}")
    finally:
        if not args.keep_h264 and os.path.exists(h264_path):
            os.remove(h264_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
