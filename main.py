#!/usr/bin/env python3
"""
audiothing assembler - Text-to-Speech generation CLI.
Coordinates configuration, sanitization, chunking, storage, and speech synthesis.
"""

import argparse
import sys
import wave
from pathlib import Path

from audiothing import (
    DEFAULT_API_KEY,
    DEFAULT_FUNCTION_ID,
    DEFAULT_LANG,
    DEFAULT_PARAGRAPH_PAUSE,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SENTENCE_PAUSE,
    DEFAULT_SERVER,
    DEFAULT_VOICE,
    get_next_audio_filename,
    prepare_speech_units,
    sanitize_text,
    synthesize_speech,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthesize audio from a text file using NVIDIA Riva TTS with automatic incremental output files."
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("input.txt"),
        help="Path to the input text file (default: input.txt)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to save generated audio files (default: output)",
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help="Riva gRPC server URI (default: grpc.nvcf.nvidia.com:443)",
    )
    parser.add_argument(
        "--function-id",
        default=DEFAULT_FUNCTION_ID,
        help="NVCF Function ID",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="NVCF API key / token",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help="Voice name to use (default: Chatterbox-Multilingual.en-US.Male)",
    )
    parser.add_argument(
        "--language-code",
        default=DEFAULT_LANG,
        help="Language code (default: en-US)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Sample rate in Hz (default: 22050)",
    )
    parser.add_argument(
        "--sentence-pause",
        type=float,
        default=DEFAULT_SENTENCE_PAUSE,
        help=f"Pause duration between sentences in seconds (default: {DEFAULT_SENTENCE_PAUSE})",
    )
    parser.add_argument(
        "--paragraph-pause",
        type=float,
        default=DEFAULT_PARAGRAPH_PAUSE,
        help=f"Pause duration between paragraphs in seconds (default: {DEFAULT_PARAGRAPH_PAUSE})",
    )
    parser.add_argument(
        "--no-sanitize",
        action="store_true",
        help="Disable automatic text sanitization (stripping quotes, backticks, dashes, underscores, markdown)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.api_key:
        print("Error: NVIDIA/NVCF API key is required.", file=sys.stderr)
        print("Set NVCF_API_KEY in your .env or environment, or pass --api-key <key>.", file=sys.stderr)
        sys.exit(1)

    # Verify input text file
    if not args.input.is_file():
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        print(f"Please create '{args.input}' with the text you want to synthesize.", file=sys.stderr)
        sys.exit(1)

    raw_text = args.input.read_text(encoding="utf-8").strip()
    if not raw_text:
        print(f"Error: Input file '{args.input}' is empty.", file=sys.stderr)
        sys.exit(1)

    # Sanitize text for TTS unless explicitly disabled
    if not args.no_sanitize:
        text = sanitize_text(raw_text)
        if not text:
            print(f"Error: No valid speech content left in '{args.input}' after sanitization.", file=sys.stderr)
            sys.exit(1)
    else:
        text = raw_text

    # Prepare speech units with paragraph boundary metadata
    speech_units = prepare_speech_units(text)
    if not speech_units:
        print(f"Error: No valid text chunks found in '{args.input}'.", file=sys.stderr)
        sys.exit(1)

    # Determine next incremental output filename
    output_path = get_next_audio_filename(args.output_dir)
    print(f"Input file: {args.input} ({len(text)} characters, {len(speech_units)} chunks)")
    print(f"Output target: {output_path}\n")

    # Synthesize audio
    try:
        synthesize_speech(
            speech_units=speech_units,
            output_path=output_path,
            server=args.server,
            function_id=args.function_id,
            api_key=args.api_key,
            voice=args.voice,
            language_code=args.language_code,
            sample_rate=args.sample_rate,
            sentence_pause=args.sentence_pause,
            paragraph_pause=args.paragraph_pause,
        )
    except Exception as e:
        print(f"\nError during speech synthesis: {e}", file=sys.stderr)
        sys.exit(1)

    # Summary
    if output_path.is_file():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        with wave.open(str(output_path), "rb") as f:
            duration = f.getnframes() / float(f.getframerate())
        print(f"Successfully generated: {output_path}")
        print(f"Duration: {duration:.2f}s | Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
