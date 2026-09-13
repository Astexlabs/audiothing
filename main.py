#!/usr/bin/env python3
"""
audiothing assembler - Text-to-Speech generation CLI.
Coordinates configuration, sanitization, chunking, storage, and speech synthesis.
"""

import argparse
import sys
import time
import wave
from pathlib import Path

from audiothing import (
    DEFAULT_API_KEY,
    DEFAULT_FUNCTION_ID,
    DEFAULT_LANG,
    DEFAULT_MAX_CHARS,
    DEFAULT_PARAGRAPH_PAUSE,
    DEFAULT_REQUEST_DELAY,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SENTENCE_PAUSE,
    DEFAULT_SERVER,
    DEFAULT_VOICE,
    get_next_audio_filename,
    get_next_comparison_filenames,
    prepare_speech_units,
    pronounce_initialisms,
    pronounce_initialisms_phonetic,
    sanitize_text,
    synthesize_speech,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthesize audio from a text file using NVIDIA Riva TTS with automatic incremental output files."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("input.txt"),
        help="Path to the input text file (default: input.txt)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
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
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=f"Maximum characters per speech chunk (default: {DEFAULT_MAX_CHARS})",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
        help=f"Delay between synthesis requests in seconds (default: {DEFAULT_REQUEST_DELAY})",
    )
    parser.add_argument(
        "--no-sanitize",
        action="store_true",
        help="Disable automatic text sanitization (stripping quotes, backticks, dashes, underscores, markdown)",
    )
    parser.add_argument(
        "--pronounce-initialisms",
        action="store_true",
        help="Experimentally spell out standalone uppercase initialisms of 2-3 letters (for example, AI -> ay eye)",
    )
    parser.add_argument(
        "--phonetic-initialisms",
        action="store_true",
        help="Use IPA-style phonetic signs for initialisms (for example, AI -> eɪ aɪ)",
    )
    parser.add_argument(
        "--a-b",
        action="store_true",
        help="Generate paired audio files with and without experimental initialism pronunciation",
    )
    parser.add_argument(
        "--text-only",
        "--dry-run",
        dest="text_only",
        action="store_true",
        help="Print processed text and skip all audio generation",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Verify input text file
    if not args.input.is_file():
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        print(
            f"Please create '{args.input}' with the text you want to synthesize.",
            file=sys.stderr,
        )
        sys.exit(1)

    raw_text = args.input.read_text(encoding="utf-8").strip()
    if not raw_text:
        print(f"Error: Input file '{args.input}' is empty.", file=sys.stderr)
        sys.exit(1)

    # Sanitize text for TTS unless explicitly disabled
    if not args.no_sanitize:
        text = sanitize_text(raw_text)
        if not text:
            print(
                f"Error: No valid speech content left in '{args.input}' after sanitization.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        text = raw_text

    if args.a_b:
        text_variants = [
            ("without classifier", text),
            ("with classifier", pronounce_initialisms(text)),
            ("with phonetic signs", pronounce_initialisms_phonetic(text)),
        ]
    else:
        if args.phonetic_initialisms:
            text = pronounce_initialisms_phonetic(text)
        elif args.pronounce_initialisms:
            text = pronounce_initialisms(text)
        text_variants = [
            (
                (
                    "with phonetic signs"
                    if args.phonetic_initialisms
                    else "with classifier"
                    if args.pronounce_initialisms
                    else "default"
                ),
                text,
            )
        ]

    if args.text_only:
        if args.a_b:
            for label, run_text in text_variants:
                print(f"--- {label} ---")
                print(run_text)
        else:
            print(text_variants[0][1])
        return

    if not args.api_key:
        print("Error: NVIDIA/NVCF API key is required.", file=sys.stderr)
        print(
            "Set NVCF_API_KEY in your .env or environment, or pass --api-key <key>.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.a_b:
        output_paths = get_next_comparison_filenames(args.output_dir)
        # may cause ratelimit errs when all 3 are ran together
        runs = [
            (
                text_variants[0][0],
                text_variants[0][1],
                output_paths[0],
            ),  # no classifier
            (
                text_variants[1][0],
                text_variants[1][1],
                output_paths[1],
            ),  # with classifier
            (
                text_variants[2][0],
                text_variants[2][1],
                output_paths[2],
            ),  # phonetics EXPERIMENTAL
        ]
    else:
        runs = [
            (
                text_variants[0][0],
                text_variants[0][1],
                get_next_audio_filename(args.output_dir),
            )
        ]

    prepared_runs = []
    for label, run_text, output_path in runs:
        # Prepare speech units with paragraph boundary metadata.
        speech_units = prepare_speech_units(run_text, max_chars=args.max_chars)
        if not speech_units:
            print(
                f"Error: No valid text chunks found in '{args.input}'.", file=sys.stderr
            )
            sys.exit(1)
        prepared_runs.append((label, run_text, speech_units, output_path))

    if args.a_b:
        print(
            "A/B/C comparison: generating original, word-based, and phonetic variants"
        )

    for run_index, (label, run_text, speech_units, output_path) in enumerate(
        prepared_runs
    ):
        print(f"\n--- {label} ---")
        print(
            f"Input file: {args.input} ({len(run_text)} characters, {len(speech_units)} chunks)"
        )
        print(f"Output target: {output_path}\n")

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
                request_delay=args.request_delay,
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

        if run_index < len(prepared_runs) - 1 and args.request_delay > 0:
            print(
                f"Waiting {args.request_delay:.2f}s before the next comparison run..."
            )
            time.sleep(args.request_delay)


if __name__ == "__main__":
    main()
