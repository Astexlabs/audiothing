#!/usr/bin/env python3
"""
Text-to-Speech generation using NVIDIA Riva / NVCF Chatterbox-Multilingual.
Reads prompt text from an input file, handles automatic chunking to adhere
to Triton model input limits, preserves natural speech pauses between sentences
and paragraphs, and increments output audio filenames (+1).
"""

import argparse
import os
import re
import sys
import time
import wave
from pathlib import Path

import grpc
import riva.client
from riva.client.proto.riva_audio_pb2 import AudioEncoding

def load_dotenv(dotenv_path: Path = Path(".env")) -> None:
    """Simple parser to load key-value pairs from a .env file into os.environ."""
    if dotenv_path.is_file():
        with dotenv_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v


load_dotenv()

DEFAULT_SERVER = os.getenv("RIVA_SERVER", "grpc.nvcf.nvidia.com:443")
DEFAULT_FUNCTION_ID = os.getenv("NVCF_FUNCTION_ID", "ddacc747-1269-4fab-bfd9-8f593dead106")
DEFAULT_API_KEY = os.getenv("NVCF_API_KEY", os.getenv("NVIDIA_API_KEY", ""))
DEFAULT_VOICE = os.getenv("RIVA_VOICE", "Chatterbox-Multilingual.en-US.Male")
DEFAULT_LANG = os.getenv("RIVA_LANGUAGE_CODE", "en-US")
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_SENTENCE_PAUSE = 0.25   # seconds of silence between sentences in the same paragraph
DEFAULT_PARAGRAPH_PAUSE = 0.50  # seconds of silence between paragraphs


def get_next_audio_filename(output_dir: Path, base_name: str = "audio", ext: str = ".wav") -> Path:
    """
    Finds the next incremental filename in output_dir (e.g. audio_1.wav, audio_2.wav, ...).
    Always performs a +1 so existing audio files are never overwritten.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(base_name)}_?(\d+){re.escape(ext)}$")
    existing_nums = []

    for item in output_dir.iterdir():
        if item.is_file():
            match = pattern.match(item.name)
            if match:
                existing_nums.append(int(match.group(1)))
            elif item.name == f"{base_name}{ext}":
                existing_nums.append(0)

    next_num = max(existing_nums, default=0) + 1
    return output_dir / f"{base_name}_{next_num}{ext}"


def prepare_speech_units(text: str, max_chars: int = 350) -> list[tuple[str, bool]]:
    """
    Splits input text into manageable chunks respecting sentence and clause boundaries.
    The Triton Chatterbox model has a hard limit of 500 tokens/characters per request,
    so each chunk is kept strictly under max_chars (default 350).

    Returns a list of (chunk_text, is_paragraph_end) tuples so appropriate pause durations
    can be inserted between sentences vs paragraphs.
    """
    raw_paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    units: list[tuple[str, bool]] = []

    for p in raw_paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', p)
        p_chunks: list[str] = []
        cur = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue

            if len(s) > max_chars:
                if cur:
                    p_chunks.append(cur)
                    cur = ""
                # Split large sentence on clause punctuation (,;:)
                subparts = re.split(r'(?<=[,;:])\s+', s)
                for part in subparts:
                    part = part.strip()
                    if not part:
                        continue
                    if len(part) > max_chars:
                        for word in part.split():
                            if cur and len(cur) + len(word) + 1 > max_chars:
                                p_chunks.append(cur)
                                cur = word
                            else:
                                cur = f"{cur} {word}" if cur else word
                    else:
                        if cur and len(cur) + len(part) + 1 > max_chars:
                            p_chunks.append(cur)
                            cur = part
                        else:
                            cur = f"{cur} {part}" if cur else part
                if cur:
                    p_chunks.append(cur)
                    cur = ""
            else:
                if cur and len(cur) + len(s) + 1 > max_chars:
                    p_chunks.append(cur)
                    cur = s
                else:
                    cur = f"{cur} {s}" if cur else s
        if cur:
            p_chunks.append(cur)

        # Mark all chunks in this paragraph
        for i, chunk in enumerate(p_chunks):
            is_end = (i == len(p_chunks) - 1)
            units.append((chunk, is_end))

    return units


def synthesize_chunk_with_retry(
    service: riva.client.SpeechSynthesisService,
    chunk: str,
    voice: str,
    language_code: str,
    sample_rate: int,
    max_retries: int = 5,
) -> bytes:
    """
    Synthesizes a single chunk using service.synthesize with exponential backoff on rate limits.
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = service.synthesize(
                chunk,
                voice_name=voice,
                language_code=language_code,
                encoding=AudioEncoding.LINEAR_PCM,
                sample_rate_hz=sample_rate,
            )
            return resp.audio
        except grpc.RpcError as e:
            err_str = str(e)
            if ("RESOURCE_EXHAUSTED" in err_str or "rate limit" in err_str) and attempt < max_retries:
                wait_time = attempt * 2
                print(f"  [Rate limit encountered, waiting {wait_time}s before retry {attempt}/{max_retries}...]")
                time.sleep(wait_time)
                continue
            raise


def synthesize_speech(
    speech_units: list[tuple[str, bool]],
    output_path: Path,
    server: str,
    function_id: str,
    api_key: str,
    voice: str,
    language_code: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    sentence_pause: float = DEFAULT_SENTENCE_PAUSE,
    paragraph_pause: float = DEFAULT_PARAGRAPH_PAUSE,
) -> None:
    """
    Synthesizes speech units individually and writes to a WAV file with natural silence pauses.
    Individual chunk synthesis eliminates the clipping, mumbling, and audio overlap that occurs
    when sending batch requests over an unpaced gRPC streaming channel.
    """
    auth_header = api_key if api_key.startswith("Bearer ") else f"Bearer {api_key}"
    metadata = [
        ("function-id", function_id),
        ("authorization", auth_header),
    ]

    auth = riva.client.Auth(
        uri=server,
        use_ssl=True,
        metadata_args=metadata,
    )
    service = riva.client.SpeechSynthesisService(auth)

    sentence_pause_bytes = b"\x00" * (int(sample_rate * sentence_pause) * 2)
    paragraph_pause_bytes = b"\x00" * (int(sample_rate * paragraph_pause) * 2)

    total = len(speech_units)
    print(f"Connecting to Riva server at {server}...")
    print(f"Voice: {voice} | Language: {language_code} | Sample Rate: {sample_rate}Hz")
    print(f"Sentence pause: {sentence_pause:.2f}s | Paragraph pause: {paragraph_pause:.2f}s")
    print(f"Synthesizing {total} chunks with clean pacing...\n")

    start_time = time.time()
    try:
        with wave.open(str(output_path), "wb") as out_f:
            out_f.setnchannels(1)
            out_f.setsampwidth(2)  # 16-bit PCM
            out_f.setframerate(sample_rate)

            for idx, (chunk, is_paragraph_end) in enumerate(speech_units, start=1):
                preview = (chunk[:45] + "...") if len(chunk) > 45 else chunk
                print(f"[{idx:2d}/{total}] ({len(chunk)}c) \"{preview}\"")
                audio_bytes = synthesize_chunk_with_retry(
                    service=service,
                    chunk=chunk,
                    voice=voice,
                    language_code=language_code,
                    sample_rate=sample_rate,
                )
                if audio_bytes:
                    out_f.writeframesraw(audio_bytes)

                # Add natural pause between chunks (skip trailing pause after final chunk)
                if idx < total:
                    pause = paragraph_pause_bytes if is_paragraph_end else sentence_pause_bytes
                    out_f.writeframesraw(pause)

        elapsed = time.time() - start_time
        print(f"\nSynthesis completed in {elapsed:.2f}s.")
    except Exception:
        if output_path.exists():
            output_path.unlink()
        raise


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

    text = args.input.read_text(encoding="utf-8").strip()
    if not text:
        print(f"Error: Input file '{args.input}' is empty.", file=sys.stderr)
        sys.exit(1)

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
