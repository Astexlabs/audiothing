"""
NVIDIA Riva speech synthesis client integration.
"""

import time
import wave
from pathlib import Path

import grpc
import riva.client
from riva.client.proto.riva_audio_pb2 import AudioEncoding

from audiothing.config import (
    DEFAULT_PARAGRAPH_PAUSE,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SENTENCE_PAUSE,
)


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
