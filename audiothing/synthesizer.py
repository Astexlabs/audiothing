"""Riva speech synthesis client integration."""

import time
import wave
from pathlib import Path

import grpc
import riva.client
from riva.client.proto.riva_audio_pb2 import AudioEncoding

from audiothing.config import (
    DEFAULT_REQUEST_DELAY,
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
    """Synthesize a chunk with retry and backoff."""
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
            err_str = str(e).lower()
            status_code = e.code() if hasattr(e, "code") else None
            is_rate_limited = (
                status_code == grpc.StatusCode.RESOURCE_EXHAUSTED
                or "resource_exhausted" in err_str
                or "rate limit" in err_str
                or "too many requests" in err_str
            )
            if is_rate_limited and attempt < max_retries:
                wait_time = min(30, 2 ** (attempt - 1))
                print(
                    f"  [Rate limit encountered, backing off {wait_time}s before retry {attempt}/{max_retries}...]"
                )
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
    request_delay: float = DEFAULT_REQUEST_DELAY,
) -> None:
    """Synthesize speech units to WAV with natural pauses."""
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
    print(
        f"Sentence pause: {sentence_pause:.2f}s | Paragraph pause: {paragraph_pause:.2f}s"
    )
    print(f"Synthesizing {total} chunks with clean pacing...\n")

    start_time = time.time()
    try:
        with wave.open(str(output_path), "wb") as out_f:
            out_f.setnchannels(1)
            out_f.setsampwidth(2)
            out_f.setframerate(sample_rate)

            for idx, (chunk, is_paragraph_end) in enumerate(speech_units, start=1):
                preview = (chunk[:45] + "...") if len(chunk) > 45 else chunk
                print(f'[{idx:2d}/{total}] ({len(chunk)}c) "{preview}"')
                audio_bytes = synthesize_chunk_with_retry(
                    service=service,
                    chunk=chunk,
                    voice=voice,
                    language_code=language_code,
                    sample_rate=sample_rate,
                )
                if audio_bytes:
                    out_f.writeframesraw(audio_bytes)

                if idx < total:
                    pause = (
                        paragraph_pause_bytes
                        if is_paragraph_end
                        else sentence_pause_bytes
                    )
                    out_f.writeframesraw(pause)
                    if request_delay > 0:
                        time.sleep(request_delay)

        elapsed = time.time() - start_time
        print(f"\nSynthesis completed in {elapsed:.2f}s.")
    except Exception:
        if output_path.exists():
            output_path.unlink()
        raise
