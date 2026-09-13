import unittest
from types import SimpleNamespace
from unittest.mock import patch

import grpc

from audiothing.synthesizer import synthesize_chunk_with_retry


class RateLimitError(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.RESOURCE_EXHAUSTED


class FakeService:
    def __init__(self):
        self.calls = 0

    def synthesize(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls < 3:
            raise RateLimitError("rate limited")
        return SimpleNamespace(audio=b"audio")


class TestSynthesizerRetry(unittest.TestCase):
    def test_rate_limit_uses_exponential_backoff(self):
        service = FakeService()
        with patch("audiothing.synthesizer.time.sleep") as sleep:
            audio = synthesize_chunk_with_retry(
                service=service,
                chunk="AI test",
                voice="test-voice",
                language_code="en-US",
                sample_rate=22050,
            )

        self.assertEqual(audio, b"audio")
        self.assertEqual(service.calls, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])


if __name__ == "__main__":
    unittest.main()
