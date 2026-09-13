# audiothing

Text-to-Speech (TTS) generation tool powered by NVIDIA Riva / NVCF Chatterbox-Multilingual.

## Features

- Automatic chunking respecting sentence and clause boundaries within Triton model token limits.
- Natural speech pauses between sentences and paragraphs.
- Non-destructive sequential audio file naming (`audio_1.wav`, `audio_2.wav`, etc.).
- Automatic environment variable loading from `.env`.

## Setup

1. Install dependencies using `uv` or `pip`:
   ```bash
   uv sync
   # or
   pip install -e .
   ```

2. Copy the example environment configuration and add your NVIDIA API key:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to set your `NVCF_API_KEY`:
   ```ini
   NVCF_API_KEY=your_nvidia_api_key_here
   ```

## Usage

Place text you want to convert to speech in `input.txt` (or specify a custom file with `--input`):

```bash
uv run python main.py
```

### Options

```
options:
  -h, --help            show this help message and exit
  -i, --input INPUT     Path to the input text file (default: input.txt)
  -o, --output-dir DIR  Directory to save generated audio files (default: output)
  --server SERVER       Riva gRPC server URI (default: grpc.nvcf.nvidia.com:443)
  --function-id ID      NVCF Function ID
  --api-key KEY         NVCF API key / token
  --voice VOICE         Voice name (default: Chatterbox-Multilingual.en-US.Male)
  --language-code CODE  Language code (default: en-US)
  --sample-rate RATE    Sample rate in Hz (default: 22050)
  --sentence-pause SEC  Pause between sentences in seconds (default: 0.25)
  --paragraph-pause SEC Pause between paragraphs in seconds (default: 0.50)
```
