# audiothing

Text-to-Speech (TTS) generation tool powered by NVIDIA Riva / NVCF Chatterbox-Multilingual.

## Features

- Automatic speech-optimized text sanitization: strips quotes (`"`, `'`), backticks (`` ` ``), hyphens (`-`), underscores (`_`), and markdown artifacts so TTS models don't pronounce punctuation literally.
- Optional experimental initialism pronunciation: spells out standalone two- and three-letter uppercase groups such as `AI` and `API` as `ay eye` and `ay pee eye`, without adding pause punctuation.
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
  --max-chars COUNT     Maximum characters per speech chunk (default: 220)
  --request-delay SEC   Delay between synthesis requests (default: 0.50)
  --no-sanitize         Disable automatic text sanitization
  --pronounce-initialisms
                        Experimentally spell out standalone uppercase initialisms of 2-3 letters
  --a-b                 Generate three audio files: original, word classifier, and phonetic signs
  --phonetic-initialisms
                        Use IPA-style phonetic signs for initialisms
  --text-only, --dry-run
                        Print processed text and skip all audio generation
```

The initialism option is disabled by default. It is intentionally conservative:
one-letter tokens, common uppercase words, mixed-case names, and groups longer
than three letters are left unchanged.

Use `--a-b` to generate a directly comparable three-way test:

```text
audio_1_without_classifier.wav
audio_1_with_classifier.wav
audio_1_with_phonetic_signs.wav
```

Use `--text-only` to inspect the exact text sent to the voice model without
requiring an API key or generating audio. Combine it with `--a-b` to print
both versions.
