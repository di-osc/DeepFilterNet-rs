# DeepFilterNet-rs

Python bindings for the official DeepFilterNet Rust realtime runtime.

This package exposes a small PyO3 wrapper around DeepFilterNet's Rust `DfTract`
streaming runtime. It is intended for realtime audio enhancement pipelines that
need a Python API without shelling out to the `deep-filter` binary.

The required DeepFilterNet Rust runtime source is inlined under
`rust/src/deep_filter` from upstream `Rikorose/DeepFilterNet` tag `v0.5.6`.
The bundled default model now ships as Python package data under
`python/deepfilternet_rs/models`. Builds do not depend on the upstream Git
repository at compile time.

The package also includes a small `deepfilternet` command line tool for
denoising common audio files through `ffmpeg`-based decode and encode steps.

## Install

```bash
pip install deepfilternet-rs
```

When installing from source, a Rust toolchain is required because the package is
built with `maturin`. Prebuilt wheels do not require Rust on the target machine.

## Usage

`DeepFilterNetRealtime` is a streaming API. It does not read files for you, and
it expects already-decoded mono audio samples as a one-dimensional
`numpy.float32` array.

Important input requirements:

- Sample rate must match `processor.sample_rate`. Official bundled models use `48000 Hz`.
- Audio must be mono. If your source is stereo or multichannel, downmix it before calling `process_chunk`.
- Samples should be normalized float audio, typically in the `[-1.0, 1.0]` range.
- Input dtype should be `np.float32`. Convert explicitly if your upstream decoder returns `int16`, `float64`, or another dtype.
- You can pass any chunk length to `process_chunk()`. It does not need to equal `frame_length`, but `frame_length` is still useful as a natural streaming block size.

```python
import numpy as np
from deepfilternet_rs import DeepFilterNetRealtime

processor = DeepFilterNetRealtime(
    model_path=None,
    atten_lim=100.0,
    log_level="warn",
    compensate_delay=True,
    post_filter_beta=0.0,
)

audio = np.zeros(processor.frame_length, dtype=np.float32)
enhanced = processor.process_chunk(audio)
tail = processor.finalize()
```

When `model_path=None`, the Python package automatically resolves the bundled
default model file from `deepfilternet_rs.models` and passes its path into the
Rust runtime.

### Streaming Example

The example below shows a more realistic streaming loop:

```python
from pathlib import Path
import wave

import numpy as np

from deepfilternet_rs import DeepFilterNetRealtime


def read_wav_mono_48k(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()

        if channels != 1:
            raise ValueError(f"expected mono wav, got {channels} channels")
        if sample_rate != 48000:
            raise ValueError(f"expected 48000 Hz wav, got {sample_rate} Hz")
        if sample_width != 2:
            raise ValueError(f"expected 16-bit PCM wav, got {sample_width * 8}-bit")

        pcm = np.frombuffer(
            wav_file.readframes(wav_file.getnframes()),
            dtype=np.int16,
        )
    return pcm.astype(np.float32) / np.iinfo(np.int16).max


def write_wav_mono_48k(path: str | Path, audio: np.ndarray) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = np.round(pcm * np.iinfo(np.int16).max).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(48000)
        wav_file.writeframes(pcm.tobytes())


processor = DeepFilterNetRealtime(
    model_path=None,
    atten_lim=100.0,
    log_level="warn",
    compensate_delay=True,
    post_filter_beta=0.0,
)

audio = read_wav_mono_48k("input.wav")
chunk_size = processor.frame_length * 10
chunks: list[np.ndarray] = []

for start in range(0, len(audio), chunk_size):
    chunk = audio[start : start + chunk_size].astype(np.float32, copy=False)
    enhanced = processor.process_chunk(chunk)
    if enhanced.size:
        chunks.append(enhanced)

tail = processor.finalize()
if tail.size:
    chunks.append(tail)

output = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
write_wav_mono_48k("output.wav", output)
```

Notes about streaming behavior:

- `process_chunk()` may return fewer samples than you pass in at the beginning because the model has internal buffering and optional delay compensation.
- `finalize()` is required at the end of a stream. It flushes the remaining buffered audio and returns the tail that has not been emitted yet.
- If you skip `finalize()`, you will usually lose the last buffered part of the stream.
- If you do not want the tail and want to abort a stream early, call `close()` instead.
- For non-WAV inputs, or for automatic resampling/downmixing, prefer the `deepfilternet input.xxx output.yyy` CLI, which uses `ffmpeg`.

## CLI

```bash
deepfilternet input.wav output.wav
```

The CLI accepts common input formats that `ffmpeg` can decode, such as WAV,
FLAC, MP3, and M4A. Audio is automatically converted to mono 48kHz before
running the DeepFilterNet model, and the output format follows the destination
file extension.

Optional runtime parameters mirror the Python API defaults:

```bash
deepfilternet input.wav output.wav \
  --model-path custom_model.tar.gz \
  --atten-lim 100.0 \
  --log-level warn \
  --compensate-delay \
  --post-filter-beta 0.0 \
  --min-db-thresh -15.0 \
  --max-db-erb-thresh 35.0 \
  --max-db-df-thresh 35.0
```

Parameter tuning notes:

- `--atten-lim`: controls how aggressively extra attenuation is allowed. Lower values usually keep more room tone and background ambience, but noise may remain more obvious. Higher values push stronger suppression. `100.0` is the least restrictive default.
- `--post-filter-beta`: controls extra residual-noise cleanup after the main model pass. `0.0` is the most natural starting point. Raising it can make noisy clips sound cleaner, but if it is too high, speech may become thinner, duller, or slightly watery.
- `--compensate-delay`: enabled by default and recommended for normal file output. It removes the model delay so the result lines up better with the source timing. Turning it off is mostly useful for debugging or low-level alignment experiments.
- `--log-level`: now controls both the Rust runtime logger and `ffmpeg -v ...`. Use `warn` or `error` for normal runs. Use `info` or `debug` when you want more diagnostics. `trace` is intentionally very noisy and may print large amounts of model-loading and graph-parsing output.
- `--model-path`: leave unset to use the bundled default model. Set it when you want to test or ship a different official DeepFilterNet model package.
- `--min-db-thresh`, `--max-db-erb-thresh`, `--max-db-df-thresh`: advanced SNR threshold parameters passed directly to the underlying DeepFilterNet runtime. They affect when different decoder paths remain active across noisier or cleaner regions. These are useful for controlled experiments and model tuning, but they are much easier to mis-tune than `--atten-lim` or `--post-filter-beta`, so most users should leave them at their defaults.

Current CLI constraints:

- `ffmpeg` must be installed and available on `PATH`.
- Input and output formats are limited to what the local `ffmpeg` build
  supports.
- The model still runs internally at mono 48000 Hz.

## API

### `DeepFilterNetRealtime`

Constructor arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `model_path` | `str | None` | `None` | Optional official DeepFilterNet `.tar.gz` model path. `None` uses the bundled default model from the Python package resources. |
| `atten_lim` | `float` | `100.0` | Attenuation limit in dB. `100.0` means no explicit limit. |
| `log_level` | `str | None` | `None` | Runtime log verbosity. Values like `error`, `warn`, `info`, `debug`, and `trace` affect Rust-side logging. |
| `compensate_delay` | `bool` | `True` | Drop initial algorithmic-delay samples from output. |
| `post_filter_beta` | `float` | `0.0` | Post-filter beta. `0.0` disables the post-filter. |
| `min_db_thresh` | `float` | `-15.0` | Advanced local SNR threshold for the decoder DNN path. |
| `max_db_erb_thresh` | `float` | `35.0` | Advanced upper SNR threshold for the ERB decoder path. |
| `max_db_df_thresh` | `float` | `35.0` | Advanced upper SNR threshold for the DF decoder path. |

Properties:

| Property | Description |
|---|---|
| `sample_rate` | Backend processing sample rate. Official DeepFilterNet models use 48000 Hz. |
| `frame_length` | Frame length in samples. Official DeepFilterNet models use 480 samples. |

Methods:

| Method | Description |
|---|---|
| `process_chunk(audio)` | Process a one-dimensional `float32` numpy array and return enhanced `float32` samples. The input can be any length; incomplete frames are buffered. |
| `finalize()` | Flush buffered samples with zero padding and close the processor. |
| `close()` | Clear buffers and close the processor without flushing. |

Practical streaming notes:

- `process_chunk(audio)` expects mono `np.float32` samples at exactly `sample_rate`.
- If your input is not already `48000 Hz`, resample it before using the Python streaming API.
- If your input is stereo, downmix it before calling `process_chunk(audio)`.
- `frame_length` is a good default chunk size, but larger multiples such as `frame_length * 10` are also fine.
- The output length from `process_chunk(audio)` is not guaranteed to equal the input length for every call, especially at the beginning of a stream when delay compensation is enabled.

## Release

Publishing is handled by GitHub Actions. Create a GitHub release or run the
release workflow manually after configuring the `PYPI_API_TOKEN` repository
secret.
