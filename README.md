# DeepFilterNet-rs

Python bindings for the official DeepFilterNet Rust realtime runtime.

This package exposes a small PyO3 wrapper around DeepFilterNet's Rust `DfTract`
streaming runtime. It is intended for realtime audio enhancement pipelines that
need a Python API without shelling out to the `deep-filter` binary.

The required DeepFilterNet Rust runtime source is inlined under
`rust/src/deep_filter` from upstream `Rikorose/DeepFilterNet` tag `v0.5.6`,
together with the bundled `DeepFilterNet3_onnx.tar.gz` default model under
`rust/models`. Builds do not depend on the upstream Git repository at compile
time.

The package also includes a small `deepfilternet` command line tool for
denoising common audio files through `ffmpeg`-based decode and encode steps.

## Install

```bash
pip install deepfilternet-rs
```

When installing from source, a Rust toolchain is required because the package is
built with `maturin`. Prebuilt wheels do not require Rust on the target machine.

## Usage

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
- `--log-level`: use `warn` or `error` for normal runs. Use `info`, `debug`, or `trace` when you want to inspect backend behavior and troubleshoot model or ffmpeg issues.
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
| `model_path` | `str | None` | `None` | Optional official DeepFilterNet `.tar.gz` model path. `None` uses the bundled default model from the Rust crate. |
| `atten_lim` | `float` | `100.0` | Attenuation limit in dB. `100.0` means no explicit limit. |
| `log_level` | `str | None` | `None` | Reserved for compatibility with existing callers. |
| `compensate_delay` | `bool` | `True` | Drop initial algorithmic-delay samples from output. |
| `post_filter_beta` | `float` | `0.0` | Post-filter beta. `0.0` disables the post-filter. |

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

## Release

Publishing is handled by GitHub Actions. Create a GitHub release or run the
release workflow manually after configuring the `PYPI_API_TOKEN` repository
secret.
