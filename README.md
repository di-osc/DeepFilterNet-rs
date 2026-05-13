# DeepFilterNet-rs

Python bindings for the official DeepFilterNet Rust realtime runtime.

This package exposes a small PyO3 wrapper around DeepFilterNet's Rust `DfTract`
streaming runtime. It is intended for realtime audio enhancement pipelines that
need a Python API without shelling out to the `deep-filter` binary.

The required DeepFilterNet Rust crate source is vendored under `vendor/libDF`
from upstream `Rikorose/DeepFilterNet` tag `v0.5.6`, together with the bundled
`DeepFilterNet3_onnx.tar.gz` default model under `vendor/models`. Builds do not
depend on the upstream Git repository at compile time.

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
