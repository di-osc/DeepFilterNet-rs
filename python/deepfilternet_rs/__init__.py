from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from deepfilternet_rs._native import DeepFilterNetRealtime as _DeepFilterNetRealtime

Float32Array = NDArray[np.float32]


def get_default_model_path() -> Path:
    """Return the bundled default DeepFilterNet model path inside the package."""
    model_resource = resources.files("deepfilternet_rs.models").joinpath(
        "DeepFilterNet3_onnx.tar.gz"
    )
    return Path(model_resource)


class DeepFilterNetRealtime:
    """Realtime-style DeepFilterNet processor for mono float32 audio.

    This wrapper resolves the bundled default model path in Python and forwards
    all processing calls to the Rust runtime.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        atten_lim: float = 100.0,
        log_level: str | None = None,
        compensate_delay: bool = True,
        post_filter_beta: float = 0.0,
        min_db_thresh: float = -15.0,
        max_db_erb_thresh: float = 35.0,
        max_db_df_thresh: float = 35.0,
    ) -> None:
        """Create a streaming DeepFilterNet processor.

        Parameters
        ----------
        model_path:
            Optional path to an official DeepFilterNet ``.tar.gz`` model. If
            omitted, the bundled package model is used. Pass an explicit path
            when you want to compare another model package or pin a specific
            model file outside the installed Python package.
        atten_lim:
            Maximum attenuation in dB. Lower values keep more of the original
            background sound and ambience, which can feel more natural but may
            leave more noise behind. Higher values allow stronger suppression.
            A value near ``100.0`` is effectively the least restrictive
            setting and usually gives the most aggressive cleanup.
        log_level:
            Compatibility logging argument passed through to the Rust binding.
            Use values like ``error``, ``warn``, ``info``, ``debug``, or
            ``trace`` when you need more backend visibility. Higher-verbosity
            values are more useful for debugging than for normal processing.
        compensate_delay:
            Whether to remove the algorithmic delay introduced by the STFT and
            model lookahead. Keep this enabled for most offline denoising so
            the output lines up better with the source. Disable it when you
            want the raw unshifted model output for debugging or alignment
            experiments.
        post_filter_beta:
            Extra post-filter strength. ``0.0`` disables the post-filter and
            is the safest starting point for natural speech. Larger values add
            more residual-noise cleanup, which can sound cleaner on noisy
            recordings but may also make speech thinner, duller, or slightly
            artifact-heavy if pushed too far.
        min_db_thresh:
            Advanced local SNR threshold for the decoder DNN path. Lower values
            keep this path active even in dirtier regions and can help with
            harder noise. Higher values make it more conservative, which may
            preserve more natural character in some cases but can also reduce
            cleanup in difficult sections.
        max_db_erb_thresh:
            Advanced upper SNR threshold for the ERB decoder path. Lower values
            make this branch stop acting earlier once the signal gets cleaner.
            Higher values keep it active across a wider SNR range. Changing it
            shifts the balance between cleanliness and naturalness, so the
            default is usually the safest choice.
        max_db_df_thresh:
            Advanced upper SNR threshold for the DF decoder path. Lower values
            restrict this path to noisier regions, while higher values keep it
            active even when the signal is already fairly clean. If pushed too
            high or too low, this can alter tone and artifact behavior in less
            obvious ways, so it is best changed only when comparing outputs
            carefully.
        """
        resolved_model_path = (
            str(get_default_model_path()) if model_path is None else str(model_path)
        )
        self._inner = _DeepFilterNetRealtime(
            model_path=resolved_model_path,
            atten_lim=atten_lim,
            log_level=log_level,
            compensate_delay=compensate_delay,
            post_filter_beta=post_filter_beta,
            min_db_thresh=min_db_thresh,
            max_db_erb_thresh=max_db_erb_thresh,
            max_db_df_thresh=max_db_df_thresh,
        )

    @property
    def sample_rate(self) -> int:
        """Backend processing sample rate in Hz."""
        return self._inner.sample_rate

    @property
    def frame_length(self) -> int:
        """Frame hop length in samples expected by :meth:`process_chunk`."""
        return self._inner.frame_length

    def process_chunk(self, audio: Float32Array) -> Float32Array:
        """Process a one-dimensional ``float32`` audio chunk."""
        return self._inner.process_chunk(audio)

    def finalize(self) -> Float32Array:
        """Flush buffered samples and close the processor."""
        return self._inner.finalize()

    def close(self) -> None:
        """Discard buffered samples and close the processor without flushing."""
        self._inner.close()


__all__ = ["DeepFilterNetRealtime", "Float32Array", "get_default_model_path"]
