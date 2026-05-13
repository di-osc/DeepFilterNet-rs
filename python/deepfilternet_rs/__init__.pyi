from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]


class DeepFilterNetRealtime:
    """Realtime-style DeepFilterNet processor for mono float32 audio.

    This class wraps the official DeepFilterNet Rust runtime and exposes a
    chunked Python API that accepts one-dimensional ``float32`` NumPy arrays.
    Input audio is buffered internally until full model frames are available.
    """

    def __init__(
        self,
        model_path: str | None = None,
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
            omitted, the bundled default model is used.
        atten_lim:
            Maximum attenuation in dB. Lower values preserve more original
            background sound and ambience. Higher values allow stronger noise
            suppression.
        log_level:
            Reserved compatibility argument for caller APIs. The current Rust
            binding accepts it but does not yet apply it to runtime logging.
        compensate_delay:
            Whether to remove the algorithmic delay introduced by the STFT and
            model lookahead from the emitted output.
        post_filter_beta:
            Extra post-filter strength. ``0.0`` disables the post-filter.
            Higher values can reduce residual noise more aggressively, but may
            also make speech sound thinner or introduce artifacts.
        min_db_thresh:
            Advanced local SNR threshold controlling when the decoder DNN path
            remains active in noisier regions.
        max_db_erb_thresh:
            Advanced local SNR threshold controlling the upper activity range
            of the ERB decoder path.
        max_db_df_thresh:
            Advanced local SNR threshold controlling the upper activity range
            of the DF decoder path.
        """
        ...

    @property
    def sample_rate(self) -> int:
        """Backend processing sample rate in Hz.

        Official DeepFilterNet models commonly run at ``48000`` Hz.
        """
        ...

    @property
    def frame_length(self) -> int:
        """Frame hop length in samples expected by :meth:`process_chunk`.

        Official DeepFilterNet models commonly use ``480`` samples.
        """
        ...

    def process_chunk(self, audio: Float32Array) -> Float32Array:
        """Process a one-dimensional ``float32`` audio chunk.

        The input can be shorter or longer than ``frame_length``. Partial
        frames are buffered internally until more samples arrive. The returned
        array contains any newly available enhanced samples.
        """
        ...

    def finalize(self) -> Float32Array:
        """Flush buffered samples and close the processor.

        Any remaining partial frame is zero-padded internally before the final
        enhancement step. Calling :meth:`finalize` more than once returns an
        empty array after the first close.
        """
        ...

    def close(self) -> None:
        """Discard buffered samples and close the processor without flushing."""
        ...
