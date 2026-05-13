from __future__ import annotations

import numpy as np

from deepfilternet_rs import DeepFilterNetRealtime


def test_deepfilternet_realtime_smoke() -> None:
    processor = DeepFilterNetRealtime(
        model_path=None,
        atten_lim=100.0,
        log_level="warn",
        compensate_delay=True,
        post_filter_beta=0.0,
    )

    assert processor.sample_rate == 48000
    assert processor.frame_length == 480

    audio = np.zeros(processor.frame_length * 4, dtype=np.float32)
    enhanced = processor.process_chunk(audio)
    final = processor.finalize()

    assert enhanced.dtype == np.float32
    assert final.dtype == np.float32
    assert enhanced.ndim == 1
    assert final.ndim == 1
    assert enhanced.size + final.size <= audio.size
