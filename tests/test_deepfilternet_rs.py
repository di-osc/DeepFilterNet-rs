from __future__ import annotations

import wave
from pathlib import Path
import subprocess

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


def test_deepfilternet_realtime_accepts_snr_threshold_kwargs() -> None:
    processor = DeepFilterNetRealtime(
        min_db_thresh=-12.0,
        max_db_erb_thresh=30.0,
        max_db_df_thresh=28.0,
    )

    assert processor.sample_rate == 48000
    assert processor.frame_length == 480
    processor.close()


def test_cli_parser_defaults() -> None:
    from deepfilternet_rs.cli import build_parser

    args = build_parser().parse_args(["input.wav", "output.wav"])

    assert args.input_path == Path("input.wav")
    assert args.output_path == Path("output.wav")
    assert args.model_path is None
    assert args.atten_lim == 100.0
    assert args.log_level == "warn"
    assert args.compensate_delay is True
    assert args.post_filter_beta == 0.0
    assert args.min_db_thresh == -15.0
    assert args.max_db_erb_thresh == 35.0
    assert args.max_db_df_thresh == 35.0


def test_cli_processes_wav_with_runtime_defaults(
    tmp_path: Path, monkeypatch
) -> None:
    from deepfilternet_rs import cli

    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"

    samples = np.array([0, 1000, -1000, 2000, -2000, 500], dtype=np.int16)
    with wave.open(str(input_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(48000)
        wav_file.writeframes(samples.tobytes())

    captured: dict[str, object] = {}

    class FakeProcessor:
        sample_rate = 48000
        frame_length = 480

        def __init__(
            self,
            model_path,
            atten_lim,
            log_level,
            compensate_delay,
            post_filter_beta,
            min_db_thresh,
            max_db_erb_thresh,
            max_db_df_thresh,
        ) -> None:
            captured["init"] = {
                "model_path": model_path,
                "atten_lim": atten_lim,
                "log_level": log_level,
                "compensate_delay": compensate_delay,
                "post_filter_beta": post_filter_beta,
                "min_db_thresh": min_db_thresh,
                "max_db_erb_thresh": max_db_erb_thresh,
                "max_db_df_thresh": max_db_df_thresh,
            }

        def process_chunk(self, audio: np.ndarray) -> np.ndarray:
            captured["process_chunk"] = audio.copy()
            return audio * 0.5

        def finalize(self) -> np.ndarray:
            captured["finalize"] = True
            return np.array([], dtype=np.float32)

    monkeypatch.setattr(cli, "DeepFilterNetRealtime", FakeProcessor)

    exit_code = cli.main([str(input_path), str(output_path)])

    assert exit_code == 0
    assert captured["init"] == {
        "model_path": None,
        "atten_lim": 100.0,
        "log_level": "warn",
        "compensate_delay": True,
        "post_filter_beta": 0.0,
        "min_db_thresh": -15.0,
        "max_db_erb_thresh": 35.0,
        "max_db_df_thresh": 35.0,
    }
    np.testing.assert_allclose(
        captured["process_chunk"],
        samples.astype(np.float32) / np.iinfo(np.int16).max,
        rtol=1e-6,
        atol=2e-6,
    )
    assert captured["finalize"] is True

    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 48000
        output_samples = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)

    expected = np.round(samples.astype(np.float32) * 0.5).astype(np.int16)
    np.testing.assert_array_equal(output_samples, expected)


def test_cli_uses_ffmpeg_for_decode_resample_and_encode(
    tmp_path: Path, monkeypatch
) -> None:
    from deepfilternet_rs import cli

    input_path = tmp_path / "input.mp3"
    output_path = tmp_path / "output.flac"
    input_path.write_bytes(b"fake")

    captured: dict[str, object] = {"commands": []}
    decoded = np.array([0.0, 0.25, -0.25, 0.5], dtype=np.float32)

    class FakeProcessor:
        sample_rate = 48000
        frame_length = 480

        def __init__(
            self,
            model_path,
            atten_lim,
            log_level,
            compensate_delay,
            post_filter_beta,
            min_db_thresh,
            max_db_erb_thresh,
            max_db_df_thresh,
        ) -> None:
            captured["init"] = {
                "model_path": model_path,
                "atten_lim": atten_lim,
                "log_level": log_level,
                "compensate_delay": compensate_delay,
                "post_filter_beta": post_filter_beta,
                "min_db_thresh": min_db_thresh,
                "max_db_erb_thresh": max_db_erb_thresh,
                "max_db_df_thresh": max_db_df_thresh,
            }

        def process_chunk(self, audio: np.ndarray) -> np.ndarray:
            captured["process_chunk"] = audio.copy()
            return audio * 0.5

        def finalize(self) -> np.ndarray:
            return np.array([0.125], dtype=np.float32)

    def fake_run(*args, **kwargs):
        cmd = list(args[0])
        captured["commands"].append(cmd)
        if "-f" in cmd and "f32le" in cmd and "pipe:1" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=decoded.tobytes())
        if "-f" in cmd and "f32le" in cmd and "pipe:0" in cmd:
            captured["encoded_bytes"] = kwargs["input"]
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(cli, "DeepFilterNetRealtime", FakeProcessor)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    exit_code = cli.main([str(input_path), str(output_path)])

    assert exit_code == 0
    assert len(captured["commands"]) == 2
    decode_cmd = captured["commands"][0]
    assert Path(decode_cmd[0]).name == "ffmpeg"
    assert decode_cmd[1:3] == ["-v", "error"]
    assert decode_cmd[-1] == "pipe:1"
    assert "-i" in decode_cmd and str(input_path) in decode_cmd
    assert "-ac" in decode_cmd and decode_cmd[decode_cmd.index("-ac") + 1] == "1"
    assert "-ar" in decode_cmd and decode_cmd[decode_cmd.index("-ar") + 1] == "48000"

    encode_cmd = captured["commands"][1]
    assert Path(encode_cmd[0]).name == "ffmpeg"
    assert encode_cmd[1:3] == ["-v", "error"]
    assert encode_cmd[-1] == str(output_path)
    assert "-i" in encode_cmd and "pipe:0" in encode_cmd
    assert "-ar" in encode_cmd and encode_cmd[encode_cmd.index("-ar") + 1] == "48000"

    np.testing.assert_allclose(captured["process_chunk"], decoded, rtol=1e-6, atol=1e-6)
    encoded = np.frombuffer(captured["encoded_bytes"], dtype=np.float32)
    np.testing.assert_allclose(encoded, np.array([0.0, 0.125, -0.125, 0.25, 0.125], dtype=np.float32))


def test_cli_passes_custom_snr_thresholds(monkeypatch, tmp_path: Path) -> None:
    from deepfilternet_rs import cli

    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    input_path.write_bytes(b"fake")

    captured: dict[str, object] = {}

    class FakeProcessor:
        sample_rate = 48000
        frame_length = 480

        def __init__(
            self,
            model_path,
            atten_lim,
            log_level,
            compensate_delay,
            post_filter_beta,
            min_db_thresh,
            max_db_erb_thresh,
            max_db_df_thresh,
        ) -> None:
            captured["init"] = {
                "min_db_thresh": min_db_thresh,
                "max_db_erb_thresh": max_db_erb_thresh,
                "max_db_df_thresh": max_db_df_thresh,
            }

        def process_chunk(self, audio: np.ndarray) -> np.ndarray:
            return audio

        def finalize(self) -> np.ndarray:
            return np.array([], dtype=np.float32)

    monkeypatch.setattr(cli, "DeepFilterNetRealtime", FakeProcessor)
    monkeypatch.setattr(cli, "_decode_audio", lambda path, sample_rate: np.zeros(4, dtype=np.float32))
    monkeypatch.setattr(cli, "_encode_audio", lambda path, audio, sample_rate: None)

    exit_code = cli.main(
        [
            str(input_path),
            str(output_path),
            "--min-db-thresh",
            "-10.0",
            "--max-db-erb-thresh",
            "22.5",
            "--max-db-df-thresh",
            "18.0",
        ]
    )

    assert exit_code == 0
    assert captured["init"] == {
        "min_db_thresh": -10.0,
        "max_db_erb_thresh": 22.5,
        "max_db_df_thresh": 18.0,
    }
