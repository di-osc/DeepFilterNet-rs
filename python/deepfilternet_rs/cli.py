from __future__ import annotations

from argparse import BooleanOptionalAction
from pathlib import Path
import shutil
import subprocess

import numpy as np
from jsonargparse import ArgumentParser

from deepfilternet_rs import DeepFilterNetRealtime, get_default_model_path


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="deepfilternet",
        description="Denoise an audio file with DeepFilterNet via ffmpeg I/O.",
    )
    parser.add_argument("input_path", type=Path, help="Input audio file path.")
    parser.add_argument("output_path", type=Path, help="Output audio file path.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Optional DeepFilterNet model archive path. Leave unset to use the bundled "
            "default model. Use this when you want to switch to another official model "
            "package without changing code."
        ),
    )
    parser.add_argument(
        "--atten-lim",
        type=float,
        default=100.0,
        help=(
            "Maximum attenuation in dB. Lower values keep more background sound and "
            "usually preserve ambience better, but may leave more noise. Higher values "
            "allow stronger suppression. 100.0 is effectively no extra attenuation cap "
            "and is the most aggressive option among typical settings."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="warn",
        help=(
            "Backend log level passed through to the runtime, for example error, warn, "
            "info, debug, or trace. Higher-verbosity levels like debug and trace are "
            "useful when diagnosing model or ffmpeg issues, while warn is a quieter "
            "default for normal batch use."
        ),
    )
    parser.add_argument(
        "--compensate-delay",
        action=BooleanOptionalAction,
        default=True,
        help=(
            "Whether to remove the model's algorithmic delay from the beginning of the "
            "output. Keep this enabled for most file-based denoising so timing stays "
            "closer to the source. Disable it if you need the raw unshifted model output "
            "for debugging or exact streaming alignment experiments."
        ),
    )
    parser.add_argument(
        "--post-filter-beta",
        type=float,
        default=0.0,
        help=(
            "Extra post-filter strength. 0.0 disables the post-filter and is usually the "
            "safest choice for natural speech. Higher values apply stronger residual-noise "
            "reduction, which can sound cleaner in noisy recordings but may also make "
            "speech thinner or introduce artifacts if pushed too far."
        ),
    )
    parser.add_argument(
        "--min-db-thresh",
        type=float,
        default=-15.0,
        help=(
            "Advanced: minimum local SNR threshold for running the decoder DNN path. "
            "Lower values make the model stay active even in dirtier regions and can help "
            "with harder noise, but may increase processing on very noisy material. Higher "
            "values make it more conservative. Unless you are doing detailed tuning, keep "
            "the default."
        ),
    )
    parser.add_argument(
        "--max-db-erb-thresh",
        type=float,
        default=35.0,
        help=(
            "Advanced: maximum local SNR threshold for the ERB decoder path. Lower values "
            "make this branch stop acting earlier on cleaner regions. Higher values keep it "
            "active across a wider SNR range. Changing this can shift the balance between "
            "cleanliness and naturalness, so the default is usually best."
        ),
    )
    parser.add_argument(
        "--max-db-df-thresh",
        type=float,
        default=35.0,
        help=(
            "Advanced: maximum local SNR threshold for the DF decoder path. Lower values "
            "restrict this path to noisier regions; higher values let it remain active even "
            "when the signal is already fairly clean. If pushed too high or too low, this can "
            "change tone and artifact behavior in non-obvious ways, so prefer the default "
            "unless you are comparing outputs carefully."
        ),
    )
    return parser


def _ensure_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for CLI audio decoding/encoding but was not found")
    return ffmpeg


def _ffmpeg_log_level(log_level: str) -> str:
    normalized = log_level.lower()
    if normalized == "warn":
        return "warning"
    if normalized in {"off", "error", "warning", "info", "debug", "trace"}:
        return normalized
    if normalized == "verbose":
        return "verbose"
    return "error"


def _decode_audio(path: Path, sample_rate: int, log_level: str) -> np.ndarray:
    ffmpeg = _ensure_ffmpeg()
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            _ffmpeg_log_level(log_level),
            "-i",
            str(path),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def _encode_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    ffmpeg_log_level = "error"
    return _encode_audio_with_log_level(path, audio, sample_rate, ffmpeg_log_level)


def _encode_audio_with_log_level(
    path: Path, audio: np.ndarray, sample_rate: int, log_level: str
) -> None:
    ffmpeg = _ensure_ffmpeg()
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-v",
            _ffmpeg_log_level(log_level),
            "-y",
            "-f",
            "f32le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            str(path),
        ],
        check=True,
        input=np.asarray(audio, dtype=np.float32).tobytes(),
        capture_output=True,
    )


def denoise_audio(
    input_path: Path,
    output_path: Path,
    *,
    model_path: Path | None,
    atten_lim: float,
    log_level: str,
    compensate_delay: bool,
    post_filter_beta: float,
    min_db_thresh: float,
    max_db_erb_thresh: float,
    max_db_df_thresh: float,
) -> None:
    processor = DeepFilterNetRealtime(
        model_path=str(model_path) if model_path is not None else str(get_default_model_path()),
        atten_lim=atten_lim,
        log_level=log_level,
        compensate_delay=compensate_delay,
        post_filter_beta=post_filter_beta,
        min_db_thresh=min_db_thresh,
        max_db_erb_thresh=max_db_erb_thresh,
        max_db_df_thresh=max_db_df_thresh,
    )
    audio = _decode_audio(input_path, processor.sample_rate, log_level)
    enhanced = processor.process_chunk(audio)
    tail = processor.finalize()
    output_audio = np.concatenate([enhanced, tail]).astype(np.float32, copy=False)
    _encode_audio_with_log_level(output_path, output_audio, processor.sample_rate, log_level)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    denoise_audio(
        args.input_path,
        args.output_path,
        model_path=args.model_path,
        atten_lim=args.atten_lim,
        log_level=args.log_level,
        compensate_delay=args.compensate_delay,
        post_filter_beta=args.post_filter_beta,
        min_db_thresh=args.min_db_thresh,
        max_db_erb_thresh=args.max_db_erb_thresh,
        max_db_df_thresh=args.max_db_df_thresh,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
