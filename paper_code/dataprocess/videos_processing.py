from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import cv2
from tqdm import tqdm


PathLike = Union[str, Path]


def extract_video_frames(
    videos_dir: PathLike,
    output_root: Optional[PathLike] = None,
    video_glob: str = "*.mp4",
    zero_padding: int = 5,
    show_progress: bool = True,
) -> Dict[Path, int]:
    """
    Extract PNG frames from a directory of videos.

    For each video found under ``videos_dir`` (filtered by ``video_glob``),
    a subdirectory named after the video file (without extension) is created
    under ``output_root``. Every frame is written there as a zero-padded PNG,
    e.g. ``00000.png``.

    Args:
        videos_dir: Directory that stores the input video files.
        output_root: Directory to write extracted frames. Defaults to ``videos_dir``.
        video_glob: Glob pattern that selects which videos to process.
        zero_padding: Number of digits to use for the frame filenames.
        show_progress: If True, display tqdm progress bars.

    Returns:
        Dictionary mapping each processed video path to the number of frames extracted.

    Raises:
        FileNotFoundError: If ``videos_dir`` does not exist.
        RuntimeError: If a video cannot be opened for reading.
    """

    videos_dir = Path(videos_dir)
    if not videos_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {videos_dir}")

    output_root = Path(output_root) if output_root is not None else videos_dir
    output_root.mkdir(parents=True, exist_ok=True)

    processed: Dict[Path, int] = {}
    video_paths = sorted(videos_dir.glob(video_glob))
    with tqdm(
        video_paths,
        desc="Videos",
        unit="video",
        disable=not show_progress,
    ) as video_iter:
        for video_path in video_iter:
            if video_path.is_dir():
                continue

            frames_dir = output_root / video_path.stem
            frames_dir.mkdir(parents=True, exist_ok=True)

            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"Failed to open video file: {video_path}")

            frame_count = 0
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            with tqdm(
                total=total_frames if total_frames > 0 else None,
                desc=video_path.stem,
                unit="frame",
                leave=False,
                disable=not show_progress,
            ) as frame_pbar:
                try:
                    while True:
                        success, frame = capture.read()
                        if not success:
                            break

                        frame_path = frames_dir / f"{frame_count:0{zero_padding}d}.png"
                        if not cv2.imwrite(str(frame_path), frame):
                            raise RuntimeError(f"Failed to write frame to {frame_path}")

                        frame_pbar.update(1)
                        frame_count += 1
                finally:
                    capture.release()
            processed[video_path] = frame_count

    return processed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract frames from videos into per-video folders."
    )
    parser.add_argument(
        "videos_dir",
        type=Path,
        help="Directory that contains input video files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory to store extracted frames. Defaults to the input directory.",
    )
    parser.add_argument(
        "--glob",
        dest="video_glob",
        default="*.mp4",
        help="Glob pattern for selecting videos (default: *.mp4).",
    )
    parser.add_argument(
        "--zero-padding",
        type=int,
        default=5,
        help="Digits used when naming frame files (default: 5).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        processed = extract_video_frames(
            videos_dir=args.videos_dir,
            output_root=args.output_root,
            video_glob=args.video_glob,
            zero_padding=args.zero_padding,
            show_progress=not args.no_progress,
        )
    except Exception as exc:  # pragma: no cover - CLI safety net
        print(f"[videos_processing] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    for video_path, frame_count in processed.items():
        print(f"{video_path} -> {frame_count} frames")


if __name__ == "__main__":
    main()
