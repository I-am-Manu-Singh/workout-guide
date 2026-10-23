#!/usr/bin/env python3
"""
Create a smoother looping exercise GIF/WebP from three PNG poses using RifeMetal.

For a bench press with these files:
  03.png = bar lowered near chest
  02.png = middle
  01.png = arms extended

the animation order is 03 -> 02 -> 01, then reverses 01 -> 02 -> 03.
RifeMetal generates multiple in-between frames for EACH adjacent pair.

Install Python dependencies:
  python3 -m pip install pillow numpy opencv-python

Example:
  python3 exercise_pose_interpolator.py \
    --input ./bench_press \
    --output ./output/bench_press \
    --rife ~/Downloads/rife-metal-macos-arm64/bin/rife-metal \
    --model ~/Downloads/rife-metal-macos-arm64/share/rife-metal/rife-v4.25.rmw \
    --order 03.png 02.png 01.png \
    --steps 5 --fps 10

RifeMetal CLI supports:
  rife-metal -0 a.png -1 b.png --output prefix.png --model model.rmw \
    --timesteps 0.2,0.4,0.6,0.8
which creates prefix_t0.2.png, prefix_t0.4.png, etc.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def blacken_flat_edge_background(image: Image.Image, tolerance: int) -> Image.Image:
    """Turn only a flat, edge-connected background into black."""
    rgba = np.asarray(image.convert("RGBA")).copy()
    rgb = rgba[:, :, :3].astype(np.int16)
    h, w = rgb.shape[:2]
    corners = np.array(
        [rgb[0, 0], rgb[0, w - 1], rgb[h - 1, 0], rgb[h - 1, w - 1]],
        dtype=np.float32,
    )
    bg = np.median(corners, axis=0)
    distance = np.max(np.abs(rgb - bg), axis=2)
    candidate = (distance <= tolerance).astype(np.uint8)

    count, labels = cv2.connectedComponents(candidate, connectivity=8)
    if count > 1:
        border_labels = np.unique(np.concatenate([
            labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
        ]))
        mask = np.isin(labels, border_labels) & (candidate == 1)
        rgba[mask, :3] = 0
        rgba[mask, 3] = 255

    # Composite transparency on black.
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    out = rgba[:, :, :3].astype(np.float32) * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def load_pose(path: Path, size: tuple[int, int] | None, tolerance: int) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    if size and image.size != size:
        image.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", size, (0, 0, 0, 255))
        canvas.alpha_composite(
            image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
        )
        image = canvas
    return blacken_flat_edge_background(image, tolerance)


def interpolate_pair(executable: Path, model: Path, first: Path, second: Path,
                     output_prefix: Path, timesteps: list[float]) -> list[Path]:
    import re

    timestep_arg = ",".join(f"{t:.4f}".rstrip("0").rstrip(".")
                            for t in timesteps)
    cmd = [
        str(executable),
        "-0", str(first),
        "-1", str(second),
        "--output", str(output_prefix),
        "--model", str(model),
        "--timesteps", timestep_arg,
    ]
    print(f"Interpolating {first.name} -> {second.name} at {timestep_arg}")
    subprocess.run(cmd, check=True)

    # RifeMetal rounds output timestamps, e.g. 0.166667 -> 0.1667.
    # Discover files and sort numerically instead of expecting exact float strings.
    pattern = re.compile(
        re.escape(output_prefix.stem) + r"_t([0-9]+(?:\.[0-9]+)?)" +
        re.escape(output_prefix.suffix) + r"$"
    )
    found: list[tuple[float, Path]] = []
    for path in output_prefix.parent.glob(output_prefix.stem + "_t*.png"):
        match = pattern.match(path.name)
        if match:
            found.append((float(match.group(1)), path))

    found.sort(key=lambda item: item[0])
    if len(found) != len(timesteps):
        nearby = sorted(output_prefix.parent.glob(output_prefix.stem + "*"))
        raise RuntimeError(
            f"Expected {len(timesteps)} interpolated frames, found {len(found)}. "
            "Files created: " + ", ".join(p.name for p in nearby)
        )
    return [path for _, path in found]


def export_animation(frames: list[Image.Image], output: Path, fps: int,
                     write_gif: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(1, round(1000 / fps))

    webp = output.with_suffix(".webp")
    frames[0].save(
        webp, format="WEBP", save_all=True, append_images=frames[1:],
        duration=duration, loop=0, lossless=True, method=6
    )
    print(f"Saved animated WebP: {webp}")

    if write_gif:
        gif = output.with_suffix(".gif")
        palette_frames = [
            f.convert("RGB").quantize(
                colors=256, method=Image.Quantize.MEDIANCUT
            ) for f in frames
        ]
        palette_frames[0].save(
            gif, format="GIF", save_all=True, append_images=palette_frames[1:],
            duration=duration, loop=0, optimize=True, disposal=2
        )
        print(f"Saved animated GIF: {gif}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Output filename without extension")
    parser.add_argument("--rife", type=Path, required=True,
                        help="Path to RifeMetal executable")
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to .rmw model file")
    parser.add_argument("--order", nargs=3, default=["03.png", "02.png", "01.png"],
                        metavar=("BOTTOM", "MIDDLE", "TOP"))
    parser.add_argument("--steps", type=int, default=5,
                        help="Number of in-between frames per adjacent pose pair (default: 5)")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--bg-tolerance", type=int, default=35)
    parser.add_argument("--no-gif", action="store_true",
                        help="Export animated WebP only")
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.fps < 1:
        parser.error("--fps must be at least 1")
    if not args.rife.is_file():
        parser.error(f"RifeMetal executable not found: {args.rife}")
    if not args.model.is_file():
        parser.error(f"Model file not found: {args.model}")

    pose_paths = [args.input / name for name in args.order]
    missing = [str(p) for p in pose_paths if not p.is_file()]
    if missing:
        parser.error("Missing pose file(s): " + ", ".join(missing))

    first = Image.open(pose_paths[0])
    size = first.size
    poses = [load_pose(p, size, args.bg_tolerance) for p in pose_paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Evenly spaced interpolation times, excluding endpoints because the source
    # poses themselves are included in the sequence.
    timesteps = [i / (args.steps + 1) for i in range(1, args.steps + 1)]

    with tempfile.TemporaryDirectory(prefix="rife_metal_") as temp:
        temp_dir = Path(temp)
        normalized_paths: list[Path] = []
        for i, pose in enumerate(poses):
            path = temp_dir / f"pose_{i}.png"
            pose.save(path)
            normalized_paths.append(path)

        forward: list[Image.Image] = [poses[0]]
        for pair_index in range(2):
            prefix = temp_dir / f"pair_{pair_index}.png"
            interpolated_paths = interpolate_pair(
                args.rife, args.model,
                normalized_paths[pair_index],
                normalized_paths[pair_index + 1],
                prefix, timesteps
            )
            for frame_path in interpolated_paths:
                with Image.open(frame_path) as frame:
                    forward.append(blacken_flat_edge_background(
                        frame.convert("RGBA"), args.bg_tolerance
                    ))
            forward.append(poses[pair_index + 1])

    # Smooth ping-pong loop: bottom -> middle -> top -> middle -> bottom.
    # Exclude duplicate endpoints from the reversed half.
    loop_frames = forward + forward[-2:0:-1]

    export_animation(loop_frames, args.output, args.fps, not args.no_gif)


if __name__ == "__main__":
    main()
