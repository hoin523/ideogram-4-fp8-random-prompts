#!/usr/bin/env python3
"""Generate one image from prompt.txt.

This is the convenient single-prompt entrypoint. It works from the repo root:

    python3 scripts/generate_prompt.py

It also works from inside scripts/:

    python3 generate_prompt.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from generate_space_batch import SpaceError, auth_headers, download_file, post_job, wait_for_result


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_prompt(path: Path) -> str:
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise SystemExit(f"{path} is empty. Paste one prompt into it first.")
    if prompt.startswith("Paste one prompt here"):
        raise SystemExit(f"{path} still contains the placeholder text. Replace it with your prompt.")
    return prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", default=str(REPO_ROOT / "prompt.txt"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "manual"))
    parser.add_argument("--mode", default="Turbo · 12 steps")
    parser.add_argument("--upsampler", default="Qwen (local)")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=int(time.time()) % (2**31 - 1))
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_path = Path(args.prompt_file).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = read_prompt(prompt_path)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    image_path = out_dir / f"{run_id}.webp"
    caption_path = out_dir / f"{run_id}.caption.json"

    payload = {
        "prompt": prompt,
        "mode": args.mode,
        "upsampler": args.upsampler,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "randomize_seed": False,
    }
    headers = auth_headers()
    started = time.time()
    print(f"Generating {args.width}x{args.height} seed={args.seed}", flush=True)
    event_id = post_job(payload, headers)
    image_data, returned_seed, caption = wait_for_result(event_id, args.timeout, headers)
    download_file(image_data["url"], image_path, headers)
    caption_path.write_text(json.dumps(caption, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elapsed = round(time.time() - started, 2)

    summary = {
        "prompt": prompt,
        "seed": returned_seed,
        "mode": args.mode,
        "upsampler": args.upsampler,
        "width": args.width,
        "height": args.height,
        "elapsed_seconds": elapsed,
        "image": str(image_path),
        "caption": str(caption_path),
        "event_id": event_id,
    }
    summary_path = out_dir / f"{run_id}.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved image: {image_path}")
    print(f"Saved caption: {caption_path}")
    print(f"Elapsed: {elapsed}s")


if __name__ == "__main__":
    try:
        main()
    except SpaceError as exc:
        print("")
        print("Generation failed.")
        print(f"Reason: {exc}")
        if exc.payload and "ZeroGPU quota" in str(exc.payload.get("error", "")):
            print("This is Hugging Face ZeroGPU quota, not a Python setup problem.")
            print("Wait until the retry time shown above, or use a paid/pro quota.")
        raise SystemExit(1)
