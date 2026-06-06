#!/usr/bin/env python3
"""Generate a small Ideogram 4 prompt batch through the public HF Space API."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import get_token


SPACE_URL = "https://ideogram-ai-ideogram4.hf.space"
POST_URL = f"{SPACE_URL}/gradio_api/call/v2/generate"
EVENT_URL = f"{SPACE_URL}/gradio_api/call/generate"


def auth_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def post_job(payload: dict[str, Any], headers: dict[str, str]) -> str:
    response = requests.post(POST_URL, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    event_id = response.json()["event_id"]
    return event_id


def wait_for_result(event_id: str, timeout: int, headers: dict[str, str]) -> list[Any]:
    response = requests.get(f"{EVENT_URL}/{event_id}", headers=headers, stream=True, timeout=timeout)
    response.raise_for_status()
    event_name = None
    data_lines: list[str] = []
    start = time.monotonic()

    for raw in response.iter_lines(decode_unicode=True):
        if time.monotonic() - start > timeout:
            raise TimeoutError(f"Timed out waiting for event {event_id}")
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            if event_name == "complete" and data_lines:
                return json.loads("\n".join(data_lines))
            if event_name == "error" and data_lines:
                raise RuntimeError("\n".join(data_lines))
            event_name = None
            data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())

    raise RuntimeError(f"No complete event received for {event_id}")


def download_file(url: str, output_path: Path, headers: dict[str, str]) -> None:
    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def run_batch(args: argparse.Namespace) -> None:
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = auth_headers()

    results: list[dict[str, Any]] = []
    for idx, item in enumerate(prompts, start=1):
        seed = args.seed_base + idx - 1
        image_path = out_dir / f"{item['id']}.webp"
        caption_path = out_dir / f"{item['id']}.caption.json"
        if image_path.exists() and caption_path.exists():
            print(f"[{idx:02d}/{len(prompts):02d}] {item['id']} already exists; reusing", flush=True)
            results.append(
                {
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "seed": seed,
                    "mode": args.mode,
                    "upsampler": args.upsampler,
                    "width": args.width,
                    "height": args.height,
                    "elapsed_seconds": None,
                    "image": str(image_path.relative_to(out_dir.parent)),
                    "caption": str(caption_path.relative_to(out_dir.parent)),
                    "event_id": None,
                }
            )
            continue
        payload = {
            "prompt": item["prompt"],
            "mode": args.mode,
            "upsampler": args.upsampler,
            "width": args.width,
            "height": args.height,
            "seed": seed,
            "randomize_seed": False,
        }
        print(f"[{idx:02d}/{len(prompts):02d}] {item['id']} seed={seed}", flush=True)
        started = time.time()
        event_id = post_job(payload, headers)
        image_data, returned_seed, caption = wait_for_result(event_id, args.timeout, headers)
        image_url = image_data["url"]
        download_file(image_url, image_path, headers)
        caption_path.write_text(
            json.dumps(caption, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        elapsed = round(time.time() - started, 2)
        results.append(
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "seed": returned_seed,
                "mode": args.mode,
                "upsampler": args.upsampler,
                "width": args.width,
                "height": args.height,
                "elapsed_seconds": elapsed,
                "image": str(image_path.relative_to(out_dir.parent)),
                "caption": str(caption_path.relative_to(out_dir.parent)),
                "event_id": event_id,
            }
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Ideogram 4 Random Prompt Batch Results",
        "",
        f"- Space: {SPACE_URL}",
        "- Space model observed in app.py: `ideogram-ai/ideogram-4-nf4`",
        "- User-requested model page: `ideogram-ai/ideogram-4-fp8`",
        f"- Mode: `{args.mode}`",
        f"- Upsampler: `{args.upsampler}`",
        f"- Resolution: `{args.width}x{args.height}`",
        "",
        "| # | Prompt | Seed | Time | Image | Caption |",
        "|---|---|---:|---:|---|---|",
    ]
    for idx, row in enumerate(results, start=1):
        lines.append(
            f"| {idx} | {row['prompt']} | {row['seed']} | {row['elapsed_seconds']}s | "
            f"[webp]({row['image']}) | [json]({row['caption']}) |"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="prompts.json")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--mode", default="Turbo · 12 steps")
    parser.add_argument("--upsampler", default="Qwen (local)")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed-base", type=int, default=260606001)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    run_batch(parse_args())
