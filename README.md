# Ideogram 4 Random Prompt Test

This repository records a small image-generation smoke test inspired by the prompt "a pelican riding a bicycle".

## Important Model Note

The user-provided model page is `ideogram-ai/ideogram-4-fp8`. The fp8 page says the model is gated and the public Hugging Face page currently lists no Inference Provider deployment. The official public Space linked from the model page is `ideogram-ai/ideogram4`; inspecting the Space source showed it currently runs `ideogram-ai/ideogram-4-nf4` through Gradio on ZeroGPU.

So the committed successful image artifacts are generated through the official public Ideogram 4 Space, not by downloading the gated fp8 weights locally. The batch reached the free ZeroGPU limit after 7 successful images; `results/failure-log.md` records the blocked attempts for prompts 8-10.

## Batch settings

- Space: <https://huggingface.co/spaces/ideogram-ai/ideogram4>
- API root: <https://ideogram-ai-ideogram4.hf.space/gradio_api>
- Mode: `Turbo · 12 steps`
- Upsampler: `Qwen (local)`
- Resolution: `512x512`
- Seeds: fixed, starting at `260606001`

## Run

### One prompt

Create `prompt.txt` from the example, paste your prompt into `prompt.txt`, then run:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp prompt.example.txt prompt.txt
python3 scripts/generate_prompt.py
```

If you are already inside `scripts/`, run:

```bash
python3 generate_prompt.py
```

Single-prompt outputs are written to `results/manual/`.

### Batch prompts

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python3 scripts/generate_space_batch.py
```

Generated files are written to `results/`:

- `*.webp`: generated images
- `*.caption.json`: the upsampled JSON caption fed to the model
- `summary.json`: machine-readable run summary
- `README.md`: result table
- `failure-log.md`: quota and alternate-path failure notes

## Current Result

- 10 prompts prepared in `prompts.json`
- 7 images generated successfully
- 3 prompts blocked by ZeroGPU quota / unavailable fp8 public Space / long local fp8 download
