# Failure Log

This run attempted 10 prompts.

## What succeeded

Prompts 1-7 were generated through the official Hugging Face Space:

- Space: `ideogram-ai/ideogram4`
- Observed backend in Space source: `ideogram-ai/ideogram-4-nf4`
- Mode: `Turbo · 12 steps`
- Upsampler: `Qwen (local)`
- Resolution: `512x512`

## What blocked prompts 8-10

1. The official Space exhausted ZeroGPU quota after 7 successful generations.

   Error observed:

   ```text
   You have exceeded your free ZeroGPU quota (120s requested vs. 55s left).
   Try again in 23:58:14.
   ```

2. The alternate public fp8 Space `jing96963/ideoooo` exposes `ideogram-ai/ideogram-4-fp8`, but its public API returned:

   ```text
   event: error
   data: null
   ```

   Its README says the Space requires a secret named `HF_TOKEN`, so the public instance appears unable to access the gated model at runtime.

3. Local fp8 execution was installed and started with the official `ideogram-oss/ideogram4` code. Access to the gated model succeeded after passing the local HF token, but the fp8 checkpoint is about 25.7 GB. Download was started into a workspace cache and then stopped because finishing the download plus local MPS generation would take too long for this run.

## Re-run options

After ZeroGPU quota resets, run:

```bash
python scripts/generate_space_batch.py
```

The script resumes existing files and will continue from missing prompt outputs.
