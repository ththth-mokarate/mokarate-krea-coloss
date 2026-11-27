# mokarate-krea-coloss

SaladCloud image for verifying **Krea 2 Coloss Turbo**
(colosslabs/Krea-2-Coloss-Turbo, a photoreal/NSFW fine-tune of Krea 2
Turbo) with an in-session BF16 -> INT8 ConvRot conversion.

- Base: `ghcr.io/ththth-mokarate/mokarate-comfyui-kromoss:runtime`
  (ComfyUI 0.46.0 + comfyui-api 1.19.2).
- Adds `ctq` (silveroxides/convert_to_quant, PyPI
  `convert-to-quant==1.3.4`).
- `CMD ["/app/startup.sh"]` (no ENTRYPOINT - Salad pattern):
  1. A tiny dual-stack file server (`/app/server.py`) owns port 3000 so
     the Salad readiness probe passes during phase 1 and the outside
     driver can follow `/phase1.log`, `/ctq.log` and the `PHASE1_*`
     markers.
  2. Downloads the gated Coloss BF16 (signed URL passed via the
     `COLOSS_BF16_URL` container env var), converts it with
     `ctq --int8 --convrot --convrot-group-size 256 --scaling_mode row
     --comfy_quant --save-quant-metadata --krea2 --simple --low-memory
     --device cuda`, deletes the BF16, then downloads the public
     `qwen3vl_4b_fp8_scaled` text encoder and `qwen_image_vae` from
     Comfy-Org/Krea-2.
  3. Stops the file server and `exec`s `./comfyui-api` for the test
     generations.
