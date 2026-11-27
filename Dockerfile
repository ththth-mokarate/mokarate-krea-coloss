# Krea 2 Coloss Turbo verification image for SaladCloud.
#
# Layers the ctq converter (silveroxides/convert_to_quant, PyPI
# convert-to-quant==1.3.4 - flags verified against the wheel: --krea2,
# --int8 --convrot --convrot-group-size 256, --scaling_mode row,
# --comfy_quant, --save-quant-metadata, --simple, --low-memory,
# --device cuda) on top of the proven Kromoss runtime image
# (ComfyUI 0.46.0 + comfyui-api 1.19.2, torch 2.13/CUDA 13.0).
# The container's CMD runs /app/startup.sh: phase 1 downloads the gated
# Coloss bf16 (signed URL via env), converts it to int8_convrot, fetches
# the public TE/VAE, then execs comfyui-api for the test generations.
FROM ghcr.io/ththth-mokarate/mokarate-comfyui-kromoss:runtime

ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN pip install --no-cache-dir "convert-to-quant==1.3.4"

COPY server.py /app/server.py
COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

EXPOSE 3000
CMD ["/app/startup.sh"]
