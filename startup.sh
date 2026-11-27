#!/bin/bash
# Krea 2 Coloss Turbo verification container.
#
# Phase 1 (behind the tiny file server on port 3000, so Salad's /ready
# probe passes and the outside driver can follow along):
#   1. download the gated Coloss bf16 (signed URL via env COLOSS_BF16_URL)
#   2. convert it to int8_convrot with ctq (silveroxides/convert_to_quant)
#   3. delete the bf16, download the public TE + VAE from Comfy-Org/Krea-2
#   4. write PHASE1_DONE
# Phase 2: stop the file server and exec comfyui-api (the runtime image's
# normal entry), which serves /ready + /prompt on the same port.
#
# On any phase-1 failure: write PHASE1_FAIL (+ reason in phase1.log) and
# idle with the file server up; the driver sees the marker and stops the
# group. The script is idempotent: if the int8 file already exists, the
# download+convert steps are skipped (container-restart safe).
set -u
PHASE=/opt/phase1
MODELS=/opt/ComfyUI/models
DM=$MODELS/diffusion_models
BF16=$DM/krea2_coloss_turbo_bf16.safetensors
INT8=$DM/krea2_coloss_turbo_int8_convrot.safetensors
TE=$MODELS/text_encoders/qwen3vl_4b_fp8_scaled.safetensors
VAE=$MODELS/vae/qwen_image_vae.safetensors
BF16_SIZE=26283332704
INT8_MIN=13000000000
TE_SIZE=5242467968
VAE_SIZE=253806246
TE_URL="https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors"
VAE_URL="https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors"

mkdir -p "$PHASE" "$DM" "$MODELS/text_encoders" "$MODELS/vae"
exec >> "$PHASE/phase1.log" 2>&1
echo "=== startup $(date -u +%FT%TZ) ==="
df -h / | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true

python3 /app/server.py &
SRV_PID=$!
echo "file server pid $SRV_PID"
sleep 2

size_of() { stat -c%s "$1" 2>/dev/null || echo 0; }

fail() {
  echo "PHASE1_FAIL: $1"
  touch "$PHASE/PHASE1_FAIL"
  echo "idling with file server up; the driver will stop the group"
  sleep infinity
}

download() { # url dest exact_size
  local url=$1 dest=$2 want=$3 cur try code
  cur=$(size_of "$dest")
  if [ "$cur" = "$want" ]; then
    echo "already complete: $dest ($cur bytes)"
    return 0
  fi
  if [ "$cur" -gt "$want" ]; then
    echo "oversized partial, deleting: $dest ($cur > $want)"
    rm -f "$dest"
  fi
  for try in 1 2 3 4; do
    code=$(curl -L -C - --connect-timeout 30 --speed-time 120 \
           --speed-limit 10240 -o "$dest" -w "%{http_code}" "$url")
    cur=$(size_of "$dest")
    echo "download try $try: http=$code size=$cur want=$want ($dest)"
    if [ "$cur" = "$want" ]; then return 0; fi
    if [ "$code" = "403" ]; then
      echo "DOWNLOAD_403: $dest"
      return 3
    fi
    sleep 15
  done
  [ "$(size_of "$dest")" = "$want" ]
}

if [ ! -f "$PHASE/PHASE1_DONE" ]; then
  if [ "$(size_of "$INT8")" -lt "$INT8_MIN" ]; then
    if [ -z "${COLOSS_BF16_URL:-}" ]; then
      fail "COLOSS_BF16_URL env var is empty"
    fi
    download "$COLOSS_BF16_URL" "$BF16" "$BF16_SIZE" \
      || fail "bf16 download failed (rc=$?)"
    echo "bf16 downloaded: $(size_of "$BF16") bytes; converting with ctq..."
    rm -f "$INT8"
    CTQ_START=$(date +%s)
    ctq -i "$BF16" -o "$INT8" --int8 --convrot --convrot-group-size 256 \
        --scaling_mode row --comfy_quant --save-quant-metadata --krea2 \
        --simple --low-memory --device cuda > "$PHASE/ctq.log" 2>&1
    echo "ctq exit: $? after $(( $(date +%s) - CTQ_START ))s"
    tail -5 "$PHASE/ctq.log"
    if [ "$(size_of "$INT8")" -lt "$INT8_MIN" ]; then
      fail "int8 output missing/too small ($(size_of "$INT8") bytes)"
    fi
    echo "int8 size: $(size_of "$INT8") bytes"
  else
    echo "int8 already present ($(size_of "$INT8") bytes) - skipping download+convert"
  fi
  rm -f "$BF16"
  echo "bf16 deleted; disk now:"
  df -h / | tail -1
  download "$TE_URL" "$TE" "$TE_SIZE" || fail "TE download failed (rc=$?)"
  download "$VAE_URL" "$VAE" "$VAE_SIZE" || fail "VAE download failed (rc=$?)"
  touch "$PHASE/PHASE1_DONE"
  echo "PHASE1_DONE $(date -u +%FT%TZ)"
fi

echo "handing port 3000 to comfyui-api"
kill $SRV_PID 2>/dev/null
wait $SRV_PID 2>/dev/null
for i in $(seq 1 60); do
  if python3 - <<'PY'
import socket, sys
s = socket.socket(socket.AF_INET6)
try:
    s.setsockopt(socket.IPPROTO_IPV6, 26, 0)
except OSError:
    pass
try:
    s.bind(("::", 3000))
except OSError:
    sys.exit(1)
sys.exit(0)
PY
  then
    echo "port 3000 free after ${i}s"
    break
  fi
  sleep 1
done
cd /opt/ComfyUI
exec ./comfyui-api
