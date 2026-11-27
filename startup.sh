#!/bin/bash
# Krea 2 Coloss Turbo verification container.
#
# Phase 1 (behind the tiny file server on port 3000, so Salad's /ready
# probe passes and the outside driver can follow along):
#   0. verify the mounted disk really has room (64 GB storage raise)
#   1. download the gated Coloss bf16 (signed URL via env COLOSS_BF16_URL)
#   2. convert it to int8_convrot with ctq, then GATE on: ctq exit 0,
#      output size in the 12.5-15 GB band, and the .comfy_quant metadata
#      (format=int8_tensorwise, convrot, per_row, group size 256);
#      write manifest.txt next to the logs
#   3. expose the int8 via the file server (INT8_READY) so the driver
#      can preserve it on the VM; KEEP the bf16 (T0 reference generation)
#   4. download the public TE + VAE from Comfy-Org/Krea-2
#   5. write PHASE1_DONE and wait for the driver's GET /release signal
# Phase 2: stop the file server and exec comfyui-api (the runtime
# image's normal entry), which serves /ready + /prompt on the same port.
#
# On any phase-1 failure: write PHASE1_FAIL (+ reason in phase1.log)
# and idle with the file server up; the driver sees the marker and
# stops the group. Idempotent: an in-band int8 file skips straight to
# the metadata gate (container-restart safe).
set -u
PHASE=/opt/phase1
MODELS=/opt/ComfyUI/models
DM=$MODELS/diffusion_models
BF16=$DM/krea2_coloss_turbo_bf16.safetensors
INT8=$DM/krea2_coloss_turbo_int8_convrot.safetensors
TE=$MODELS/text_encoders/qwen3vl_4b_fp8_scaled.safetensors
VAE=$MODELS/vae/qwen_image_vae.safetensors
BF16_SIZE=26283332704
INT8_LO=12500000000
INT8_HI=15000000000
TE_SIZE=5242467968
VAE_SIZE=253806246
TE_URL="https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors"
VAE_URL="https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors"
MIN_FREE=47000000000

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
mark() { echo "### $1 $(date -u +%FT%TZ)${2:+ $2}"; }

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

gate_quant_meta() { # .comfy_quant metadata gate on $INT8
  python3 - "$INT8" <<'PY'
import json, sys
from safetensors import safe_open
path = sys.argv[1]
found = {}
def scan(obj, depth=0):
    if depth > 4:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("format", "convrot", "per_row", "convrot_groupsize") \
                    and k not in found:
                found[k] = v
            scan(v, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:4]:
            scan(v, depth + 1)
raw = None
with safe_open(path, framework="pt") as f:
    md = f.metadata() or {}
    if "comfy_quant" in md:
        raw = md["comfy_quant"]
    keys = list(f.keys())
    if raw is None:
        for k in keys:
            if "comfy_quant" in k:
                t = f.get_tensor(k)
                raw = bytes(t.numpy().tobytes()).decode("utf-8", "replace")
                break
print(f"GATE3 tensors: {len(keys)}; first keys: {keys[:3]}")
if raw is None:
    print("GATE3 FAIL: no comfy_quant metadata found")
    sys.exit(1)
try:
    data = json.loads(raw)
except Exception as e:
    print("GATE3 FAIL: comfy_quant is not JSON:", e)
    sys.exit(1)
scan(data)
print("GATE3 found:", found)
ok = (found.get("format") == "int8_tensorwise"
      and found.get("convrot") in (True, "true")
      and found.get("per_row") in (True, "true")
      and str(found.get("convrot_groupsize")) == "256")
print("GATE3", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
}

# --- disk gate: verify the mounted volume BEFORE downloading ---------
FREE=$(df -B1 / | awk 'NR==2 {print $4}')
echo "disk free on /: $FREE bytes (need >= $MIN_FREE)"
if [ "$FREE" -lt "$MIN_FREE" ]; then
  fail "insufficient disk: $FREE free < $MIN_FREE required (64GB storage raise missing?)"
fi
mark DISK_OK "free=$FREE"

if [ ! -f "$PHASE/PHASE1_DONE" ]; then
  ISZ=$(size_of "$INT8")
  if [ "$ISZ" -lt "$INT8_LO" ] || [ "$ISZ" -gt "$INT8_HI" ]; then
    if [ -z "${COLOSS_BF16_URL:-}" ]; then
      fail "COLOSS_BF16_URL env var is empty"
    fi
    mark DL_BF16_START "want=$BF16_SIZE"
    download "$COLOSS_BF16_URL" "$BF16" "$BF16_SIZE" \
      || fail "bf16 download failed (rc=$?)"
    mark DL_BF16_END "bytes=$(size_of "$BF16")"
    mark CONV_START
    rm -f "$INT8"
    CTQ_START=$(date +%s)
    ctq -i "$BF16" -o "$INT8" --int8 --convrot --convrot-group-size 256 \
        --scaling_mode row --comfy_quant --save-quant-metadata --krea2 \
        --simple --low-memory --device cuda > "$PHASE/ctq.log" 2>&1
    CTQ_RC=$?
    echo "ctq exit: $CTQ_RC after $(( $(date +%s) - CTQ_START ))s"
    tail -5 "$PHASE/ctq.log"
    [ "$CTQ_RC" = "0" ] || fail "ctq exited $CTQ_RC (see ctq.log)"
    gate_quant_meta || fail "comfy_quant metadata gate failed (see GATE3 lines)"
    ISZ=$(size_of "$INT8")
    if [ "$ISZ" -lt "$INT8_LO" ] || [ "$ISZ" -gt "$INT8_HI" ]; then
      fail "int8 size $ISZ outside expected band [$INT8_LO, $INT8_HI]"
    fi
    echo "GATE size: PASS ($ISZ bytes in band)"
    CTQ_VER=$(python3 -c "import importlib.metadata as m; print(m.version('convert-to-quant'))" 2>/dev/null || echo unknown)
    {
      echo "tool: ctq (convert-to-quant) $CTQ_VER"
      echo "command: ctq -i $BF16 -o $INT8 --int8 --convrot --convrot-group-size 256 --scaling_mode row --comfy_quant --save-quant-metadata --krea2 --simple --low-memory --device cuda"
      echo "input_bf16_bytes: $(size_of "$BF16")"
      echo "output_int8_bytes: $ISZ"
      echo "converted_at: $(date -u +%FT%TZ)"
    } > "$PHASE/manifest.txt"
    cat "$PHASE/manifest.txt"
    mark CONV_END "int8_bytes=$ISZ"
  else
    echo "int8 already present ($ISZ bytes, in band) - re-checking metadata gate"
    gate_quant_meta || fail "comfy_quant metadata gate failed (pre-existing int8)"
  fi
  # Expose the converted file through the file server so the driver can
  # download and preserve it on the VM before this disk disappears.
  # The bf16 is KEPT: it is the T0 reference for the fidelity comparison
  # (64 GB storage leaves room; the disk is ephemeral anyway).
  ln -sf "$INT8" "$PHASE/krea2_coloss_turbo_int8_convrot.safetensors"
  echo "INT8_READY size=$(size_of "$INT8")"
  download "$TE_URL" "$TE" "$TE_SIZE" || fail "TE download failed (rc=$?)"
  mark DL_TE_END "bytes=$(size_of "$TE")"
  download "$VAE_URL" "$VAE" "$VAE_SIZE" || fail "VAE download failed (rc=$?)"
  mark DL_VAE_END "bytes=$(size_of "$VAE")"
  touch "$PHASE/PHASE1_DONE"
  echo "PHASE1_DONE $(date -u +%FT%TZ)"
  # Hold the file server up until the driver has preserved the int8
  # file and signals via GET /release (which drops a RELEASE flag file
  # next to the logs). Cap the wait so a dead driver cannot block the
  # test generations forever.
  echo "waiting for driver RELEASE (cap 75 min)"
  for i in $(seq 1 450); do
    if [ -f "$PHASE/RELEASE" ]; then
      echo "RELEASE received after ~$((i * 10))s"
      break
    fi
    sleep 10
  done
  if [ ! -f "$PHASE/RELEASE" ]; then
    echo "RELEASE wait cap reached - proceeding without it"
  fi
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
