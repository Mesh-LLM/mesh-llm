#!/usr/bin/env bash
for i in $(seq 1 30); do
  OUT=$(gcloud compute ssh sb-controller --zone us-central1-a --command "pgrep -c curl 2>/dev/null || echo 0; stat -c%s ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf 2>/dev/null || echo 0" 2>/dev/null)
  CURLS=$(echo "$OUT" | head -1); SIZE=$(echo "$OUT" | tail -1)
  echo "$(date +%H:%M:%S) curl_procs=$CURLS size=$SIZE"
  if [ "$CURLS" = "0" ] && [ "$SIZE" -gt 15000000000 ]; then echo "MODEL_READY size=$SIZE"; exit 0; fi
  if [ "$CURLS" = "0" ] && [ "$SIZE" -lt 15000000000 ] && [ "$i" -gt 2 ]; then echo "DOWNLOAD_DIED size=$SIZE"; exit 1; fi
  sleep 60
done
echo "DOWNLOAD_TIMEOUT"; exit 1
