#!/bin/bash
# Screen-record the sim-trained agent playing the real game, then export a GIF.
# Requires Screen Recording permission for the terminal app running this.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO/.venv/bin/python"
RAW=/tmp/gd_rec.mov
OUTDIR="$REPO/results"
mkdir -p "$OUTDIR"
rm -f "$RAW"

echo "[rec] starting screen recording -> $RAW"
screencapture -v "$RAW" &          # full-screen video; SIGINT stops+finalizes
REC=$!
sleep 1.5

echo "[rec] running the agent on the real game ..."
"$VENV" "$REPO/play_real_agent.py" --run "$REPO/runs/ga_stereo_open" --algo ga --episodes 2 || true

sleep 0.5
echo "[rec] stopping recording"
kill -INT "$REC" 2>/dev/null || true
wait "$REC" 2>/dev/null || true

if [ ! -s "$RAW" ]; then
  echo "[rec] NO VIDEO CAPTURED — Screen Recording permission likely not active."
  exit 1
fi
echo "[rec] captured: $(ls -lh "$RAW" | awk '{print $5}')"

echo "[rec] converting to GIF + MP4"
ffmpeg -y -loglevel error -i "$RAW" -vf "fps=20,scale=1100:-1:flags=lanczos" "$OUTDIR/real_stereo_madness.gif"
ffmpeg -y -loglevel error -i "$RAW" -vf "scale=1280:-2" -pix_fmt yuv420p "$OUTDIR/real_stereo_madness.mp4"
echo "[rec] done -> results/real_stereo_madness.{gif,mp4}"
ls -lh "$OUTDIR/real_stereo_madness.gif" "$OUTDIR/real_stereo_madness.mp4"
