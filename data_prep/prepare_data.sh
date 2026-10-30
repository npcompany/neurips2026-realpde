#!/bin/bash
# Build the organizers' /data layout on a pod: train_sim/, train_real/ without 7575_0.h5, and the
# official mean_std_real.pt. Downloads the two public tarballs of the Track-1 Drive folder
# (the ids scripts/fetch_data.sh uses), resumable.
#
#   final/pod/prepare_data.sh /workspace/data
#
# Needs ~33 GB free (16 GB of tarballs, deleted after extraction, + 16 GB extracted).
set -euo pipefail
DEST=${1:?destination directory, e.g. /workspace/data}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DEST"

dl() {  # dl <file_id> <dest_path> <expected_bytes>
  local id="$1" dest="$2" want="$3"
  if [[ -f "$dest" && "$(stat -c%s "$dest")" == "$want" ]]; then
    echo "SKIP  $dest"; return 0
  fi
  if [[ -f "$dest" ]] && head -c 512 "$dest" | grep -qi "<html\|<!doctype"; then rm -f "$dest"; fi
  echo "GET   $dest ($want bytes)"
  curl -sL --retry 5 --retry-delay 10 -C - \
    "https://drive.usercontent.google.com/download?id=${id}&export=download&confirm=t" -o "$dest"
  [[ "$(stat -c%s "$dest")" == "$want" ]] || { echo "FAILED $dest (size mismatch; Drive quota?)"; exit 1; }
}

extract() {  # extract <tarball> <dirname>: tolerate a tarball with or without the top-level dir
  local tgz="$1" name="$2"
  if [[ -d "$DEST/$name" && "$(ls "$DEST/$name"/*.h5 2>/dev/null | wc -l)" -gt 50 ]]; then
    echo "SKIP  extract $name"; return 0
  fi
  local tmp="$DEST/.extract_$name"
  rm -rf "$tmp"; mkdir -p "$tmp"
  tar xzf "$tgz" -C "$tmp"
  rm -rf "$DEST/$name"
  if [[ -d "$tmp/$name" ]]; then mv "$tmp/$name" "$DEST/$name"; else mv "$tmp" "$DEST/$name"; fi
  rm -rf "$tmp"
}

dl 1zeya9aoDcoWUe91ew4aAUHBjLsWC-2kk "$DEST/train_real.tar.gz" 7393582393
dl 1GjaI_ZNDU1ApRAqXvil4GJNk-3Cp07XA "$DEST/train_sim.tar.gz" 8826593178
extract "$DEST/train_real.tar.gz" train_real
extract "$DEST/train_sim.tar.gz" train_sim
rm -f "$DEST/train_real/7575_0.h5"          # the organizers' /data does not have it
cp "$HERE/mean_std_real.pt" "$DEST/mean_std_real.pt"
rm -f "$DEST/train_real.tar.gz" "$DEST/train_sim.tar.gz"
echo "train_real: $(ls "$DEST/train_real"/*.h5 | wc -l) files (expect 81)"
echo "train_sim : $(ls "$DEST/train_sim"/*.h5 | wc -l) files (expect 100)"
ls -la "$DEST/mean_std_real.pt"
df -h "$DEST" | tail -1
