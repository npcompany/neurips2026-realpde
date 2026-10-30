#!/bin/bash
# Assemble the training package of one track into dist/<track>/ and optionally build its image.
#
#   ./build.sh track1 [IMAGE_TAG]
#   ./build.sh track2 [IMAGE_TAG]
#
# dist/<track>/ is the directory we submitted for the Decision phase: the shared library (src/),
# the orchestration (rpde_final/), the scripts the recipes invoke, the organizers' kit files listed
# in vendor_manifest.txt, and the track's solution.py, Dockerfile, README.md and METHOD.md (the
# submitted packages also carried their training logs, which this repository leaves out).
# The kit files are not in this repository: unpack the official starting kits into vendor/ first
# (see README.md).
set -euo pipefail
cd "$(dirname "$0")"
TRACK=${1:?track1|track2}
TAG=${2:-}
DIST=dist/$TRACK
rm -rf "$DIST"
mkdir -p "$DIST/scripts"
EXCL=(--exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' --exclude '*.egg-info')

rsync -a "${EXCL[@]}" src "$DIST/"
while read -r f; do
    [ -f "$f" ] || { echo "missing organizer-kit file: $f (unpack the starting kits into vendor/, see README.md)"; exit 1; }
done < vendor_manifest.txt
rsync -a --files-from=vendor_manifest.txt . "$DIST/"

# scripts invoked by rpde_final/recipes.py (Runtime.script / Runtime.run)
COMMON_SCRIPTS=(
    gpu.sh                              # resolved by clean_bundle.py (unused: --no-gpu-lock)
    gen/build_resampled_sim.py          # the sim_units resample
    sps/sps_cache.py sps/sps_apply.py sps/sps_assemble.py   # interval heads, bundle assembly
)
T1_SCRIPTS=(
    ship/clean_bundle.py ship/t1_nowarm_log.py ship/t1_graph_bundle.py ship/t1_hostmem_bundle.py
    eval/t1_daemon_verify.py
)
T2_SCRIPTS=(
    t2/t2_stream_cache.py t2/t2_recal_lab.py t2/t2_autocal_lab.py
    ship/clean_bundle.py t2/t2_daemon_verify.py
)
SCRIPTS=("${COMMON_SCRIPTS[@]}")
if [ "$TRACK" = track1 ]; then SCRIPTS+=("${T1_SCRIPTS[@]}"); else SCRIPTS+=("${T2_SCRIPTS[@]}"); fi
for s in "${SCRIPTS[@]}"; do
    mkdir -p "$DIST/scripts/$(dirname "$s")"
    cp "scripts/$s" "$DIST/scripts/$s"
done

rsync -a "${EXCL[@]}" rpde_final "$DIST/"
cp ruff.toml "$DIST/ruff.toml"
cp "$TRACK/solution.py" "$TRACK/Dockerfile" "$TRACK/README.md" "$TRACK/METHOD.md" "$DIST/"
echo "package: $DIST ($(du -sh "$DIST" | cut -f1), $(find "$DIST" -type f | wc -l) files)"
if [ -n "$TAG" ]; then
    docker build --platform linux/amd64 -t "$TAG" "$DIST"
    echo "image: $TAG"
fi
