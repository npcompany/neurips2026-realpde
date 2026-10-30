#!/bin/bash
# Serialise GPU work on the shared T4.
#
#   scripts/gpu.sh <log-name> <command...>
#
# Two agents share one 15 GB T4 and an FNO fine-tune at batch 8 does not leave
# room for a second one. `flock` on a repo-local file makes every job that goes
# through here queue instead of collide; a job launched any other way is
# invisible to it, which is why the wait also polls for a training process and
# for free memory.
#
# Output is tee'd to runs/logs/<log-name>.log so a background chain can be
# followed after the fact -- the harness's own capture ends with the tool call.
set -u
cd "$(dirname "$0")/.."

NAME="${1:?usage: gpu.sh <log-name> <command...>}"
shift

LOCK=.gpu.lock
LOGDIR=runs/logs
mkdir -p "$LOGDIR"
LOG="$LOGDIR/$NAME.log"

#: Refuse to start below this many MiB free. An FNO fine-tune at batch 8 peaks
#: near 6 GB, so 8 GB is the smallest headroom that is not a gamble.
MIN_FREE_MIB=${MIN_FREE_MIB:-8000}

#: And below this much *disk*. An FNO run writes last.pt and best.pt at ~384 MB each,
#: and a torch.save that runs out of space leaves a truncated checkpoint behind rather
#: than failing cleanly -- which is then loaded by whatever runs next. Refuse up front.
MIN_FREE_DISK_MIB=${MIN_FREE_DISK_MIB:-1500}

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
free_disk_mib() { df -Pm . | awk 'NR==2 {print $4}'; }

# Python block-buffers stdout when it is a pipe, so a `tee`d log stays empty until
# the job exits -- which defeats the point of having a log for a 1-hour run.
export PYTHONUNBUFFERED=1

# Volunteer as the OOM killer's first choice. The guard here is for *host* RAM, not the
# GPU: 15.8 GB and no swap means an over-allocating job reaches the global OOM killer,
# which then picks a victim by heuristic -- measured 2026-08-11, it took the interactive
# tmux session four times. `ulimit -v` is the better guard but is unusable alongside CUDA
# (the driver reserves tens of GB of address space), so GPU jobs get this instead. See
# scripts/run.sh for the numpy-only path.
echo 900 >/proc/self/oom_score_adj 2>/dev/null || true

{
    echo "=== $NAME queued at $(date -u +%FT%TZ) ==="
    echo "=== cmd: $* ==="
} >>"$LOG"

# Checked *before* the lock: it does not depend on the GPU, and a job that is going to
# refuse should not first sit in the queue for an hour to do it.
if [ "$(free_disk_mib)" -lt "$MIN_FREE_DISK_MIB" ]; then
    echo "REFUSING $NAME: only $(free_disk_mib) MiB disk free, need $MIN_FREE_DISK_MIB." \
        "Prune runs/ -- best.pt is ~384 MB per run and PROTOCOL records it is NOT better" \
        "than last.pt on the official scorer." | tee -a "$LOG"
    exit 1
fi

# flock holds the whole job, so the next queued one starts when this exits.
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "waiting for the GPU lock (held by another job)..." | tee -a "$LOG"
    flock 9
fi

# The lock only covers jobs that use it. Anything the other agent launches
# directly is caught here instead.
while [ "$(free_mib)" -lt "$MIN_FREE_MIB" ]; do
    echo "$(date -u +%T) only $(free_mib) MiB free, need $MIN_FREE_MIB -- waiting" | tee -a "$LOG"
    sleep 60
done

echo "=== $NAME started at $(date -u +%FT%TZ), $(free_mib) MiB GPU / $(free_disk_mib) MiB disk free ===" | tee -a "$LOG"
start=$(date +%s)
"$@" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
echo "=== $NAME exit=$status after $((($(date +%s) - start) / 60)) min ===" | tee -a "$LOG"
exit "$status"
