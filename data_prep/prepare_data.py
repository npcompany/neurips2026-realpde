"""Build the organizers' /data layout with the standard library only (the training images have
no curl/wget/git): train_sim/, train_real/ without 7575_0.h5, mean_std_real.pt (the kit's copy,
taken from the image's own vendor/ tree).

    python prepare_data.py /vol/data          # ~16 GB extracted, ~20-30 min

Resumable: a tarball with the right size is not downloaded again, an extracted set is kept.
"""
import os, shutil, sys, tarfile, time, urllib.request

DEST = sys.argv[1] if len(sys.argv) > 1 else "/vol/data"
FILES = [  # (Drive file id, name, bytes) -- the public Track-1 folder, as scripts/fetch_data.sh
    ("1zeya9aoDcoWUe91ew4aAUHBjLsWC-2kk", "train_real", 7393582393),
    ("1GjaI_ZNDU1ApRAqXvil4GJNk-3Cp07XA", "train_sim", 8826593178),
]
MEAN_STD = "/workspace/vendor/realpde_t2_starting_kit_v6/example_data/mean_std_real.pt"
os.makedirs(DEST, exist_ok=True)


def download(fid, path, want):
    if os.path.exists(path) and os.path.getsize(path) == want:
        print("SKIP ", path); return
    url = f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        head = r.read(1 << 20)
        if b"<html" in head[:512].lower() or b"<!doctype" in head[:512].lower():
            sys.exit(f"FAILED {path}: Drive answered an HTML page (quota or consent); retry later")
        f.write(head); n = len(head)
        while True:
            b = r.read(1 << 24)
            if not b: break
            f.write(b); n += len(b)
            if n % (1 << 30) < (1 << 24): print(f"  {n / 1e9:.1f} GB  {n / 1e6 / (time.time() - t0):.0f} MB/s", flush=True)
    got = os.path.getsize(path)
    if got != want: sys.exit(f"FAILED {path}: {got} bytes, expected {want}")
    print(f"GOT  {path} in {(time.time() - t0) / 60:.1f} min")


def extract(tgz, name):
    out = os.path.join(DEST, name)
    if os.path.isdir(out) and len([f for f in os.listdir(out) if f.endswith(".h5")]) > 50:
        print("SKIP  extract", name); return
    tmp = os.path.join(DEST, ".extract_" + name)
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    with tarfile.open(tgz) as t: t.extractall(tmp)
    shutil.rmtree(out, ignore_errors=True)
    inner = os.path.join(tmp, name)
    os.rename(inner if os.path.isdir(inner) else tmp, out)
    shutil.rmtree(tmp, ignore_errors=True)
    print("EXTRACTED", out)


for fid, name, want in FILES:
    tgz = os.path.join(DEST, name + ".tar.gz")
    download(fid, tgz, want)
    extract(tgz, name)
    os.remove(tgz)
bad = os.path.join(DEST, "train_real", "7575_0.h5")
if os.path.exists(bad): os.remove(bad)          # the organizers' /data does not have it
shutil.copyfile(MEAN_STD, os.path.join(DEST, "mean_std_real.pt"))   # copyfile: network volumes refuse chmod
for name, n in (("train_real", 81), ("train_sim", 100)):
    k = len([f for f in os.listdir(os.path.join(DEST, name)) if f.endswith(".h5")])
    print(f"{name}: {k} files (expect {n})")
print(os.listdir(DEST), shutil.disk_usage(DEST).free // 2**30, "GiB free")
