"""
Code Blue ECG — Fast CPU-Only Rare-Class PhysioNet Downloader & Indexer (~3-4 min, ~$0.02)
Runs in parallel while v7 trains on GPU, caching verified rare-class 12-lead WFDB datasets into Modal Volume:
  1) Brugada-HUCA v2.0.0 (PhysioNet, 7.3 MB): strictly basal_pattern in {1, 2} (visible Brugada ECGs) + controls
  2) Chapman-Shaoxing + Ningbo ecg-arrhythmia v1.0.0 (PhysioNet, 45,152 12-lead ECGs):
     extracts verified SNOMED-CT cases for our 4 under-powered classes:
       - HIGH_GRADE_AV_BLOCK (2nd-deg Mobitz II / 3rd-deg Complete AVB)
       - ATRIAL_FLUTTER
       - WPW_PREEXCITATION
       - PACEMAKER_RHYTHM
Saves `/vol/rare_physionet/rare_manifest.json` ready for instant v8 GPU training right after v7 finishes!
"""
import modal

app = modal.App("codeblue-cache-rare-physionet")
vol = modal.Volume.from_name("codeblue-ecg-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "unzip")
    .pip_install("wfdb", "physionet", "pandas", "numpy")
)

# Verified PhysioNet Challenge 2021 / SNOMED-CT codes for our 4 weak rare classes
SNOMED_TO_RARE = {
    "27885002": "HIGH_GRADE_AV_BLOCK",   # Complete / 3rd-degree AV block
    "28189009": "HIGH_GRADE_AV_BLOCK",   # Mobitz type II AV block
    "195042002": "HIGH_GRADE_AV_BLOCK",  # Second-degree AV block
    "164890007": "ATRIAL_FLUTTER",       # Atrial flutter
    "74390002": "WPW_PREEXCITATION",     # Wolff-Parkinson-White pattern
    "195060002": "WPW_PREEXCITATION",    # Ventricular pre-excitation
    "10370003": "PACEMAKER_RHYTHM",      # Pacing rhythm
}


@app.function(
    image=image,
    cpu=4.0,
    memory=8192,
    timeout=3600,
    volumes={"/vol": vol},
)
def cache_rare_physionet():
    import json, os, subprocess, time
    import pandas as pd

    t0 = time.time()
    BASE = "/vol/rare_physionet"
    BRUG_DIR = os.path.join(BASE, "brugada_huca")
    CSN_DIR = os.path.join(BASE, "ecg_arrhythmia")
    os.makedirs(BRUG_DIR, exist_ok=True)
    os.makedirs(CSN_DIR, exist_ok=True)

    # 1. Download Brugada-HUCA v2.0.0 (7.3 MB)
    if not os.path.exists(os.path.join(BRUG_DIR, ".ready")):
        print("[CPU] Downloading Brugada-HUCA v2.0.0 from PhysioNet (7.3 MB)...")
        z_b = os.path.join(BASE, "brugada_tmp.zip")
        r_b = subprocess.run(["curl", "-L", "-f", "--retry", "4", "-sS", "-o", z_b, "https://physionet.org/static/published-projects/brugada-huca/brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-2.0.0.zip"])
        if r_b.returncode == 0 and os.path.exists(z_b) and os.path.getsize(z_b) > 1_000_000:
            subprocess.run(["unzip", "-q", "-o", z_b, "-d", BRUG_DIR], check=True)
            os.remove(z_b)
        else:
            subprocess.run(["physionet", "download", "brugada-huca", "--version", "2.0.0"], cwd=BRUG_DIR, check=True)
        open(os.path.join(BRUG_DIR, ".ready"), "w").write("ok\n")
        vol.commit()
    else:
        print("[CPU] Brugada-HUCA v2.0.0 already cached.")

    # Index Brugada-HUCA strictly using basal_pattern in {1, 2} (visible basal ECG pattern!)
    meta_path = None
    for root, _, files in os.walk(BRUG_DIR):
        if "metadata.csv" in files:
            meta_path = os.path.join(root, "metadata.csv")
            break
    assert meta_path, f"metadata.csv not found inside {BRUG_DIR}"
    df_b = pd.read_csv(meta_path)
    print(f"[CPU] Brugada-HUCA metadata columns: {list(df_b.columns)} | rows={len(df_b)}")

    # Map patient_id -> .hea path
    hea_by_pid = {}
    for root, _, files in os.walk(BRUG_DIR):
        for fn in files:
            if fn.endswith(".hea"):
                stem = os.path.splitext(fn)[0]
                hea_by_pid[stem] = os.path.join(root, stem)

    brugada_pos, brugada_ctrl = [], []
    for _, r in df_b.iterrows():
        pid = str(r["patient_id"]).strip()
        rec_path = hea_by_pid.get(pid)
        if not rec_path:
            continue
        bp = int(r.get("basal_pattern", 0))
        br = int(r.get("brugada", 0))
        if bp in (1, 2):
            brugada_pos.append({"wfdb_path": rec_path, "patient_id": pid, "basal_pattern": bp, "classes": ["BRUGADA_TYPE_1"]})
        elif br == 0 and bp == 0:
            brugada_ctrl.append({"wfdb_path": rec_path, "patient_id": pid, "basal_pattern": 0, "classes": ["NORMAL_SINUS_RHYTHM"]})

    print(f"[CPU] Brugada-HUCA indexed: visible basal_pattern(1|2)={len(brugada_pos)} | clean controls={len(brugada_ctrl)}")

    # 2. Download Chapman-Shaoxing + Ningbo ecg-arrhythmia v1.0.0 (45,152 12-lead ECGs)
    if not os.path.exists(os.path.join(CSN_DIR, ".ready")):
        print("[CPU] Downloading PhysioNet ecg-arrhythmia v1.0.0 (45,152 12-lead ECGs)...")
        zip_url = "https://physionet.org/static/published-projects/ecg-arrhythmia/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip"
        zpath = os.path.join(BASE, "csn_tmp.zip")
        r = subprocess.run(["curl", "-L", "-f", "--retry", "4", "-sS", "-o", zpath, zip_url])
        if r.returncode == 0 and os.path.exists(zpath) and os.path.getsize(zpath) > 10_000_000:
            print("[CPU] Unzipping ecg-arrhythmia into Volume...")
            subprocess.run(["unzip", "-q", "-o", zpath, "-d", CSN_DIR], check=True)
            os.remove(zpath)
        else:
            print("[CPU] Direct zip fallback -> using physionet CLI...")
            subprocess.run(
                ["physionet", "download", "ecg-arrhythmia", "--version", "1.0.0"],
                cwd=CSN_DIR,
                check=True,
            )
        open(os.path.join(CSN_DIR, ".ready"), "w").write("ok\n")
        vol.commit()
    else:
        print("[CPU] PhysioNet ecg-arrhythmia already cached.")

    # Scan all .hea headers for our 4 rare SNOMED-CT classes
    counts = {c: 0 for c in set(SNOMED_TO_RARE.values())}
    rare_csn = []
    total_hea = 0
    for root, _, files in os.walk(CSN_DIR):
        for fn in files:
            if not fn.endswith(".hea"):
                continue
            total_hea += 1
            fp = os.path.join(root, fn)
            matched = set()
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.startswith("#Dx:"):
                            codes = [c.strip() for c in line.split(":", 1)[1].split(",")]
                            for code in codes:
                                if code in SNOMED_TO_RARE:
                                    matched.add(SNOMED_TO_RARE[code])
                            break
            except Exception:
                continue
            if matched:
                for c in matched:
                    counts[c] += 1
                rare_csn.append({
                    "wfdb_path": os.path.splitext(fp)[0],
                    "patient_id": os.path.splitext(fn)[0],
                    "classes": sorted(matched),
                })

    manifest = {
        "brugada_visible_pos": brugada_pos,
        "brugada_clean_ctrl": brugada_ctrl,
        "csn_rare_records": rare_csn,
        "summary_counts": {
            "BRUGADA_TYPE_1_visible_basal": len(brugada_pos),
            "BRUGADA_clean_controls": len(brugada_ctrl),
            "csn_total_scanned": total_hea,
            **counts,
        },
    }
    out_path = os.path.join(BASE, "rare_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    vol.commit()

    print(f"\n================ RARE PHYSIONET CACHE COMPLETE ({(time.time()-t0)/60:.1f} min) ================")
    for k, v in manifest["summary_counts"].items():
        print(f"  {k:32s}: {v}")
    print(f"Saved manifest to {out_path}")


@app.local_entrypoint()
def main():
    cache_rare_physionet.remote()
