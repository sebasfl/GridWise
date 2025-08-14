# src/ingest/fetch_bdg2.py
import argparse, hashlib, pathlib, sys, requests
import pandas as pd

BASE = "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2"

# key: (remote_path_in_repo, local_rel_path_under_out_root, optional_sha256)
FILES = {
    "metadata": ("data/metadata/metadata.csv", "data/metadata/metadata.csv", None),
    "raw_electricity": ("data/meters/raw/electricity.csv", "data/meters/raw/electricity.csv", None),
    "cleaned_electricity": ("data/meters/cleaned/electricity_cleaned.csv", "data/meters/cleaned/electricity_cleaned.csv", None),
}

def sha256sum(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def fetch_one(ref: str, key: str, out_root: pathlib.Path):
    remote, local_rel, expect_sha = FILES[key]
    url = f"{BASE}/{ref}/{remote}"
    out_path = out_root / local_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    with out_path.open("wb") as f:
        for chunk in r.iter_content(1 << 20):
            if chunk:
                f.write(chunk)

    got = sha256sum(out_path)
    if expect_sha and got != expect_sha:
        raise SystemExit(f"[{key}] SHA mismatch! expected {expect_sha}, got {got}")
    print(f"OK: {key} → {out_path} (sha256={got})")

def build_parquet_from_cleaned(cleaned_csv: pathlib.Path, out_parquet: pathlib.Path):
    """
    The BDG2 cleaned file is WIDE:
        columns = ['timestamp', '<building_1>', '<building_2>', ...]
    We melt to LONG and emit the schema that the trainer expects:
        ['timestamp_local', 'building_id', 'meter', 'value']
    """
    print(f"Reading cleaned CSV: {cleaned_csv}")
    # parse dates efficiently
    df = pd.read_csv(cleaned_csv)
    if "timestamp" not in df.columns:
        raise SystemExit("CSV is missing 'timestamp' column")

    # Melt wide -> long
    value_vars = [c for c in df.columns if c != "timestamp"]
    out = df.melt(
        id_vars=["timestamp"],
        value_vars=value_vars,
        var_name="building_id",
        value_name="value",
    )

    # Rename and add columns expected by the trainer
    out = out.rename(columns={"timestamp": "timestamp_local"})
    out["meter"] = "electricity"

    # Clean types
    out["timestamp_local"] = pd.to_datetime(out["timestamp_local"], errors="coerce")
    out = out.dropna(subset=["timestamp_local"])
    out["building_id"] = out["building_id"].astype(str)

    out = out.sort_values(["building_id", "timestamp_local"]).reset_index(drop=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)
    print(f"Wrote Parquet: {out_parquet} (rows={len(out)})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="master", help="Commit SHA or branch of BDG2 (e.g. master)")
    ap.add_argument("--out_root", default="/app/data/external/bdg2", help="Where to save downloads")
    ap.add_argument("--what", default="metadata,raw_electricity,cleaned_electricity",
                    help="Comma-separated keys to fetch")
    ap.add_argument("--emit_parquet", default=None,
                    help="If set, write long-format parquet here (e.g. /app/data/processed/bdg2_electricity_long.parquet)")
    args = ap.parse_args()

    out_root = pathlib.Path(args.out_root)
    keys = [w.strip() for w in args.what.split(",") if w.strip()]

    for key in keys:
        if key not in FILES:
            sys.exit(f"Unknown key: {key}. Options: {', '.join(FILES)}")
        fetch_one(args.ref, key, out_root)

    if args.emit_parquet:
        cleaned = out_root / "data/meters/cleaned/electricity_cleaned.csv"
        if not cleaned.exists():
            sys.exit(f"Cleaned CSV not found at {cleaned}")
        build_parquet_from_cleaned(cleaned, pathlib.Path(args.emit_parquet))

if __name__ == "__main__":
    main()
