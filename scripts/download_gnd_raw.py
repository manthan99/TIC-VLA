#!/usr/bin/env python3
"""Download the raw GND dataset from GMU Dataverse (resumable, parallel).

The official downloader (jingGM/GND) fetches serially with no resume, which is
painful for ~0.9 TB. This does the same API calls but with HTTP range resume,
retries, and a few workers.

Usage:
    python scripts/download_gnd_raw.py --dest /data/patelm/ticvla/GND_raw
    python scripts/download_gnd_raw.py --dest ... --only AU,NOVA   # subset first
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

SERVER = "https://dataverse.orc.gmu.edu"
PERSISTENT_ID = "doi:10.13021/ORC2020/JUIW5F"


def list_files() -> list[dict]:
    r = requests.get(
        f"{SERVER}/api/datasets/:persistentId/versions/:latest/files",
        params={"persistentId": PERSISTENT_ID}, timeout=120,
    )
    r.raise_for_status()
    out = []
    for f in r.json().get("data", []):
        data = f.get("dataFile", {})
        if data.get("id") and (f.get("label") or data.get("filename")):
            out.append({
                "id": data["id"],
                "name": f.get("label") or data["filename"],
                "subdir": f.get("directoryLabel") or "",
                "size": int(data.get("filesize", 0) or 0),
            })
    return out


def fetch(info: dict, dest_root: Path, attempts: int = 6) -> str:
    target = dest_root / info["subdir"] / info["name"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and info["size"] and target.stat().st_size == info["size"]:
        return f"skip  {target.name}"

    url = f"{SERVER}/api/access/datafile/{info['id']}"
    for attempt in range(attempts):
        have = target.stat().st_size if target.exists() else 0
        if info["size"] and have == info["size"]:
            return f"done  {target.name}"
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                # Server ignored the range request; restart the file.
                mode = "ab" if (have and r.status_code == 206) else "wb"
                r.raise_for_status()
                with open(target, mode) as fh:
                    for chunk in r.iter_content(chunk_size=8 << 20):
                        fh.write(chunk)
            size = target.stat().st_size
            if not info["size"] or size == info["size"]:
                return f"ok    {target.name} ({size/1e9:.2f} GB)"
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return f"FAIL  {target.name}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=str, default="/data/patelm/ticvla/GND_raw")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated site prefixes to fetch (e.g. AU,NOVA)")
    args = parser.parse_args()

    dest = Path(args.dest)
    files = list_files()
    if args.only:
        wanted = tuple(s.strip() for s in args.only.split(",") if s.strip())
        files = [f for f in files if (f["subdir"] or f["name"]).startswith(wanted)]
    total = sum(f["size"] for f in files)
    print(f"{len(files)} files, {total/1e12:.2f} TB -> {dest}", flush=True)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, f, dest): f for f in files}
        for fut in as_completed(futures):
            done += 1
            print(f"[{done}/{len(files)}] {fut.result()}", flush=True)

    got = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(f"\ncomplete: {got/1e12:.2f} TB in {dest}")


if __name__ == "__main__":
    main()
