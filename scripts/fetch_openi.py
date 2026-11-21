#!/usr/bin/env python3
"""Fetch open-access AVF figures from Open-i for *qualitative* inspection only.

This script deliberately does **not** create labels. Open-i figures are not a
paired, patient-level AVF cohort, and they must never be used to claim clinical
performance. They are useful for smoke-testing preprocessing, checking skin-tone
coverage, and assembling a slide deck.

Usage:
    python scripts/fetch_openi.py --query "arteriovenous fistula dialysis" --n 40
    python scripts/fetch_openi.py --query "av fistula stenosis" --dry-run
    python scripts/fetch_openi.py --from-file openi_response.json --out data/openi

The Open-i API is occasionally down or returns an HTML maintenance page. The
script detects that and exits with a clear message rather than writing HTML as
if it were an image.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://openi.nlm.nih.gov/api/search"
BASE = "https://openi.nlm.nih.gov"
UA = "AVF-Sentinel-research/0.1 (hackathon; contact: local)"


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        body = resp.read()
        content_type = resp.headers.get("Content-Type", "")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        snippet = body[:200].decode("utf-8", "replace").replace("\n", " ")
        raise RuntimeError(
            "Open-i did not return JSON "
            f"(HTTP content-type={content_type!r}). It may be under maintenance. "
            f"Response starts: {snippet!r}"
        ) from None


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    if not data:
        raise RuntimeError("empty response")
    # Reject HTML error pages masquerading as images.
    head = data[:200].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        raise RuntimeError("received HTML instead of an image")
    dest.write_bytes(data)


def _absolute(url: str) -> str:
    if not url:
        return ""
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/"):
        return BASE + url
    return url


def fetch(query: str, n: int, out: Path, dry_run: bool) -> int:
    params = {"query": query, "m": 1, "n": n}
    url = API + "?" + urllib.parse.urlencode(params)
    print(f"Open-i query: {query!r} (n={n})")
    print(f"GET {url}")
    payload = _get_json(url)
    items = payload.get("list") or payload.get("results") or []
    if not items:
        print("No results. Check the query or the Open-i status page.")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, item in enumerate(items, 1):
        img_url = _absolute(item.get("imgLarge") or item.get("imgThumb") or item.get("image") or "")
        if not img_url:
            continue
        ext = Path(urllib.parse.urlparse(img_url).path).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff"}:
            ext = ".jpg"
        img_id = str(item.get("imgId") or item.get("id") or f"openi_{i:04d}").replace("/", "_")
        dest = out / f"{img_id}{ext}"
        label = "DRY-RUN" if dry_run else "downloaded"
        if not dry_run:
            try:
                time.sleep(0.4)
                _download(img_url, dest)
            except (urllib.error.URLError, RuntimeError, TimeoutError) as exc:
                print(f"  [{i:03d}] skip {img_id}: {exc}")
                label = f"failed: {exc}"
                dest = Path("")
            else:
                print(f"  [{i:03d}] {dest.name}")
        else:
            print(f"  [{i:03d}] {img_url}")
        rows.append(
            {
                "img_id": img_id,
                "file": str(dest) if dest else "",
                "status": label,
                "title": item.get("title", ""),
                "journal": item.get("journal", ""),
                "pmid": item.get("pmid", ""),
                "pmcid": item.get("pmcid", ""),
                "source_url": img_url,
                "query": query,
            }
        )

    with (out / "index.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["img_id"])
        writer.writeheader()
        writer.writerows(rows)
    (out / "README.txt").write_text(
        "Open-access figures fetched for QUALITATIVE INSPECTION ONLY.\n"
        "No labels are created. This is not a patient-level, paired AVF dataset.\n"
        "Do not use these images to train or claim clinical performance.\n"
        "Check each figure's license and cite the original article.\n",
        encoding="utf-8",
    )
    print(f"Wrote {out/'index.csv'} ({len(rows)} items)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch open-access AVF figures for inspection only.")
    parser.add_argument("--query", default="arteriovenous fistula dialysis")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--out", default="data/openi")
    parser.add_argument("--from-file", default="", help="read a saved JSON response instead of calling the API")
    parser.add_argument("--dry-run", action="store_true", help="list URLs without downloading")
    args = parser.parse_args()

    if args.from_file:
        payload = json.loads(Path(args.from_file).read_text(encoding="utf-8"))
        items = payload.get("list") or payload.get("results") or []
        if not items:
            print("No items in the saved response.")
            return 1
        # Reuse fetch() by monkeypatching would be silly; write a small inline path.
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, item in enumerate(items, 1):
            img_url = _absolute(item.get("imgLarge") or item.get("imgThumb") or "")
            if not img_url:
                continue
            dest = out / f"openi_{i:04d}.jpg"
            if not args.dry_run:
                try:
                    _download(img_url, dest)
                except Exception as exc:  # noqa: BLE001 - report and continue
                    print(f"skip {img_url}: {exc}")
                    continue
            rows.append({"img_id": f"openi_{i:04d}", "file": str(dest), "source_url": img_url})
        print(f"Wrote {len(rows)} items to {out}")
        return 0

    try:
        return fetch(args.query, args.n, Path(args.out), args.dry_run)
    except (urllib.error.URLError, RuntimeError, TimeoutError) as exc:
        print(f"Open-i fetch failed: {exc}", file=sys.stderr)
        print(
            "Open-i is sometimes unavailable. Try again later, or save a JSON response "
            "manually and rerun with --from-file. Do not substitute unlabelled web images "
            "for a real cohort.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
