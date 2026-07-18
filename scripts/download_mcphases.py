#!/usr/bin/env python3
"""Download mcPHASES from PhysioNet using credentials in .env."""

from __future__ import annotations

import os
import re
import sys
import zipfile
from pathlib import Path

import urllib.request
import http.cookiejar

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
OUT_DIR = ROOT / "data" / "mcphases"
ZIP_PATH = OUT_DIR / "mcphases-1.0.0.zip"
LOGIN_URL = "https://physionet.org/login/"
ZIP_URL = "https://physionet.org/content/mcphases/get-zip/1.0.0/"


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def main() -> int:
    if not ENV_PATH.exists():
        print(f"Missing {ENV_PATH}", file=sys.stderr)
        return 1

    env = load_env(ENV_PATH)
    user = env.get("PHYSIONET_USER")
    password = env.get("PHYSIONET_PASSWORD")
    if not user or not password:
        print("PHYSIONET_USER / PHYSIONET_PASSWORD required in .env", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

    print("Logging in to PhysioNet...")
    login_html = opener.open(LOGIN_URL).read().decode("utf-8", errors="replace")
    match = re.search(
        r'name=["\']csrfmiddlewaretoken["\'] value=["\']([^"\']+)',
        login_html,
    )
    if not match:
        print("Could not find CSRF token on login page", file=sys.stderr)
        return 1

    csrf = match.group(1)
    form = urllib.parse.urlencode(
        {
            "csrfmiddlewaretoken": csrf,
            "username": user,
            "password": password,
            "next": "/",
        }
    ).encode()
    req = urllib.request.Request(
        LOGIN_URL,
        data=form,
        headers={
            "Referer": LOGIN_URL,
            "Origin": "https://physionet.org",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    opener.open(req).read()

    session_ok = any(c.name == "sessionid" for c in jar)
    if not session_ok:
        print("Login failed: no session cookie", file=sys.stderr)
        return 1
    print("Login OK")

    print(f"Downloading zip -> {ZIP_PATH}")
    with opener.open(ZIP_URL) as resp, open(ZIP_PATH, "wb") as out:
        content_type = resp.headers.get("Content-Type", "")
        if "zip" not in content_type and "octet-stream" not in content_type:
            preview = resp.read(500).decode("utf-8", errors="replace")
            print(f"Unexpected response ({content_type}):\n{preview}", file=sys.stderr)
            return 1
        total = resp.headers.get("Content-Length")
        total_i = int(total) if total else None
        downloaded = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total_i:
                pct = 100.0 * downloaded / total_i
                print(f"\r  {downloaded / 1e6:.1f} / {total_i / 1e6:.1f} MB ({pct:.1f}%)", end="", flush=True)
            else:
                print(f"\r  {downloaded / 1e6:.1f} MB", end="", flush=True)
    print()

    size = ZIP_PATH.stat().st_size
    if size < 1_000_000:
        print(f"Download looks too small ({size} bytes)", file=sys.stderr)
        return 1
    print(f"Downloaded {size / 1e6:.1f} MB")

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(OUT_DIR)
    print(f"Extracted to {OUT_DIR}")

    # Summarize contents
    csvs = sorted(OUT_DIR.rglob("*.csv"))
    print(f"CSV files: {len(csvs)}")
    for p in csvs[:30]:
        print(f"  {p.relative_to(OUT_DIR)}")
    if len(csvs) > 30:
        print(f"  ... and {len(csvs) - 30} more")
    return 0


if __name__ == "__main__":
    import urllib.parse

    raise SystemExit(main())
