#!/bin/sh
set -eu
umask 077

# Use SQLite's online backup API so WAL-backed writers can remain active. The
# destination is deliberately fixed: retention must never escape this mount.
exec python - <<'PY'
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from news_bot.research.config import ResearchConfig
from news_bot.research.store import ResearchStore


backup_root = Path("/app/backups")
if backup_root.is_symlink():
    raise SystemExit("backup directory must not be a symlink")
backup_root.mkdir(mode=0o700, parents=True, exist_ok=True)
resolved_root = backup_root.resolve(strict=True)
if resolved_root != backup_root:
    raise SystemExit("backup directory resolved outside /app/backups")

config = ResearchConfig.from_env(include_flex=False, include_model_secret=False)
database_path = config.database_path
if database_path.is_symlink() or not database_path.is_file():
    raise SystemExit("research database is missing or unsafe")
if database_path.resolve(strict=True).parent != config.data_dir.resolve(strict=True):
    raise SystemExit("research database resolved outside RESEARCH_DATA_DIR")

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
target = backup_root / f"research-{timestamp}.db"
temporary = backup_root / f".research-{timestamp}-{os.getpid()}.tmp"
if target.exists() or target.is_symlink() or temporary.exists() or temporary.is_symlink():
    raise SystemExit("backup destination already exists")

store = ResearchStore(database_path)
try:
    source = store.connect()
    destination = sqlite3.connect(temporary)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()

    check = sqlite3.connect(f"file:{temporary}?mode=ro", uri=True)
    try:
        result = check.execute("PRAGMA integrity_check").fetchone()
    finally:
        check.close()
    if result != ("ok",):
        raise RuntimeError("backup integrity check failed")

    temporary.replace(target)
    target.chmod(0o600)
    directory_fd = os.open(backup_root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
except BaseException:
    if temporary.exists() and not temporary.is_symlink():
        temporary.unlink()
    raise

backups = sorted(backup_root.glob("research-*.db"), reverse=True)
for path in backups:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("unsafe entry in backup retention set")
    if path.resolve(strict=True).parent != resolved_root:
        raise RuntimeError("backup retention path escaped /app/backups")
for stale in backups[14:]:
    stale.unlink()

print(target)
PY
