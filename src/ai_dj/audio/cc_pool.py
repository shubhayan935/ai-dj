"""Ingest an open Creative Commons audio pool (MTG-Jamendo via JamendoMaxCaps) for DSP & RL."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

JAMENDO_PARQUET_SHARD_0 = (
    "https://huggingface.co/api/datasets/amaai-lab/JamendoMaxCaps/parquet/default/train/0.parquet"
)


def ingest_cc_audio_pool(
    out_dir: Path,
    *,
    max_tracks: int = 16,
    min_bytes: int = 1_500_000,
    max_bytes: int = 6_000_000,
    force: bool = False,
) -> Path:
    """Download a small pool of full-length CC-licensed MP3 tracks with provenance metadata."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not force:
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if (
            isinstance(cached, list)
            and len(cached) == max_tracks
            and all(
                isinstance(item, dict)
                and int(min_bytes) <= int(item.get("bytes", -1)) <= int(max_bytes)
                and Path(str(item.get("audio_path", ""))).exists()
                for item in cached
            )
        ):
            return manifest_path

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    query = f"""
        SELECT audio.path AS filename, audio.bytes AS raw_bytes, octet_length(audio.bytes) AS nbytes
        FROM read_parquet('{JAMENDO_PARQUET_SHARD_0}')
        WHERE octet_length(audio.bytes) BETWEEN {int(min_bytes)} AND {int(max_bytes)}
        LIMIT {int(max_tracks)}
    """
    rows = con.execute(query).fetchall()
    con.close()

    manifest: list[dict] = []
    for filename, raw_bytes, nbytes in rows:
        safe_name = Path(filename).name
        track_id = Path(safe_name).stem
        dest = out_dir / safe_name
        dest.write_bytes(bytes(raw_bytes))
        manifest.append(
            {
                "track_id": track_id,
                "filename": safe_name,
                "audio_path": str(dest),
                "bytes": int(nbytes),
                "source": "amaai-lab/JamendoMaxCaps",
                "source_url": f"https://www.jamendo.com/track/{track_id}",
                "license": None,
                "license_status": "unverified_per_track_cc_terms",
            }
        )

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
