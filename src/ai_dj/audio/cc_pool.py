"""Ingest an open Creative Commons audio pool (MTG-Jamendo via JamendoMaxCaps) for DSP & RL."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import duckdb

JAMENDO_PARQUET_SHARD_0 = (
    "https://huggingface.co/api/datasets/amaai-lab/JamendoMaxCaps/parquet/default/train/0.parquet"
)
JAMENDO_PARQUET_SHARD_TEMPLATE = (
    "https://huggingface.co/api/datasets/amaai-lab/JamendoMaxCaps/parquet/default/train/{shard}.parquet"
)
MTG_JAMENDO_GENRE_TSV_URL = (
    "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/autotagging_genre.tsv"
)

CLUB_GENRES = frozenset(
    {
        "genre---house",
        "genre---techno",
        "genre---trance",
        "genre---club",
        "genre---dance",
        "genre---deephouse",
        "genre---techhouse",
        "genre---minimal",
        "genre---drumnbass",
        "genre---eurodance",
    }
)
EXCLUDED_NON_CLUB_GENRES = frozenset(
    {
        "genre---metal",
        "genre---rock",
        "genre---punkrock",
        "genre---hardrock",
        "genre---classical",
        "genre---jazz",
        "genre---folk",
        "genre---country",
        "genre---acoustic",
        "genre---orchestral",
        "genre---ambient",
        "genre---soundtrack",
        "genre---chillout",
    }
)


def load_mtg_jamendo_club_genres(cache_path: Path | None = None) -> dict[str, list[str]]:
    """Load official MTG-Jamendo genre tags filtered strictly to club/electronic genres."""
    if cache_path is not None and cache_path.exists():
        raw_text = cache_path.read_text(encoding="utf-8")
    else:
        with urllib.request.urlopen(MTG_JAMENDO_GENRE_TSV_URL, timeout=30) as resp:
            raw_text = resp.read().decode("utf-8")
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(raw_text, encoding="utf-8")

    club_map: dict[str, list[str]] = {}
    for line in raw_text.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        tid = parts[0].replace("track_", "").lstrip("0")
        tags = set(parts[5:])
        if (tags & CLUB_GENRES) and not (tags & EXCLUDED_NON_CLUB_GENRES):
            club_map[tid] = sorted(t.replace("genre---", "") for t in tags)
    return club_map


def ingest_electronic_cc_pool(
    out_dir: Path,
    *,
    max_tracks: int = 24,
    max_shards: int = 8,
    min_bytes: int = 1_800_000,
    max_bytes: int = 7_500_000,
    force: bool = False,
) -> Path:
    """Ingest verified club/house/techno/trance/dance CC tracks using MTG-Jamendo genre tags."""
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
            and len(cached) >= max_tracks
            and all(
                isinstance(item, dict)
                and item.get("genres")
                and Path(str(item.get("audio_path", ""))).exists()
                for item in cached[:max_tracks]
            )
        ):
            return manifest_path

    club_map = load_mtg_jamendo_club_genres(cache_path=out_dir / "mtg_jamendo_genre.tsv")

    manifest: list[dict] = []
    seen_ids: set[str] = set()
    # Reuse any previously downloaded verified club tracks in out_dir
    for existing in sorted(out_dir.glob("*.mp3")):
        tid = existing.stem.lstrip("0")
        if tid in club_map and tid not in seen_ids:
            seen_ids.add(tid)
            manifest.append(
                {
                    "track_id": tid,
                    "filename": existing.name,
                    "audio_path": str(existing),
                    "bytes": int(existing.stat().st_size),
                    "genres": club_map.get(tid, ["electronic"]),
                    "source": "amaai-lab/JamendoMaxCaps+MTG-Jamendo",
                    "source_url": f"https://www.jamendo.com/track/{tid}",
                    "license": None,
                    "license_status": "unverified_per_track_cc_terms",
                }
            )

    if len(manifest) < max_tracks:
        from datasets import Audio, load_dataset

        ds = load_dataset("amaai-lab/JamendoMaxCaps", split="train", streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))
        max_rows = max(2500, max_shards * 400)
        for idx, row in enumerate(ds):
            if idx >= max_rows or len(manifest) >= max_tracks:
                break
            audio_obj = row.get("audio") or {}
            filename = str(audio_obj.get("path") or "")
            raw_bytes = audio_obj.get("bytes")
            if not filename or not raw_bytes:
                continue
            nbytes = len(raw_bytes)
            if not (int(min_bytes) <= nbytes <= int(max_bytes)):
                continue
            safe_name = Path(filename).name
            tid = Path(safe_name).stem.lstrip("0")
            if tid not in club_map or tid in seen_ids:
                continue
            seen_ids.add(tid)
            dest = out_dir / safe_name
            dest.write_bytes(bytes(raw_bytes))
            manifest.append(
                {
                    "track_id": tid,
                    "filename": safe_name,
                    "audio_path": str(dest),
                    "bytes": int(nbytes),
                    "genres": club_map.get(tid, ["electronic"]),
                    "source": "amaai-lab/JamendoMaxCaps+MTG-Jamendo",
                    "source_url": f"https://www.jamendo.com/track/{tid}",
                    "license": None,
                    "license_status": "unverified_per_track_cc_terms",
                }
            )

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


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
