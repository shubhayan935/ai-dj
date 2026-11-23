"""Parsers for MixesDB mix titles, tracklist rows, and artist/DJ normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Curated club, festival, and broadcast series recognized as a curation-quality signal.
CURATED_SERIES_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Boiler Room", re.compile(r"\bboiler\s+room\b", re.IGNORECASE)),
    ("Essential Mix", re.compile(r"\bessential\s+mix\b", re.IGNORECASE)),
    ("Resident Advisor", re.compile(r"\b(ra\.\d+|resident\s+advisor)\b", re.IGNORECASE)),
    ("Dekmantel", re.compile(r"\bdekmantel\b", re.IGNORECASE)),
    ("Fabric", re.compile(r"\bfabric\b", re.IGNORECASE)),
    ("Berghain / Panorama Bar", re.compile(r"\b(berghain|panorama\s+bar)\b", re.IGNORECASE)),
    ("Rinse FM", re.compile(r"\brinse\s*fm\b", re.IGNORECASE)),
    ("NTS", re.compile(r"\bnts\s+(radio|live)?\b", re.IGNORECASE)),
    ("FACT Mix", re.compile(r"\bfact\s+mix\b", re.IGNORECASE)),
    ("XLR8R", re.compile(r"\bxlr8r\b", re.IGNORECASE)),
    ("Mixmag", re.compile(r"\bmixmag\b", re.IGNORECASE)),
    ("Cercle", re.compile(r"\bcercle\b", re.IGNORECASE)),
    ("BBC Radio 1", re.compile(r"\bbbc\s+radio\s+1\b", re.IGNORECASE)),
    ("HÖR", re.compile(r"\b(hör|hoer)\b", re.IGNORECASE)),
    ("Awakenings", re.compile(r"\bawakenings\b", re.IGNORECASE)),
    ("Time Warp", re.compile(r"\btime\s+warp\b", re.IGNORECASE)),
]

DATE_PREFIX_RE = re.compile(r"^(\d{4}(?:-\d{2}(?:-\d{2})?)?)\s*-\s*(.+)$")
TIMESTAMP_PREFIX_RE = re.compile(r"^\[\s*([0-9:?]+)\s*\]\s*(.+)$")
TRAILING_LABEL_RE = re.compile(r"\s*\[([^\]]+)\]\s*$")
VERSION_PAREN_RE = re.compile(
    r"\s*\(([^)]*(?:mix|remix|edit|dub|version|vip|bootleg|rework|remaster|live|instrumental|acapella)[^)]*)\)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedMixTitle:
    mix_date: str | None
    year: int | None
    dj_name: str
    dj_name_norm: str
    venue_or_show: str | None
    curated_series: str | None


@dataclass(frozen=True)
class ParsedTrackTitle:
    timestamp_raw: str | None
    artist: str
    title: str
    version: str | None
    record_label: str | None


def normalize_name(name: str) -> str:
    """Normalize a DJ or artist name for grouping, deduplication, and disjoint splits."""
    text = name.lower().strip()
    text = re.sub(r"\b(feat\.?|ft\.?|featuring|presents|pres\.?|b2b|vs\.?)\b.*$", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text or "unknown"


def detect_curated_series(title: str, tags: list[str] | None = None) -> str | None:
    """Return canonical series name if the mix belongs to a known curated series."""
    haystack = title + " " + " ".join(tags or [])
    for label, pattern in CURATED_SERIES_PATTERNS:
        if pattern.search(haystack):
            return label
    return None


def parse_mix_title(title: str, tags: list[str] | None = None) -> ParsedMixTitle:
    """Parse MixesDB mix title like '2016-04-12 - Four Tet @ Boiler Room London'."""
    raw = title.strip()
    mix_date: str | None = None
    year: int | None = None
    rest = raw

    m_date = DATE_PREFIX_RE.match(raw)
    if m_date:
        mix_date = m_date.group(1)
        year = int(mix_date[:4])
        rest = m_date.group(2).strip()
    else:
        m_year = re.search(r"\b(19[7-9]\d|20[0-2]\d)\b", raw)
        if m_year:
            year = int(m_year.group(1))

    # Split DJ name vs. venue/show on '@' or ' - '
    venue_or_show: str | None = None
    if " @ " in rest:
        dj_part, venue_part = rest.split(" @ ", 1)
        dj_name = dj_part.strip()
        venue_or_show = venue_part.strip()
    elif " - " in rest:
        dj_part, venue_part = rest.split(" - ", 1)
        dj_name = dj_part.strip()
        venue_or_show = venue_part.strip()
    else:
        dj_name = rest.strip()

    # Strip trailing parentheticals from DJ name if any
    dj_name = re.sub(r"\s*\([^)]*\)\s*$", "", dj_name).strip() or "Unknown"
    curated = detect_curated_series(raw, tags)

    return ParsedMixTitle(
        mix_date=mix_date,
        year=year,
        dj_name=dj_name,
        dj_name_norm=normalize_name(dj_name),
        venue_or_show=venue_or_show,
        curated_series=curated,
    )


def parse_track_title(raw_title: str) -> ParsedTrackTitle:
    """Parse MixesDB tracklist entry like '[05] Code 61 - Drop The Deal [LCE 003]'."""
    text = raw_title.strip()
    timestamp_raw: str | None = None

    m_ts = TIMESTAMP_PREFIX_RE.match(text)
    if m_ts:
        timestamp_raw = m_ts.group(1).strip()
        text = m_ts.group(2).strip()

    record_label: str | None = None
    m_lbl = TRAILING_LABEL_RE.search(text)
    if m_lbl:
        record_label = m_lbl.group(1).strip()
        text = text[: m_lbl.start()].strip()

    if " - " in text:
        artist, title = text.split(" - ", 1)
        artist = artist.strip() or "Unknown"
        title = title.strip() or "Unknown"
    else:
        artist = "Unknown"
        title = text or "Unknown"

    version: str | None = None
    m_ver = VERSION_PAREN_RE.search(title)
    if m_ver:
        version = m_ver.group(1).strip()
        title = title[: m_ver.start()].strip() or title

    return ParsedTrackTitle(
        timestamp_raw=timestamp_raw,
        artist=artist,
        title=title,
        version=version,
        record_label=record_label,
    )
