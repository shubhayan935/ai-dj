from ai_dj.data.parsing import (
    classify_set_phase,
    extract_genres,
    normalize_name,
    parse_mix_title,
    parse_track_title,
)


def test_parse_mix_title_with_date_venue_and_series():
    parsed = parse_mix_title(
        "2016-04-12 - Four Tet @ Boiler Room London",
        tags=["House", "UK Garage"],
    )
    assert parsed.mix_date == "2016-04-12"
    assert parsed.year == 2016
    assert parsed.dj_name == "Four Tet"
    assert parsed.dj_name_norm == "four tet"
    assert parsed.venue_or_show == "Boiler Room London"
    assert parsed.curated_series == "Boiler Room"


def test_parse_track_title_with_timestamp_version_and_label():
    parsed = parse_track_title(
        '[24] Off - Electrica Salsa (12" Version) [ZYX - 5548]'
    )
    assert parsed.timestamp_raw == "24"
    assert parsed.artist == "Off"
    assert parsed.title == "Electrica Salsa"
    assert parsed.version == '12" Version'
    assert parsed.record_label == "ZYX - 5548"


def test_normalize_name_strips_collaborations():
    assert normalize_name("Bicep feat. Clara La San") == "bicep"
    assert normalize_name("Four Tet b2b Floating Points") == "four tet"


def test_extract_genres_strips_dj_names_and_shows():
    tags = [
        "Photek",
        "FACT Mix",
        "House (Old School)",
        "Drum & Bass",
        "Resident Advisor Podcast",
        "Boiler Room",
        "Tech House",
        "Promo Mix",
        "A State Of Trance",
        "Defected In The House",
        "DNB Dojo",
        "Sun & Bass",
    ]
    assert extract_genres(tags, dj_name_norm="photek") == [
        "House (Old School)",
        "Drum & Bass",
        "Tech House",
    ]


def test_classify_set_phase():
    assert classify_set_phase(0.05) == "opener"
    assert classify_set_phase(0.25) == "build"
    assert classify_set_phase(0.60) == "peak"
    assert classify_set_phase(0.82) == "cooldown"
    assert classify_set_phase(0.95) == "closer"
