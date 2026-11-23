from ai_dj.data.splits import assign_dj_split


def test_dj_split_is_deterministic_and_balanced():
    splits = [assign_dj_split(f"dj_{i}") for i in range(1000)]
    assert assign_dj_split("four tet") == assign_dj_split("four tet")
    train_frac = splits.count("train") / 1000
    val_frac = splits.count("val") / 1000
    test_frac = splits.count("test") / 1000
    assert 0.75 <= train_frac <= 0.85
    assert 0.07 <= val_frac <= 0.13
    assert 0.07 <= test_frac <= 0.13
