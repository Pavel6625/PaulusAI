from paulus import affect


def test_describe_reports_pad():
    assert "P=" in affect.describe()


def test_thanks_raises_pleasure():
    before = affect._load()["pleasure"]
    affect.feel("owner_thanks")
    assert affect._load()["pleasure"] >= before


def test_decay_moves_toward_baseline():
    affect.feel("owner_thanks")
    base = affect._baseline()["pleasure"]
    p1 = affect._load()["pleasure"]
    affect.decay()
    p2 = affect._load()["pleasure"]
    assert abs(p2 - base) <= abs(p1 - base)


def test_unknown_event_is_noop():
    assert affect.feel("nonexistent_event") == {}
