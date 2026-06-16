from paulus import appraisal
from paulus.appraisal import Appraisal


def test_joy_from_positive_actual_event():
    emo = appraisal.appraise(Appraisal(desirability=0.8))
    assert "joy" in emo and emo["joy"] > 0


def test_distress_amplified_by_neuroticism():
    low = appraisal.appraise(Appraisal(desirability=-0.8), neuroticism=0.0)
    high = appraisal.appraise(Appraisal(desirability=-0.8), neuroticism=1.0)
    assert high["distress"] > low["distress"]


def test_prospective_event_yields_hope():
    emo = appraisal.appraise(Appraisal(desirability=0.6, prospect="prospective", likelihood=0.5))
    assert "hope" in emo


def test_compound_gratitude_from_other_agency():
    emo = appraisal.appraise(Appraisal(desirability=0.6, praiseworthiness=0.6, agency="other"))
    assert "gratitude" in emo and "admiration" in emo


def test_event_template_lookup():
    assert appraisal.appraise_event("owner_thanks")
    assert appraisal.appraise_event("unknown_event") == {}


def test_dominant_picks_strongest():
    assert appraisal.dominant({"joy": 0.2, "pride": 0.9}) == "pride"
    assert appraisal.dominant({}) is None
