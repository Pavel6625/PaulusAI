from paulus import skills


def test_add_and_find_skill():
    skills.add_skill("deploy site", "when asked to deploy the website", "run make deploy")
    hits = skills.find_skills("how do I deploy the website")
    assert hits and hits[0]["name"] == "deploy site"


def test_unverified_becomes_verified_on_use():
    skills.add_skill("backup", "when backing up", "tar czf ...", status="unverified")
    skills.mark_used("backup", success=True)
    s = next(x for x in skills._load() if x["name"] == "backup")
    assert s["status"] == "verified" and s["uses"] == 1


def test_adding_existing_updates_in_place():
    skills.add_skill("note", "old", "steps1")
    msg = skills.add_skill("note", "new when", "steps2")
    assert "updated" in msg
    s = next(x for x in skills._load() if x["name"] == "note")
    assert s["when_to_use"] == "new when"
