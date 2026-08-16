from harness.agents.base import AgentResult, extract_json, validate_output
from harness.artifacts.schemas import Review


def test_extract_json_from_fence():
    text = 'Here is my review.\n```json\n{"verdict": "PASS"}\n```\nDone.'
    assert extract_json(text) == {"verdict": "PASS"}


def test_extract_json_last_fence_wins():
    text = '```json\n{"verdict": "REPAIR"}\n```\ntext\n```json\n{"verdict": "PASS"}\n```'
    assert extract_json(text) == {"verdict": "PASS"}


def test_extract_json_bare_object():
    text = 'Result: {"verdict": "PASS", "blocking_issues": []} — all good'
    assert extract_json(text)["verdict"] == "PASS"


def test_extract_json_none():
    assert extract_json("no json here") is None


def test_validate_output_ok():
    result = AgentResult(status="COMPLETED", output_text='```json\n{"verdict": "PASS"}\n```')
    model, error = validate_output(result, Review)
    assert error is None
    assert model.verdict == "PASS"


def test_validate_output_schema_error():
    result = AgentResult(status="COMPLETED", output_text='```json\n{"verdict": "MAYBE"}\n```')
    model, error = validate_output(result, Review)
    assert model is None
    assert "schema" in error
