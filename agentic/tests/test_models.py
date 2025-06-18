from agentic.core.models import Hypothesis, CandidateFeature, VettedFeature, BOResult, RealizedFeature, ParameterSpec
from pydantic import ValidationError
import pytest
import datetime

def test_parameter_spec_schema():
    p = ParameterSpec(name="alpha", lower=0, upper=1, type="float", init=0.5, description="A param")
    assert p.name == "alpha"
    assert p.lower == 0
    assert p.upper == 1
    assert p.type == "float"
    assert p.init == 0.5
    assert p.description == "A param"
    # Invalid: upper <= lower
    with pytest.raises(ValidationError):
        ParameterSpec(name="bad", lower=1, upper=0, type="float")
    # Invalid: empty name
    with pytest.raises(ValidationError):
        ParameterSpec(name="", lower=0, upper=1, type="float")
    # Invalid: wrong type
    with pytest.raises(ValidationError):
        ParameterSpec(name="a", lower=0, upper=1, type="badtype")

def test_hypothesis_schema():
    h = Hypothesis(id="h1", summary="Test", rationale="Because", depends_on=["col1"])
    assert h.id == "h1"
    assert h.summary == "Test"
    assert h.rationale == "Because"
    assert h.depends_on == ["col1"]

def test_candidate_feature_schema():
    params = {"alpha": ParameterSpec(name="alpha", lower=0, upper=1, type="float", init=0.5)}
    f = CandidateFeature(
        name="f1", type="code", spec="x + y", feature_scope="item", depends_on=["col1"], parameters=params, rationale="Test feature"
    )
    assert f.name == "f1"
    assert f.type == "code"
    assert f.spec == "x + y"
    assert f.feature_scope == "item"
    assert f.depends_on == ["col1"]
    assert isinstance(f.parameters["alpha"], ParameterSpec)
    assert f.parameters["alpha"].lower == 0
    assert f.parameters["alpha"].upper == 1
    assert f.parameters["alpha"].type == "float"
    assert f.parameters["alpha"].init == 0.5
    assert f.rationale == "Test feature"

def test_vetted_feature_inherits_candidate():
    params = {"alpha": ParameterSpec(name="alpha", lower=0, upper=1, type="float", init=0.5)}
    f = VettedFeature(
        name="vf", type="code", spec="x*y", feature_scope="user", depends_on=[], parameters=params, rationale="Vetted"
    )
    assert isinstance(f, CandidateFeature)
    assert f.feature_scope == "user"
    assert f.rationale == "Vetted"

def test_bo_result_schema():
    now = datetime.datetime.now().isoformat()
    b = BOResult(
        feature_name="f1",
        best_params={"alpha": ParameterSpec(name="alpha", lower=0, upper=1, type="float", init=0.5).model_dump()},
        best_value=0.12,
        study_name="study1",
        storage="sqlite:///test.db",
        n_trials=10,
        timestamp=now,
    )
    assert b.feature_name == "f1"
    assert b.best_params["alpha"]["lower"] == 0
    assert b.best_params["alpha"]["upper"] == 1
    assert b.best_params["alpha"]["type"] == "float"
    assert b.best_params["alpha"]["init"] == 0.5
    assert b.best_value == 0.12
    assert b.study_name == "study1"
    assert b.storage.startswith("sqlite://")
    assert b.n_trials == 10
    assert b.timestamp == now

def test_realized_feature_schema():
    now = datetime.datetime.now().isoformat()
    params = {"beta": ParameterSpec(name="beta", lower=1, upper=2, type="float", init=1.5)}
    rf = RealizedFeature(
        name="rf1",
        type="code",
        spec="x-y",
        feature_scope="item",
        depends_on=["c1"],
        parameters=params,
        rationale="Realized",
        best_params={"beta": params["beta"].model_dump()},
        best_value=0.01,
        bo_study_name="study_rf",
        bo_storage="sqlite:///rf.db",
        realization_timestamp=now,
    )
    assert rf.name == "rf1"
    assert rf.feature_scope == "item"
    assert rf.best_params["beta"]["lower"] == 1
    assert rf.best_params["beta"]["upper"] == 2
    assert rf.best_params["beta"]["type"] == "float"
    assert rf.best_params["beta"]["init"] == 1.5
    assert rf.best_value == 0.01
    assert rf.bo_study_name == "study_rf"
    assert rf.bo_storage.startswith("sqlite://")
    assert rf.realization_timestamp == now

# Test that schema validation catches missing required fields
@pytest.mark.parametrize("model, kwargs", [
    (Hypothesis, {"summary": "", "rationale": "", "depends_on": []}),
    (CandidateFeature, {"name": "", "type": "code", "spec": "", "feature_scope": "item", "rationale": ""}),
    (BOResult, {"feature_name": "", "best_params": {}, "best_value": 0, "study_name": "", "storage": "", "n_trials": 0, "timestamp": ""}),
    (RealizedFeature, {"name": "", "type": "code", "spec": "", "feature_scope": "item", "rationale": "", "parameters": {}, "best_params": {}, "best_value": 0, "bo_study_name": "", "bo_storage": "", "realization_timestamp": ""}),
])
def test_required_fields(model, kwargs):
    with pytest.raises(ValidationError):
        model(**kwargs)
