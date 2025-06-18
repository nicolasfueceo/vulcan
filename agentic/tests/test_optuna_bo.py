import tempfile
import os
import datetime
from agentic.bo.optuna_bo import run_optuna_bo, save_study_best_params, load_optuna_study
from agentic.core.models import BOResult, RealizedFeature, CandidateFeature, ParameterSpec
import json
import optuna

def test_run_optuna_bo_and_save_best_params():
    # Simple quadratic minimization
    def reward_fn(x):
        return (x - 2) ** 2
    param_specs = [{"name": "x", "lower": -5, "upper": 5}]
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = f"sqlite:///{tmpdir}/test_study.db"
        study_name = "test_study"
        study = run_optuna_bo(
            param_specs=param_specs,
            reward_fn=reward_fn,
            reward_fn_kwargs={},
            n_trials=50,
            direction="minimize",
            storage=storage,
            study_name=study_name,
        )
        # Should find x close to 2
        assert abs(study.best_params["x"] - 2) < 1.0
        # Save best params
        out_json = os.path.join(tmpdir, "best_params.json")
        save_study_best_params(study, out_json)
        with open(out_json) as f:
            data = json.load(f)
        assert "params" in data and "value" in data
        # Reload study
        loaded = load_optuna_study(study_name, storage)
        assert loaded.best_value == study.best_value

        # Create BOResult and RealizedFeature
        now = datetime.datetime.now().isoformat()
        bo_result = BOResult(
            feature_name="quadratic",
            best_params=study.best_params,
            best_value=study.best_value,
            study_name=study_name,
            storage=storage,
            n_trials=10,
            timestamp=now,
        )
        cf = CandidateFeature(
            name="quadratic",
            type="code",
            spec="(x-2)**2",
            feature_scope="item",
            depends_on=[],
            parameters={"x": ParameterSpec(name="x", lower=-5, upper=5, type="float")},
            rationale="Quadratic test feature",
        )
        rf = RealizedFeature(
            **cf.model_dump(),
            best_params=bo_result.best_params,
            best_value=bo_result.best_value,
            bo_study_name=bo_result.study_name,
            bo_storage=bo_result.storage,
            realization_timestamp=now,
        )
        assert rf.name == "quadratic"
        assert rf.best_params["x"] == study.best_params["x"]
        assert rf.best_value == study.best_value
        assert rf.bo_study_name == study_name
        assert rf.bo_storage == storage
