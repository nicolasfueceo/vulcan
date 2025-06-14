# tests/agents/strategy_team/test_evaluation_agent.py
from src.agents.strategy_team.evaluation_agent import EvaluationAgent
from src.utils.session_state import SessionState


def test_evaluation_agent_instantiation():
    agent = EvaluationAgent()
    assert agent is not None


def test_evaluation_agent_run_smoke(tmp_path):
    # Minimal SessionState mock for smoke test
    session_state = SessionState(run_dir=tmp_path)
    agent = EvaluationAgent()
    # Should not raise (will do nothing as no optimization results)
    agent.run(session_state)


def test_evaluation_agent_run_end_to_end(monkeypatch, tmp_path):
    import pandas as pd
    import numpy as np
    from types import SimpleNamespace
    from src.agents.strategy_team.evaluation_agent import EvaluationAgent
    # Create a fake SessionState with all required attributes
    class DummySession(SimpleNamespace):
        def __init__(self, run_dir):
            self.run_dir = run_dir
            self.db_path = run_dir / "fake.duckdb"
            self._state = {}
        def get_state(self, key, default=None):
            return self._state.get(key, default)
        def set_state(self, key, value):
            self._state[key] = value
    # Fake optimization result and features
    best_trial = SimpleNamespace(params={"f1__param": 1.0}, value=0.123)
    opt_results = {"best_trial": best_trial}
    realized_features = [{"name": "f1"}]
    # Patch CVDataManager to return toy data
    class DummyCVDataManager:
        def __init__(self, *a, **k): pass
        def get_fold_summary(self): return {"n_folds": 1}
        def get_fold_data(self, fold_idx, split_type):
            df = pd.DataFrame({
                "user_id": [1, 2],
                "book_id": [10, 20],
            })
            return df, df
    monkeypatch.setattr("src.data.cv_data_manager.CVDataManager", DummyCVDataManager)
    # Patch feature matrix generation
    monkeypatch.setattr("src.agents.strategy_team.optimization_agent_v2.VULCANOptimizer._generate_feature_matrix", lambda df, features, params: pd.DataFrame({"f1": [1.0]*len(df)}, index=df["user_id"]))
    # Patch LightFM-related calls
    class DummyDataset:
        def fit(self, users, items): pass
        def build_interactions(self, pairs):
            from scipy.sparse import coo_matrix
            return coo_matrix(np.ones((2,2))), None
        def build_user_features(self, tuples):
            from scipy.sparse import csr_matrix
            return csr_matrix(np.ones((2,1)))
    monkeypatch.setattr("lightfm.data.Dataset", DummyDataset)
    monkeypatch.setattr("src.evaluation.scoring._train_and_evaluate_lightfm", lambda dataset, train_df, test_interactions, user_features=None, k=10, batch_size=100000: {f"precision_at_{k}": 1.0, f"recall_at_{k}": 1.0, f"hit_rate_at_{k}": 1.0})
    monkeypatch.setattr("src.evaluation.clustering.cluster_users_kmeans", lambda X, n_clusters=5, random_state=42: {uid: 0 for uid in X.index})
    monkeypatch.setattr("src.evaluation.beyond_accuracy.compute_novelty", lambda recs, train_df: 0.5)
    monkeypatch.setattr("src.evaluation.beyond_accuracy.compute_diversity", lambda recs: 0.5)
    monkeypatch.setattr("src.evaluation.beyond_accuracy.compute_catalog_coverage", lambda recs, catalog: 0.5)
    # Patch LightFM so fit and predict are safe
    class DummyLightFM:
        def __init__(self, *a, **k): pass
        def fit(self, *a, **k): return self
        def predict(self, i, items, user_features=None):
            import numpy as np
            return np.arange(len(items))[::-1]  # descending order
    # Patch LightFM globally (both in test and agent module)
    import sys
    sys.modules["lightfm.LightFM"] = DummyLightFM
    monkeypatch.setattr("lightfm.LightFM", DummyLightFM)
    # Setup dummy session
    session = DummySession(run_dir=tmp_path)
    session._state["optimization_results"] = opt_results
    session._state["realized_features"] = realized_features
    agent = EvaluationAgent()
    # DummySession implements get_state/set_state and mimics SessionState for testing
    agent.run(session)
    # Check that final_evaluation_metrics is set
    metrics = session.get_state("final_evaluation_metrics")
    assert metrics is not None
    assert "global" in metrics and "clusters" in metrics
    # Check that final_report.json is written
    report_path = tmp_path / "artifacts" / "final_report.json"
    assert report_path.exists()
    with open(report_path) as f:
        report = f.read()
    assert "global_metrics" in report
