import optuna
from typing import List, Dict, Any, Callable

def paramspec_to_optuna(trial, param_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given a list of param specs (with name, lower, upper), return a dict of sampled values from Optuna trial.
    """
    params = {}
    for p in param_specs:
        name = p["name"]
        low = p["lower"]
        high = p["upper"]
        # Support float/int
        if p.get("type", "float") == "int":
            params[name] = trial.suggest_int(name, int(low), int(high))
        else:
            params[name] = trial.suggest_float(name, float(low), float(high))
    return params


def run_optuna_bo(
    param_specs: List[Dict[str, Any]],
    reward_fn: Callable[..., float],
    reward_fn_kwargs: Dict[str, Any],
    n_trials: int = 20,
    direction: str = "minimize",
    study_name: str = "",
    storage: str = "",
    seed: int = 0,
) -> optuna.Study:
    """
    Run Optuna BO given a param spec and a reward function.
    By default, minimizes the loss function (direction='minimize').
    reward_fn_kwargs are passed to the reward function each trial.
    Returns the Optuna study object.
    """
    def objective(trial):
        params = paramspec_to_optuna(trial, param_specs)
        return reward_fn(**params, **reward_fn_kwargs)

    sampler = optuna.samplers.TPESampler(seed=seed) if seed else None
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage if storage else None,
        sampler=sampler,
        load_if_exists=True if storage else False,
    )
    study.optimize(objective, n_trials=n_trials)
    return study


def save_study_best_params(study: optuna.Study, out_path: str):
    """
    Save best params and value from an Optuna study to a JSON file.
    """
    import json
    with open(out_path, "w") as f:
        json.dump({"params": study.best_params, "value": study.best_value}, f, indent=2)


def load_optuna_study(study_name: str, storage: str) -> optuna.Study:
    """
    Reload a persisted Optuna study from storage (e.g., SQLite URI).
    """
    return optuna.load_study(study_name=study_name, storage=storage)
