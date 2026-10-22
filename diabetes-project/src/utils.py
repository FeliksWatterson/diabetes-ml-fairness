from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def get_tuning_space():
    """Trả miền tìm kiếm của dự án, với khoảng XGBoost theo paper II.D."""
    return {
        "LR": {"C": {"kind": "float", "low": 0.001, "high": 100.0, "log": True}},
        "RF": {
            "n_estimators": {"kind": "int", "low": 200, "high": 600, "step": 100},
            "max_depth": {"kind": "categorical", "choices": [None, 5, 10, 20]},
            "min_samples_leaf": {"kind": "int", "low": 1, "high": 20},
            "min_samples_split": {"kind": "int", "low": 2, "high": 20},
            "max_features": {"kind": "categorical", "choices": ["sqrt", 0.5, 1.0]},
        },
        "SVM RBF": {
            "C": {"kind": "float", "low": 0.01, "high": 100.0, "log": True},
            "gamma": {"kind": "float", "low": 0.0001, "high": 1.0, "log": True},
        },
        "XGBoost": {
            "learning_rate": {"kind": "float", "low": 0.01, "high": 0.20},
            "max_depth": {"kind": "int", "low": 3, "high": 9},
            "n_estimators": {"kind": "int", "low": 100, "high": 500},
            "subsample": {"kind": "float", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"kind": "float", "low": 0.6, "high": 1.0},
            "min_child_weight": {"kind": "int", "low": 1, "high": 10},
        },
    }


def build_tuning_pipeline(model_name, params, seed=42, svm_probability=False):
    """Tạo pipeline mới với trọng số lớp tính theo dữ liệu của lần fit."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    factories = {
        "LR": lambda: LogisticRegression(
            class_weight="balanced", solver="lbfgs", max_iter=2000,
            random_state=seed, **params,
        ),
        "RF": lambda: RandomForestClassifier(
            class_weight="balanced", random_state=seed, n_jobs=4, **params,
        ),
        "SVM RBF": lambda: SVC(
            kernel="rbf", class_weight="balanced", probability=svm_probability,
            random_state=seed, **params,
        ),
        "XGBoost": lambda: FitBalancedXGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            reg_alpha=0, reg_lambda=1, tree_method="hist",
            random_state=seed, n_jobs=4, **params,
        ),
    }
    return Pipeline([
        ("preprocessor", build_preprocessor(scale_numeric=model_name in ["LR", "SVM RBF"])),
        ("classifier", factories[model_name]()),
    ])


class FitBalancedXGBClassifier(XGBClassifier):
    """Tính scale_pos_weight từ y của từng lần fit."""

    def fit(self, X, y, **kwargs):
        if kwargs.get("sample_weight") is not None or kwargs.get("eval_set") is not None:
            raise ValueError("This pipeline does not use sample_weight or eval_set")
        labels = np.asarray(y)
        if not np.isin(labels, [0, 1]).all() or np.unique(labels).size != 2:
            raise ValueError("Both binary classes are required in the fit data")
        self.set_params(scale_pos_weight=float((labels == 0).sum() / (labels == 1).sum()))
        return super().fit(X, y, **kwargs)


def _write_tuning_json(path, value):
    import json
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _hash_tuning_frames(*frames):
    import hashlib
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def prepare_tuning_run(X, y, support, outer_splits, inner_splits_by_outer,
                       results_dir, seed=42, budget=30, pilot_params=None, paper_sha256=None):
    """Tạo mã lượt chạy từ dữ liệu, fold, cấu hình, code và phiên bản thư viện."""
    import hashlib
    import importlib.metadata
    import io
    import json

    assert X.index.equals(y.index) and X.index.equals(support.index)
    arrays = {}
    for fold, (train, test) in enumerate(outer_splits, start=1):
        arrays[f"outer_{fold}_train"] = train
        arrays[f"outer_{fold}_test"] = test
        for inner, (fit, valid) in enumerate(inner_splits_by_outer[fold], start=1):
            arrays[f"outer_{fold}_inner_{inner}_train"] = fit
            arrays[f"outer_{fold}_inner_{inner}_valid"] = valid
    split_hash = hashlib.sha256()
    for name, array in arrays.items():
        split_hash.update(name.encode())
        split_hash.update(np.asarray(array, dtype="int64").tobytes())
    config = {
        "schema": 2, "seed": seed, "trials_per_model_outer": budget,
        "objective": "mean_inner_roc_auc", "direction": "maximize",
        "sampler": "TPE", "startup_trials": 10, "multivariate": False,
        "sampler_seed_rule": "seed + 1000 * outer_fold + trial_number",
        "pruning": False, "study_jobs": 1, "estimator_jobs": 4,
        "budget_rule": f"{budget} COMPLETE trials; failed/interrupted are recorded separately",
        "space": get_tuning_space(), "columns": X.columns.tolist(),
        "dtypes": X.dtypes.astype(str).to_dict(), "rows": len(X),
        "data_sha256": _hash_tuning_frames(X, y, support),
        "outer_train_hashes": {
            str(fold): _hash_tuning_frames(X.iloc[train], y.iloc[train])
            for fold, (train, _) in enumerate(outer_splits, start=1)
        },
        "splits_sha256": split_hash.hexdigest(), "pilot_params": pilot_params,
        "paper_sha256": paper_sha256,
        "utils_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {name: importlib.metadata.version(name) for name in [
            "numpy", "pandas", "scikit-learn", "xgboost", "optuna", "pyarrow",
        ]},
        "fixed": {
            "LR": {"solver": "lbfgs", "max_iter": 2000, "class_weight": "balanced"},
            "RF": {"class_weight": "balanced", "n_jobs": 4},
            "SVM RBF": {"kernel": "rbf", "class_weight": "balanced",
                        "search_probability": False, "outer_probability": True},
            "XGBoost": {"objective": "binary:logistic", "eval_metric": "logloss",
                        "reg_alpha": 0, "reg_lambda": 1, "tree_method": "hist",
                        "scale_pos_weight": "computed from y on every fit"},
        },
        "thresholds": {"SVM RBF": "decision score >= 0", "others": "probability >= 0.5"},
    }
    identity = json.dumps(config, sort_keys=True, ensure_ascii=True).encode()
    run_id = hashlib.sha256(identity).hexdigest()[:16]
    directory = Path(results_dir) / "tuning" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "config.json"
    if manifest.exists():
        assert json.loads(manifest.read_text(encoding="utf-8")) == config
    else:
        _write_tuning_json(manifest, config)
    splits_file = directory / "splits.npz"
    if splits_file.exists():
        with np.load(splits_file) as saved:
            assert set(saved.files) == set(arrays)
            for key, array in arrays.items():
                np.testing.assert_array_equal(saved[key], array)
    else:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        splits_file.write_bytes(buffer.getvalue())
    return {"id": run_id, "directory": directory, "config": config}


def run_tuning_study(model_name, X_train, y_train, inner_splits, run,
                     outer_fold, phase="pilot", pilot_params=None):
    """Tạo hoặc tiếp tục study theo ngân sách trial COMPLETE."""
    import optuna
    import warnings
    from time import perf_counter
    from sklearn.metrics import roc_auc_score

    if phase not in ["pilot", "official"]:
        raise ValueError("phase must be pilot or official")
    assert X_train.index.equals(y_train.index)
    assert len(inner_splits) == 5
    assert _hash_tuning_frames(X_train, y_train) == run["config"]["outer_train_hashes"][str(outer_fold)]
    with np.load(run["directory"] / "splits.npz") as saved:
        for inner, (fit, valid) in enumerate(inner_splits, start=1):
            np.testing.assert_array_equal(fit, saved[f"outer_{outer_fold}_inner_{inner}_train"])
            np.testing.assert_array_equal(valid, saved[f"outer_{outer_fold}_inner_{inner}_valid"])
    if phase == "pilot":
        assert pilot_params == run["config"]["pilot_params"][model_name]
    budget = 1 if phase == "pilot" else run["config"]["trials_per_model_outer"]
    directory = run["directory"] / phase / f"{model_name.replace(' ', '_')}_outer{outer_fold}"
    directory.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"{run['id']}_{phase}_{model_name}_{outer_fold}",
        storage="sqlite:///" + (directory / "study.sqlite3").resolve().as_posix(),
        direction="maximize", load_if_exists=True, pruner=optuna.pruners.NopPruner(),
    )
    expected = {"run_id": run["id"], "phase": phase, "model": model_name,
                "outer_fold": outer_fold, "budget": budget}
    if study.user_attrs:
        assert study.user_attrs == expected, "Study configuration mismatch"
    else:
        for key, value in expected.items():
            study.set_user_attr(key, value)
    for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    if not study.trials and pilot_params is not None and phase == "pilot":
        study.enqueue_trial(pilot_params)

    def objective(trial):
        params = {}
        for name, spec in run["config"]["space"][model_name].items():
            arguments = {key: value for key, value in spec.items() if key != "kind"}
            params[name] = getattr(trial, "suggest_" + spec["kind"])(name, **arguments)
        fold_rows, warning_rows = [], []
        for inner_fold, (fit_idx, valid_idx) in enumerate(inner_splits, start=1):
            assert not np.intersect1d(fit_idx, valid_idx).size
            assert np.array_equal(np.sort(np.r_[fit_idx, valid_idx]), np.arange(len(X_train)))
            pipeline = build_tuning_pipeline(model_name, params, seed=run["config"]["seed"])
            y_fit = y_train.iloc[fit_idx]
            started = perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipeline.fit(X_train.iloc[fit_idx], y_fit)
            fit_seconds = perf_counter() - started
            warning_rows.extend({"inner_fold": inner_fold, "category": w.category.__name__,
                                 "message": str(w.message)} for w in caught)
            score = (pipeline.decision_function(X_train.iloc[valid_idx]) if model_name == "SVM RBF"
                     else pipeline.predict_proba(X_train.iloc[valid_idx])[:, 1])
            assert np.isfinite(score).all()
            row = {"inner_fold": inner_fold, "fit_n": len(fit_idx), "valid_n": len(valid_idx),
                   "negative": int(y_fit.eq(0).sum()), "positive": int(y_fit.eq(1).sum()),
                   "auc": float(roc_auc_score(y_train.iloc[valid_idx], score)),
                   "fit_seconds": fit_seconds}
            classifier = pipeline.named_steps["classifier"]
            if model_name == "XGBoost":
                row["scale_pos_weight"] = float(classifier.scale_pos_weight)
                assert np.isclose(row["scale_pos_weight"], row["negative"] / row["positive"])
            if model_name == "SVM RBF":
                assert classifier.probability is False
            fold_rows.append(row)
            trial.set_user_attr("inner_folds", fold_rows)
            trial.set_user_attr("warnings", warning_rows)
        return float(np.mean([row["auc"] for row in fold_rows]))

    def recorded_objective(trial):
        import traceback
        try:
            return objective(trial)
        except Exception:
            trial.set_user_attr("error", traceback.format_exc())
            raise

    completed_count = lambda: len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
    assert completed_count() <= budget, "Study already exceeds the fixed budget"
    while completed_count() < budget:
        trial_number = next((t.number for t in study.trials if t.state == optuna.trial.TrialState.WAITING), len(study.trials))
        study.sampler = optuna.samplers.TPESampler(
            seed=run["config"]["seed"] + 1000 * outer_fold + trial_number,
            n_startup_trials=10, multivariate=False, constant_liar=False,
        )
        try:
            study.optimize(recorded_objective, n_trials=1, n_jobs=1,
                           catch=(ValueError, FloatingPointError))
        finally:
            study.trials_dataframe().to_csv(directory / "trials.csv", index=False)
        if len(study.trials) >= 3 and all(t.state.name == "FAIL" for t in study.trials[-3:]):
            raise RuntimeError("Three consecutive failed trials; inspect saved errors before resuming")
    complete = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    if not complete:
        raise RuntimeError("No completed trial; inspect the saved study before resuming")
    best = study.best_trial
    failed = [{"trial": t.number, "error": t.user_attrs.get("error", "interrupted"),
               "params": str(t.params)} for t in study.trials if t.state.name == "FAIL"]
    pd.DataFrame(failed, columns=["trial", "error", "params"]).to_csv(directory / "failed_trials.csv", index=False)
    warnings_rows = [{"trial": t.number, **warning} for t in study.trials
                     for warning in t.user_attrs.get("warnings", [])]
    pd.DataFrame(warnings_rows, columns=["trial", "inner_fold", "category", "message"]).to_csv(
        directory / "inner_warnings.csv", index=False)
    _write_tuning_json(directory / "best_params.json", {
        **expected, "trial": best.number, "inner_auc": best.value, "params": best.params,
        "completed_trials": len(complete),
        "failed_trials": len(failed),
        "trial_seconds": sum((t.datetime_complete - t.datetime_start).total_seconds()
                             for t in study.trials if t.datetime_start and t.datetime_complete),
    })
    return study, directory


def evaluate_tuned_outer(model_name, study, directory, X_train, y_train,
                          X_test, y_test, support_test, seed=42, models_dir=None):
    """Lưu hoặc nạp pipeline và kết quả outer-test có cùng cấu hình."""
    import hashlib
    import json
    import joblib
    import warnings
    from time import perf_counter
    from sklearn.metrics import confusion_matrix
    assert study.user_attrs["phase"] == "official", "Pilot trials are not outer evaluation"
    complete = [t for t in study.trials if t.state.name == "COMPLETE"]
    if len(complete) != study.user_attrs["budget"]:
        raise ValueError("Official evaluation requires the full trial budget to complete")
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index) and X_test.index.equals(support_test.index)
    assert not X_train.index.intersection(X_test.index).size
    directory = Path(directory)
    models_dir = Path(models_dir) if models_dir else directory / "model"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{model_name.replace(' ', '_')}_outer{study.user_attrs['outer_fold']}.joblib"
    metadata_path = model_path.with_suffix(".json")
    expected = {
        **study.user_attrs, "seed": seed, "best_params": study.best_params,
        "train_sha256": _hash_tuning_frames(X_train, y_train),
        "test_sha256": _hash_tuning_frames(X_test, y_test, support_test),
        "columns": X_train.columns.tolist(),
    }
    marker = directory / "outer_complete.json"
    if marker.exists():
        checkpoint = json.loads(marker.read_text(encoding="utf-8"))
        assert checkpoint["identity"] == expected
        for path, digest in checkpoint["artifacts"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
        metrics = json.loads((directory / "outer_metrics.json").read_text(encoding="utf-8"))
        predictions = pd.read_parquet(directory / "outer_predictions.parquet")
        pd.testing.assert_frame_equal(predictions[["SEQN", "cycle"]], support_test[["SEQN", "cycle"]])
        pd.testing.assert_series_equal(predictions["y_true"], y_test.rename("y_true"))
        return metrics, predictions
    if metadata_path.exists():
        model_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert model_metadata["identity"] == expected
        assert hashlib.sha256(model_path.read_bytes()).hexdigest() == model_metadata["model_sha256"]
        model = joblib.load(model_path)
    else:
        model = build_tuning_pipeline(model_name, study.best_params, seed, svm_probability=True)
        started = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X_train, y_train)
        fit_seconds = perf_counter() - started
        temporary = model_path.with_suffix(".joblib.tmp")
        joblib.dump(model, temporary, compress=3)
        temporary.replace(model_path)
        model_metadata = {
            "identity": expected, "fit_seconds": fit_seconds,
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
        }
        _write_tuning_json(metadata_path, model_metadata)
    probability = model.predict_proba(X_test)[:, 1]
    score = model.decision_function(X_test) if model_name == "SVM RBF" else probability
    prediction = (score >= (0 if model_name == "SVM RBF" else 0.5)).astype(int)
    assert np.isfinite(score).all() and np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
    metrics = {
        **study.user_attrs, "best_params": study.best_params,
        "inner_auc": study.best_value, "outer_n": len(y_test),
        "fit_seconds": model_metadata["fit_seconds"],
        **get_model_metrics(y_test, score, probability, prediction),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "warnings": model_metadata["warnings"],
    }
    predictions = support_test[["SEQN", "cycle"]].copy()
    assert predictions["SEQN"].is_unique
    predictions["y_true"] = y_test
    predictions["score"] = score
    predictions["probability_1"] = probability
    predictions["prediction"] = prediction
    predictions["outer_fold"] = study.user_attrs["outer_fold"]
    predictions["run_id"] = study.user_attrs["run_id"]
    _write_tuning_json(Path(directory) / "outer_metrics.json", metrics)
    predictions.to_parquet(Path(directory) / "outer_predictions.parquet", index=True)
    artifacts = [model_path, metadata_path, directory / "outer_metrics.json", directory / "outer_predictions.parquet"]
    _write_tuning_json(marker, {"identity": expected, "artifacts": {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts
    }})
    return metrics, predictions


def read_baseline_snapshot(notebook_path):
    """Đọc metric baseline từ output notebook đã làm tròn 4 chữ số."""
    import json
    import re
    cells = json.loads(Path(notebook_path).read_text(encoding="utf-8"))["cells"]
    metrics = ["ROC-AUC", "Average precision", "Sensitivity", "Specificity", "Precision", "F1", "Brier score"]

    def text(cell):
        output = next(o["data"]["text/plain"] for o in cell["outputs"] if "data" in o)
        return "".join(output) if isinstance(output, list) else output

    rows = {}
    for token, default_model in [("lr_fold_metrics.round(4)", "LR"),
                                 ("xgb_fold_metrics.round(4)", "XGBoost"),
                                 ("pd.concat(rf_svm_fold_metrics, names=", None)]:
        cell = next(c for c in cells if c["cell_type"] == "code" and token in "".join(c["source"]))
        model_name = default_model
        for line in text(cell).splitlines():
            line = line.strip()
            prefix = re.match(r"^(RF|SVM RBF)\s+", line)
            if prefix:
                model_name = prefix.group(1)
                line = line[prefix.end():]
            match = re.fullmatch(r"([1-5])\s+((?:\d+\.\d+\s*)+)", line)
            if match:
                rows.setdefault((model_name, int(match.group(1))), []).extend(map(float, match.group(2).split()))
    assert len(rows) == 20 and all(len(values) == 7 for values in rows.values())
    folds = pd.DataFrame.from_dict(rows, orient="index", columns=metrics)
    folds.index = pd.MultiIndex.from_tuples(folds.index, names=["model", "outer_fold"])
    comparison_cell = next(c for c in cells if c["cell_type"] == "code" and "all_baseline_metrics =" in "".join(c["source"]))
    summary = {}
    for line in text(comparison_cell).splitlines():
        for metric in metrics:
            if line.strip().startswith(metric + " "):
                pairs = re.findall(r"(\d+\.\d+)\s*±\s*(\d+\.\d+)", line)
                summary.setdefault(metric, []).extend((float(mean), float(sd)) for mean, sd in pairs)
    assert all(len(summary[metric]) == 4 for metric in metrics)
    ordered_models = ["LR", "XGBoost", "RF", "SVM RBF"]
    stats = pd.DataFrame([
        {"model": model, "metric": metric, "mean": summary[metric][i][0], "std": summary[metric][i][1]}
        for i, model in enumerate(ordered_models) for metric in metrics
    ]).set_index(["model", "metric"])
    return folds.sort_index(), stats


def run_nested_cv(X, y, support, outer_splits, inner_splits_by_outer, run, models_dir):
    """Chạy nested CV và lưu kết quả từng outer fold."""
    import json
    from datetime import datetime, timezone
    from time import perf_counter

    assert _hash_tuning_frames(X, y, support) == run["config"]["data_sha256"]
    original_X, original_y = X.copy(deep=True), y.copy(deep=True)
    sessions_path = run["directory"] / "execution_sessions.json"
    sessions = json.loads(sessions_path.read_text()) if sessions_path.exists() else []
    session = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "seconds": 0}
    sessions.append(session)
    started = perf_counter()
    _write_tuning_json(sessions_path, sessions)
    try:
        for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            assert set(support.iloc[train_idx]["SEQN"]).isdisjoint(support.iloc[test_idx]["SEQN"])
            for model_name in get_tuning_space():
                study, directory = run_tuning_study(
                    model_name, X.iloc[train_idx], y.iloc[train_idx],
                    inner_splits_by_outer[fold], run, fold, phase="official",
                )
                metrics, _ = evaluate_tuned_outer(
                    model_name, study, directory, X.iloc[train_idx], y.iloc[train_idx],
                    X.iloc[test_idx], y.iloc[test_idx], support.iloc[test_idx],
                    seed=run["config"]["seed"], models_dir=Path(models_dir) / run["id"],
                )
                session["seconds"] = perf_counter() - started
                session["last_completed"] = {"model": model_name, "outer_fold": fold}
                _write_tuning_json(sessions_path, sessions)
                print(f"{model_name} | outer {fold}: 30 COMPLETE, ROC-AUC={metrics['ROC-AUC']:.4f}", flush=True)
        pd.testing.assert_frame_equal(X, original_X)
        pd.testing.assert_series_equal(y, original_y)
        session["status"] = "complete"
    except BaseException:
        session["status"] = "interrupted_or_error"
        raise
    finally:
        session["seconds"] = perf_counter() - started
        session["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_tuning_json(sessions_path, sessions)


def summarize_nested_cv(X, y, support, outer_splits, run, baseline_folds, baseline_stats):
    """Tổng hợp metric, OOF, thời gian và cảnh báo."""
    import json
    import optuna

    metric_names = baseline_folds.columns.tolist()
    fold_rows, trial_rows, parameter_rows, warning_rows, failed_rows = [], [], [], [], []
    oof_tables = {}
    for model_name in get_tuning_space():
        parts = []
        for fold, (_, test_idx) in enumerate(outer_splits, start=1):
            directory = run["directory"] / "official" / f"{model_name.replace(' ', '_')}_outer{fold}"
            assert (directory / "outer_complete.json").exists()
            metrics = json.loads((directory / "outer_metrics.json").read_text(encoding="utf-8"))
            best = json.loads((directory / "best_params.json").read_text(encoding="utf-8"))
            assert metrics["run_id"] == best["run_id"] == run["id"]
            assert metrics["best_params"] == best["params"]
            storage = "sqlite:///" + (directory / "study.sqlite3").resolve().as_posix()
            study = optuna.load_study(
                study_name=f"{run['id']}_official_{model_name}_{fold}", storage=storage,
            )
            complete = [t for t in study.trials if t.state.name == "COMPLETE"]
            assert len(complete) == run["config"]["trials_per_model_outer"]
            for trial in complete:
                assert len(trial.user_attrs["inner_folds"]) == 5
                assert np.isclose(trial.value, np.mean([r["auc"] for r in trial.user_attrs["inner_folds"]]))
                for row in trial.user_attrs["inner_folds"]:
                    if model_name == "XGBoost":
                        assert np.isclose(row["scale_pos_weight"], row["negative"] / row["positive"])
            predictions = pd.read_parquet(directory / "outer_predictions.parquet")
            assert predictions.index.equals(X.iloc[test_idx].index)
            pd.testing.assert_frame_equal(predictions[["SEQN", "cycle"]], support.iloc[test_idx][["SEQN", "cycle"]])
            pd.testing.assert_series_equal(predictions["y_true"], y.iloc[test_idx].rename("y_true"))
            assert predictions["outer_fold"].eq(fold).all() and predictions["run_id"].eq(run["id"]).all()
            expected_prediction = predictions["score"].ge(0) if model_name == "SVM RBF" else predictions["probability_1"].ge(0.5)
            np.testing.assert_array_equal(predictions["prediction"], expected_prediction.astype(int))
            parts.append(predictions)
            fold_rows.append({"model": model_name, "outer_fold": fold, **{m: metrics[m] for m in metric_names}})
            trial_rows.append({"model": model_name, "outer_fold": fold,
                               "complete": len(complete), "failed": best["failed_trials"],
                               "trial_seconds": best["trial_seconds"], "outer_fit_seconds": metrics["fit_seconds"]})
            parameter_rows.append({"model": model_name, "outer_fold": fold, "inner_auc": best["inner_auc"],
                                   "best_trial": best["trial"], "params": json.dumps(best["params"], sort_keys=True)})
            warning_rows.extend({"phase": "outer", "model": model_name, "outer_fold": fold,
                                 "trial": None, "inner_fold": None, **w} for w in metrics["warnings"])
            for trial in study.trials:
                warning_rows.extend({"phase": "inner", "model": model_name, "outer_fold": fold,
                                     "trial": trial.number, **w} for w in trial.user_attrs.get("warnings", []))
                if trial.state.name == "FAIL":
                    failed_rows.append({"model": model_name, "outer_fold": fold, "trial": trial.number,
                                        "error": trial.user_attrs.get("error", "interrupted")})
        combined = pd.concat(parts)
        assert combined.index.is_unique and combined["SEQN"].is_unique and len(combined) == len(X)
        assert set(combined.index) == set(X.index)
        combined = combined.reindex(X.index)
        assert combined[["score", "probability_1", "prediction"]].notna().all().all()
        assert np.isfinite(combined[["score", "probability_1"]]).all().all()
        assert combined["probability_1"].between(0, 1).all()
        pd.testing.assert_series_equal(combined["y_true"], y.rename("y_true"))
        combined.to_parquet(run["directory"] / f"oof_{model_name.replace(' ', '_')}.parquet", index=True)
        oof_tables[model_name] = combined
    folds = pd.DataFrame(fold_rows).set_index(["model", "outer_fold"]).sort_index()
    summary = folds.groupby(level="model").agg(["mean", "std"])
    comparisons = []
    for model_name in get_tuning_space():
        for metric in metric_names:
            baseline = baseline_stats.loc[(model_name, metric)]
            comparisons.append({"model": model_name, "metric": metric,
                                "baseline_mean": baseline["mean"], "baseline_std": baseline["std"],
                                "tuned_mean": summary.loc[model_name, (metric, "mean")],
                                "tuned_std": summary.loc[model_name, (metric, "std")]})
    comparison = pd.DataFrame(comparisons).set_index(["model", "metric"])
    trials = pd.DataFrame(trial_rows)
    warnings_table = pd.DataFrame(warning_rows, columns=["phase", "model", "outer_fold", "trial", "inner_fold", "category", "message"])
    failed = pd.DataFrame(failed_rows, columns=["model", "outer_fold", "trial", "error"])
    folds.to_csv(run["directory"] / "tuned_fold_metrics.csv")
    summary.to_csv(run["directory"] / "tuned_summary.csv")
    baseline_folds.to_csv(run["directory"] / "baseline_fold_metrics.csv")
    pd.concat({"baseline": baseline_folds, "tuned": folds}, names=["variant"]).to_csv(run["directory"] / "baseline_tuned_by_fold.csv")
    comparison.to_csv(run["directory"] / "baseline_tuned_comparison.csv")
    pd.DataFrame(parameter_rows).to_csv(run["directory"] / "best_params_by_fold.csv", index=False)
    trials.to_csv(run["directory"] / "trial_counts_times.csv", index=False)
    warnings_table.to_csv(run["directory"] / "warnings.csv", index=False)
    failed.to_csv(run["directory"] / "failed_trials.csv", index=False)
    sessions = json.loads((run["directory"] / "execution_sessions.json").read_text())
    runtime = {"run_id": run["id"], "complete_trials": int(trials["complete"].sum()),
               "failed_trials": len(failed), "inner_fits_complete": int(trials["complete"].sum()) * 5,
               "outer_pipelines": len(folds), "wall_seconds": sum(s["seconds"] for s in sessions),
               "trial_seconds": float(trials["trial_seconds"].sum()),
               "outer_fit_seconds": float(trials["outer_fit_seconds"].sum()),
               "warning_count": len(warnings_table),
               "convergence_warnings": int(warnings_table["category"].eq("ConvergenceWarning").sum()),
               "oof_people_per_model": {name: len(table) for name, table in oof_tables.items()},
               "baseline_precision": "recorded notebook outputs: four decimal places"}
    _write_tuning_json(run["directory"] / "runtime_summary.json", runtime)
    return {"fold_metrics": folds, "summary": summary, "comparison": comparison,
            "trials": trials, "warnings": warnings_table, "failed_trials": failed, "runtime": runtime}


def _sklearn_missing_array(values):
    """Đổi giá trị thiếu nullable sang np.nan trong mảng mới."""
    array = np.asarray(values, dtype=object)
    return np.where(pd.isna(array), np.nan, array)


def build_preprocessor(scale_numeric=False):
    """Tạo ColumnTransformer mới cho 8 đầu vào, với scaling tùy chọn."""
    numeric_steps = [
        ("missing", FunctionTransformer(
            _sklearn_missing_array, feature_names_out="one-to-one"
        )),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical = Pipeline([
        ("missing", FunctionTransformer(
            _sklearn_missing_array, feature_names_out="one-to-one"
        )),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore", drop=None, sparse_output=False
        )),
    ])
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), ["age", "bmi"]),
        ("categorical", categorical, [
            "sex", "race_ethnicity", "smoking_status",
            "physical_activity", "heart_attack", "stroke",
        ]),
    ], remainder="drop")


def load_xpt(path, columns=None):
    df = pd.read_sas(Path(path), format="xport")
    # Số 0 XPT có thể bị pandas đọc thành 2**-260.
    df = df.replace(2.0 ** -260, 0.0)
    return df if columns is None else df[columns].copy()


def _map_input_codes(series, mapping, unknown=()):
    """Ánh xạ mã hợp lệ và báo lỗi khi gặp mã ngoài miền."""
    unexpected = series.notna() & ~series.isin([*mapping, *unknown])
    if unexpected.any():
        raise ValueError(f"Unexpected codes in {series.name!r}: "
                         f"{series.loc[unexpected].unique().tolist()}")
    return series.map(mapping)


def map_nhanes_inputs(df):
    """Ánh xạ 8 đầu vào NHANES sang tên và nhóm dùng chung."""
    result = df[[
        "RIDAGEYR", "RIAGENDR", "RIDRETH3", "BMXBMI", "smoking_status",
        "physical_activity", "heart_attack", "stroke",
    ]].rename(columns={
        "RIDAGEYR": "age", "RIAGENDR": "sex",
        "RIDRETH3": "race_ethnicity", "BMXBMI": "bmi",
    }).copy()
    result["sex"] = _map_input_codes(df["RIAGENDR"], {1: 1, 2: 2}).astype("Int64")
    # RIDRETH3 không dùng mã 5, khác RIDRETH1.
    result["race_ethnicity"] = _map_input_codes(df["RIDRETH3"], {
        1: "Hispanic", 2: "Hispanic", 3: "White NH", 4: "Black NH",
        6: "Other", 7: "Other",
    }).astype("string")
    if not result["age"].dropna().between(18, 80).all():
        raise ValueError("NHANES adult age must already be topcoded at 80")
    for column, values in {
        "smoking_status": ["current", "former", "never"],
        "physical_activity": [0, 1], "heart_attack": [0, 1], "stroke": [0, 1],
    }.items():
        if not result[column].dropna().isin(values).all():
            raise ValueError(f"Unexpected values in {column}")
    return result


def map_brfss_inputs(df):
    """Ánh xạ 8 đầu vào BRFSS 2015/2017 sang bảng dùng chung."""
    age = df["_AGE80"].copy()
    if not age.dropna().between(18, 80).all():
        raise ValueError("BRFSS _AGE80 must be in 18-80")
    bmi = df["_BMI5"]
    if not bmi.dropna().between(1, 9999).all():
        raise ValueError("Unexpected _BMI5 value")
    return pd.DataFrame({
        "age": age,
        "sex": _map_input_codes(df["SEX"], {1: 1, 2: 2}, [9]).astype("Int64"),
        "race_ethnicity": _map_input_codes(df["_RACEGR3"], {
            1: "White NH", 2: "Black NH", 3: "Other", 4: "Other", 5: "Hispanic",
        }, [9]).astype("string"),
        "bmi": bmi / 100,
        "smoking_status": _map_input_codes(df["_SMOKER3"], {
            1: "current", 2: "current", 3: "former", 4: "never",
        }, [9]).astype("string"),
        "physical_activity": _map_input_codes(df["_PA150R2"], {
            1: 1, 2: 0, 3: 0,
        }, [9]).astype("Int64"),
        "heart_attack": get_yes_no(df["CVDINFR4"]),
        "stroke": get_yes_no(df["CVDSTRK3"]),
    }, index=df.index)


def get_brfss_diabetes_label(series):
    """Đổi DIABETE3 sang nhãn đái tháo đường ngoài thai kỳ, kiểu Int64 nullable."""
    return _map_input_codes(
        series, {1: 1, 2: 0, 3: 0, 4: 0}, unknown=[7, 9]
    ).astype("Int64").rename("diabetes_label")


def get_diabetes_label(df):
    """Trả nhãn nullable theo bằng chứng quan sát và giả định nhãn âm của dự án."""
    positive = (
        df["DIQ010"].eq(1)
        | df["DIQ050"].eq(1)
        | df["DIQ070"].eq(1)
        | df["LBXGH"].ge(6.5)
        | (df["glucose_usable"].eq(True) & df["LBXGLU"].ge(126))
    ).fillna(False)
    negative = (
        df["DIQ010"].isin([2, 3])
        & df["LBXGH"].notna()
        & df["LBXGH"].lt(6.5)
        & ~positive
    ).fillna(False)
    label = pd.Series(pd.NA, index=df.index, dtype="Int64", name="diabetes_label")
    label.loc[negative] = 0
    label.loc[positive] = 1
    return label


def get_yes_no(series):
    """Đổi mã Có/Không thành Int64 nullable; báo lỗi với mã ngoài miền."""
    unexpected = series.notna() & ~series.isin([1, 2, 7, 9])
    if unexpected.any():
        raise ValueError(
            f"Unexpected codes in {series.name!r}: "
            f"{series.loc[unexpected].unique().tolist()}"
        )
    result = pd.Series(pd.NA, index=series.index, dtype="Int64", name=series.name)
    result.loc[series.eq(1)] = 1
    result.loc[series.eq(2)] = 0
    return result


def get_smoking_status(df):
    """Trả Series never/former/current; trường hợp không xác định trả NA."""
    status = pd.Series(pd.NA, index=df.index, dtype="string", name="smoking_status")
    smoked = df["SMQ020"].eq(1)
    status.loc[df["SMQ020"].eq(2)] = "never"
    status.loc[smoked & df["SMQ040"].eq(3)] = "former"
    status.loc[smoked & df["SMQ040"].isin([1, 2])] = "current"
    return status


def get_physical_activity(df):
    """Trả thời lượng giải trí và trạng thái đạt ngưỡng, với trường hợp chưa rõ là NA."""
    def branch_minutes(question, days, minutes):
        result = pd.Series(pd.NA, index=df.index, dtype="Float64")
        result.loc[df[question].eq(2)] = 0.0
        valid = (
            df[question].eq(1)
            & df[days].isin(range(1, 8))
            & df[minutes].ge(10)
            & df[minutes].lt(1440)
        ).fillna(False)
        result.loc[valid] = df.loc[valid, days] * df.loc[valid, minutes]
        return result

    moderate = branch_minutes("PAQ665", "PAQ670", "PAD675")
    vigorous = branch_minutes("PAQ650", "PAQ655", "PAD660")
    weekly = moderate + 2 * vigorous
    # fillna(0) chỉ tính cận dưới của tổng, không điền thời lượng còn thiếu.
    known_minutes = moderate.fillna(0) + 2 * vigorous.fillna(0)
    status = pd.Series(pd.NA, index=df.index, dtype="Int64")
    status.loc[weekly.lt(150).fillna(False)] = 0
    status.loc[known_minutes.ge(150)] = 1
    return pd.DataFrame({
        "moderate_minutes": moderate,
        "vigorous_minutes": vigorous,
        "weekly_minutes": weekly,
        "physical_activity": status,
    })


def get_glucose_usable(df):
    """Trả cờ boolean nullable; thiếu điều kiện bắt buộc ưu tiên NA, kể cả khi điều kiện khác không đạt."""
    def no_recent_beverage(question, hours, minutes):
        answer = df[question]
        elapsed = df[hours] + df[minutes] / 60
        known_time = df[[hours, minutes]].notna().all(axis=1)
        result = pd.Series(pd.NA, index=df.index, dtype="boolean")
        # Câu đồ uống trả lời Không bỏ qua giờ/phút theo skip pattern.
        result.loc[answer.eq(2)] = True
        answered_yes = answer.eq(1) & known_time
        result.loc[answered_yes] = elapsed.loc[answered_yes].ge(8)
        return result

    coffee_ok = no_recent_beverage("PHQ020", "PHACOFHR", "PHACOFMN")
    alcohol_ok = no_recent_beverage("PHQ030", "PHAALCHR", "PHAALCMN")
    required = ["LBXGLU", "PHDSESN", "PHAFSTHR", "PHAFSTMN", "WTSAF2YR"]
    unknown = df[required].isna().any(axis=1) | coffee_ok.isna() | alcohol_ok.isna()
    food_fast_hours = df["PHAFSTHR"] + df["PHAFSTMN"] / 60
    usable = (
        df["LBXGLU"].notna()
        & df["PHDSESN"].eq(0)
        & food_fast_hours.between(8, 24, inclusive="left")
        & df["WTSAF2YR"].gt(0)
        & coffee_ok
        & alcohol_ok
    )
    return usable.astype("boolean").mask(unknown, pd.NA).rename("glucose_usable")


def prepare_final_run(X, y, support, splits, data_path, results_dir, models_dir,
                      seed=42, budget=30):
    """Gắn lượt phát triển cuối với dữ liệu, cấu hình và bộ chia CV."""
    import hashlib
    import importlib.metadata
    import json
    import platform

    assert X.index.is_unique and X.index.equals(y.index) and X.index.equals(support.index)
    assert support["SEQN"].notna().all() and support["SEQN"].is_unique
    assert y.notna().all() and y.isin([0, 1]).all()
    assert len(splits) == 5
    coverage = np.zeros(len(X), dtype=int)
    arrays = {}
    for fold, (train, valid) in enumerate(splits, 1):
        assert not np.intersect1d(train, valid).size
        assert np.array_equal(np.sort(np.r_[train, valid]), np.arange(len(X)))
        assert not set(support.iloc[train]["SEQN"]) & set(support.iloc[valid]["SEQN"])
        coverage[valid] += 1
        arrays[f"train_{fold}"], arrays[f"valid_{fold}"] = train, valid
    assert np.all(coverage == 1)
    split_hash = hashlib.sha256()
    for key, value in arrays.items():
        split_hash.update(key.encode())
        split_hash.update(np.asarray(value, dtype="int64").tobytes())
    config = {
        "stage": "final_development", "schema": 1,
        "roles": {"XGBoost": "primary", "LR": "reference"},
        "seed": seed, "budget_complete": budget,
        "cv": {"n_splits": 5, "shuffle": True, "strata": "diabetes_label x cycle"},
        "objective": "mean_cv_roc_auc", "direction": "maximize",
        "sampler": {"name": "TPE", "n_startup_trials": 10,
                    "multivariate": False, "seed_rule": "seed + trial_number"},
        "space": {model: get_tuning_space()[model] for model in ["XGBoost", "LR"]},
        "fixed": {
            "LR": {"class_weight": "balanced", "solver": "lbfgs", "max_iter": 2000},
            "XGBoost": {"objective": "binary:logistic", "eval_metric": "logloss",
                        "reg_alpha": 0, "reg_lambda": 1, "tree_method": "hist", "n_jobs": 4,
                        "scale_pos_weight": "negative / positive from y of each fit"},
        },
        "preprocessing": {"numeric": ["age", "bmi"], "numeric_imputer": "median",
                          "categorical_imputer": "most_frequent", "onehot_drop": None,
                          "onehot_unknown": "ignore", "scaled_models": ["LR"],
                          "missing_indicators": False, "winsorization": False},
        "early_stopping": False, "pruning": False, "calibration": None,
        "threshold_rule": "maximize Youden J over finite development OOF thresholds; ties: higher threshold",
        "prediction_rule": "probability >= threshold; same threshold for all groups",
        "threshold_tie_comparison": "exact integer TP*N - FP*P",
        "oof_use": "threshold development after CV parameter selection; not independent internal evaluation",
        "survey_weighted": False, "columns": X.columns.tolist(),
        "dtypes": X.dtypes.astype(str).to_dict(), "rows": len(X),
        "data_sha256": _hash_tuning_frames(X, y, support),
        "parquet_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "splits_sha256": split_hash.hexdigest(),
        "utils_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in [
            "numpy", "pandas", "scikit-learn", "xgboost", "optuna", "pyarrow", "joblib",
        ]},
    }
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(results_dir) / "final_development" / run_id
    model_directory = Path(models_dir) / "final" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text(encoding="utf-8")) == config
    else:
        _write_tuning_json(config_path, config)
    split_path = directory / "splits.npz"
    if split_path.exists():
        with np.load(split_path) as saved:
            assert set(saved.files) == set(arrays)
            for key, value in arrays.items():
                np.testing.assert_array_equal(saved[key], value)
    else:
        np.savez_compressed(split_path, **arrays)
    return {"id": run_id, "config": config, "directory": directory,
            "model_directory": model_directory}


def get_youden_threshold(y, probability):
    """Chọn ngưỡng hữu hạn có Youden J cao nhất; hòa thì lấy ngưỡng cao hơn."""
    from sklearn.metrics import roc_curve

    labels, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    assert np.isin(labels, [0, 1]).all() and np.unique(labels).size == 2
    assert labels.shape == probability.shape and np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    fpr, tpr, thresholds = roc_curve(labels, probability, drop_intermediate=False)
    positive, negative = int((labels == 1).sum()), int((labels == 0).sum())
    # So sánh tử số nguyên để hòa Youden không phụ thuộc sai số dấu phẩy động.
    numerator = np.rint(tpr * positive).astype("int64") * negative
    numerator -= np.rint(fpr * negative).astype("int64") * positive
    finite = np.isfinite(thresholds)
    best = finite & (numerator == numerator[finite].max())
    threshold = float(thresholds[best].max())
    curve = pd.DataFrame({"threshold": thresholds[finite], "sensitivity": tpr[finite],
                          "specificity": 1 - fpr[finite], "youden_j": (tpr - fpr)[finite],
                          "j_numerator": numerator[finite]})
    return threshold, curve


def fit_final_development(model_name, X, y, support, splits, run):
    """Tối ưu CV, chọn ngưỡng OOF và lưu pipeline fit toàn NHANES."""
    import hashlib
    import json
    import joblib
    import optuna
    import traceback
    import warnings
    from time import perf_counter
    from sklearn.metrics import roc_auc_score

    wall_started = perf_counter()
    config = run["config"]
    assert model_name in config["roles"]
    assert _hash_tuning_frames(X, y, support) == config["data_sha256"]
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == config["utils_sha256"]
    assert X.columns.tolist() == config["columns"]
    with np.load(run["directory"] / "splits.npz") as saved:
        for fold, (train, valid) in enumerate(splits, 1):
            np.testing.assert_array_equal(train, saved[f"train_{fold}"])
            np.testing.assert_array_equal(valid, saved[f"valid_{fold}"])
    directory = run["directory"] / model_name
    directory.mkdir(parents=True, exist_ok=True)
    model_path = run["model_directory"] / f"{model_name}.joblib"
    manifest = directory / "artifacts.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        assert saved["run_id"] == run["id"]
        for filename, digest in saved["sha256"].items():
            assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest
        return json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = optuna.storages.RDBStorage("sqlite:///" + (directory / "study.sqlite3").resolve().as_posix())
    study = optuna.create_study(
        study_name=f"{run['id']}_{model_name}_final", storage=storage,
        direction="maximize", load_if_exists=True, pruner=optuna.pruners.NopPruner(),
    )
    identity = {"run_id": run["id"], "model": model_name, "stage": "final_development"}
    if study.user_attrs:
        assert study.user_attrs == identity
    else:
        for key, value in identity.items():
            study.set_user_attr(key, value)
    for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    budget = config["budget_complete"]
    complete_count = lambda: len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
    assert complete_count() <= budget

    def fit_recorded(pipeline, train, phase, fold):
        started = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(X.iloc[train], y.iloc[train])
        elapsed = perf_counter() - started
        if model_name == "XGBoost":
            labels = y.iloc[train]
            assert np.isclose(pipeline.named_steps["classifier"].scale_pos_weight,
                              labels.eq(0).sum() / labels.eq(1).sum())
        records = [{"phase": phase, "fold": fold, "category": w.category.__name__,
                    "message": str(w.message)} for w in caught]
        return elapsed, records

    def objective(trial):
        try:
            params = {}
            for name, spec in config["space"][model_name].items():
                arguments = {k: v for k, v in spec.items() if k != "kind"}
                params[name] = getattr(trial, "suggest_" + spec["kind"])(name, **arguments)
            rows, warning_rows = [], []
            for fold, (train, valid) in enumerate(splits, 1):
                pipeline = build_tuning_pipeline(model_name, params, seed=config["seed"])
                elapsed, caught = fit_recorded(pipeline, train, "search", fold)
                probability = pipeline.predict_proba(X.iloc[valid])[:, 1]
                assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
                rows.append({"fold": fold, "auc": float(roc_auc_score(y.iloc[valid], probability)),
                             "fit_seconds": elapsed, "train_n": len(train), "valid_n": len(valid)})
                warning_rows.extend(caught)
                trial.set_user_attr("cv_folds", rows)
                trial.set_user_attr("warnings", warning_rows)
            return float(np.mean([row["auc"] for row in rows]))
        except Exception:
            trial.set_user_attr("error", traceback.format_exc())
            raise

    try:
        while complete_count() < budget:
            number = len(study.trials)
            study.sampler = optuna.samplers.TPESampler(
                seed=config["seed"] + number, n_startup_trials=10,
                multivariate=False, constant_liar=False,
            )
            try:
                study.optimize(objective, n_trials=1, n_jobs=1, catch=(ValueError, FloatingPointError))
            finally:
                study.trials_dataframe().to_csv(directory / "trials.csv", index=False)
            if len(study.trials) >= 3 and all(t.state.name == "FAIL" for t in study.trials[-3:]):
                raise RuntimeError("Three consecutive failed trials; inspect saved errors")
        assert complete_count() == budget
        best = study.best_trial
        trial_seconds = sum((t.datetime_complete - t.datetime_start).total_seconds()
                            for t in study.trials if t.datetime_start and t.datetime_complete)
        failed = [{"trial": t.number, "error": t.user_attrs.get("error", "interrupted")}
                  for t in study.trials if t.state.name == "FAIL"]
        _write_tuning_json(directory / "best_params.json", {
            **identity, "params": best.params, "best_trial": best.number,
            "mean_cv_auc": best.value, "complete_trials": complete_count(),
            "failed_trials": failed, "trial_seconds": trial_seconds,
        })
        warning_rows = [{"trial": t.number, **row} for t in study.trials
                        for row in t.user_attrs.get("warnings", [])]
        oof = np.full(len(X), np.nan)
        counts, fold_ids = np.zeros(len(X), dtype=int), np.zeros(len(X), dtype=int)
        oof_started = perf_counter()
        for fold, (train, valid) in enumerate(splits, 1):
            pipeline = build_tuning_pipeline(model_name, best.params, seed=config["seed"])
            _, caught = fit_recorded(pipeline, train, "development_oof", fold)
            warning_rows.extend({"trial": None, **row} for row in caught)
            oof[valid] = pipeline.predict_proba(X.iloc[valid])[:, 1]
            counts[valid] += 1
            fold_ids[valid] = fold
        oof_seconds = perf_counter() - oof_started
        assert np.all(counts == 1) and np.isfinite(oof).all()
        assert ((oof >= 0) & (oof <= 1)).all()
        threshold, curve = get_youden_threshold(y, oof)
        curve.to_csv(directory / "youden_candidates.csv", index=False)
        thresholds = {"default": 0.5, "youden": threshold,
                      "selection_rule": config["threshold_rule"],
                      "tie_comparison": config["threshold_tie_comparison"],
                      "prediction_rule": config["prediction_rule"], "use": config["oof_use"]}
        _write_tuning_json(directory / "thresholds.json", thresholds)
        development = support[["SEQN", "cycle"]].copy()
        development["diabetes_label"] = y
        development["fold"] = fold_ids
        development["score"] = oof
        development["probability"] = oof
        development["prediction_0_5"] = (oof >= 0.5).astype("int64")
        development["prediction_youden"] = (oof >= threshold).astype("int64")
        assert development.index.equals(X.index)
        oof_path = directory / "development_oof.parquet"
        development.to_parquet(oof_path, index=True)
        pd.testing.assert_frame_equal(pd.read_parquet(oof_path), development)
        final_pipeline = build_tuning_pipeline(model_name, best.params, seed=config["seed"])
        final_seconds, caught = fit_recorded(final_pipeline, np.arange(len(X)), "final_fit", None)
        warning_rows.extend({"trial": None, **row} for row in caught)
        before = final_pipeline.predict_proba(X)
        assert np.isfinite(before).all() and ((before >= 0) & (before <= 1)).all()
        assert final_pipeline.feature_names_in_.tolist() == config["columns"]
        np.testing.assert_array_equal(final_pipeline.classes_, [0, 1])
        temporary = model_path.with_suffix(".joblib.tmp")
        joblib.dump(final_pipeline, temporary)
        restored = joblib.load(temporary)
        np.testing.assert_array_equal(restored.predict_proba(X), before)
        np.testing.assert_array_equal(restored.predict(X), final_pipeline.predict(X))
        assert restored.feature_names_in_.tolist() == config["columns"]
        temporary.replace(model_path)
        np.testing.assert_array_equal(joblib.load(model_path).predict_proba(X), before)
        pd.DataFrame(warning_rows, columns=["trial", "phase", "fold", "category", "message"]).to_csv(
            directory / "warnings.csv", index=False)
        metadata = {**identity, "role": config["roles"][model_name], "config": config,
                    "best_params": best.params, "best_cv_auc_development": best.value,
                    "thresholds": thresholds, "complete_trials": complete_count(),
                    "failed_trials": len(failed), "warning_count": len(warning_rows),
                    "trial_seconds": trial_seconds, "oof_seconds": oof_seconds,
                    "final_fit_seconds": final_seconds,
                    "total_compute_seconds": trial_seconds + oof_seconds + final_seconds,
                    "wall_seconds": perf_counter() - wall_started,
                    "rows": len(X), "oof_coverage": "exactly once per person",
                    "serialization_check": "exact probability and prediction match on all NHANES rows",
                    "pipeline_path": str(model_path.resolve()), "oof_path": str(oof_path.resolve())}
        _write_tuning_json(directory / "metadata.json", metadata)
        _write_tuning_json(model_path.with_suffix(".metadata.json"), metadata)
        files = [p for p in directory.iterdir() if p.is_file() and p.name not in [
            "artifacts.json", "study.sqlite3", "study.sqlite3-shm", "study.sqlite3-wal",
        ]] + [model_path, model_path.with_suffix(".metadata.json")]
        _write_tuning_json(manifest, {"run_id": run["id"], "sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
        return metadata
    finally:
        storage.remove_session()
        storage.engine.dispose()


def check_external_inputs(source, columns, year):
    """Đối chiếu đầu vào BRFSS và khóa dòng nguồn trước khi dự đoán."""
    assert source.columns.is_unique and source.index.is_unique and source.index.notna().all()
    assert source["year"].eq(year).all()
    assert source["diabetes_label"].notna().all() and source["diabetes_label"].isin([0, 1]).all()
    assert {"_STATE", "_PSU", "_STSTR", "_LLCPWT", "DIABETE3"}.issubset(source.columns)
    assert source.columns[:8].tolist() == columns
    X = source.loc[:, columns].copy()
    assert X.columns.tolist() == columns and X.shape == (len(source), 8)
    for column in ["age", "bmi"]:
        assert np.isfinite(X[column].dropna().to_numpy(dtype=float)).all()
    age = X["age"].dropna()
    assert age.between(18, 80).all() and age.eq(np.floor(age)).all()
    assert X["bmi"].dropna().gt(0).all() and X["bmi"].dropna().le(99.99).all()
    domains = {
        "sex": [1, 2], "race_ethnicity": ["White NH", "Black NH", "Hispanic", "Other"],
        "smoking_status": ["never", "former", "current"], "physical_activity": [0, 1],
        "heart_attack": [0, 1], "stroke": [0, 1],
    }
    for column, values in domains.items():
        assert X[column].dropna().isin(values).all(), column
    assert X.index.equals(source["diabetes_label"].index)
    return X


def get_calibration_bins(y, probability, n_bins=10):
    """Trả số người, xác suất trung bình và tỷ lệ nhãn 1 theo quantile."""
    from sklearn.calibration import calibration_curve

    labels, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    edges = np.percentile(probability, np.linspace(0, 100, n_bins + 1))
    # Xác suất trùng ranh giới ở cùng nhóm, nên quantile có thể không đều người.
    group_ids = np.searchsorted(edges[1:-1], probability, side="left") + 1
    groups = pd.DataFrame({"group": group_ids, "probability": probability, "label": labels})
    groups = groups.groupby("group").agg(
        n=("label", "size"), mean_probability=("probability", "mean"),
        positive_rate=("label", "mean"),
    ).reindex(pd.Index(range(1, n_bins + 1), name="quantile_group"))
    groups["n"] = groups["n"].fillna(0).astype(int)
    groups["lower_bound"], groups["upper_bound"] = edges[:-1], edges[1:]
    observed_rate, mean_probability = calibration_curve(labels, probability, n_bins=n_bins, strategy="quantile")
    nonempty = groups["n"].gt(0)
    np.testing.assert_allclose(groups.loc[nonempty, "positive_rate"], observed_rate)
    np.testing.assert_allclose(groups.loc[nonempty, "mean_probability"], mean_probability)
    assert groups["n"].sum() == len(labels)
    return groups.reset_index()


def evaluate_external_pipeline(pipeline, source, X, thresholds, model_name, year):
    """Dự đoán với pipeline đã fit và tính metric ngoài mẫu, không áp trọng số."""
    import joblib
    from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

    assert X.index.equals(source.index) and X.columns.tolist() == pipeline.feature_names_in_.tolist()
    np.testing.assert_array_equal(pipeline.classes_, [0, 1])
    before_pipeline, before_source = joblib.hash(pipeline), _hash_tuning_frames(source)
    source_row_id = source.index.to_series(index=source.index, name="source_row_id")
    assert source_row_id.notna().all() and source_row_id.is_unique
    y = source["diabetes_label"].to_numpy(dtype=int)
    probability = pipeline.predict_proba(X)[:, 1]
    assert probability.shape == y.shape and np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    predictions = pd.DataFrame({
        "year": year, "source_row_id": source_row_id, "diabetes_label": source["diabetes_label"],
        "probability": probability,
        "prediction_0_5": (probability >= thresholds["default"]).astype(int),
        "prediction_youden": (probability >= thresholds["youden"]).astype(int),
    }, index=source.index)
    for column in ["sex", "age", "bmi", "race_ethnicity"]:
        predictions[column] = source[column]
    pd.testing.assert_series_equal(predictions["source_row_id"], source_row_id)
    pd.testing.assert_series_equal(predictions["diabetes_label"], source["diabetes_label"])
    assert len(predictions) == len(source) and not predictions.iloc[:, :6].isna().any().any()
    probability_metrics = {
        "year": year, "model": model_name, "n": len(y), "n_positive": int(y.sum()),
        "positive_rate": float(y.mean()), "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
    }
    threshold_rows = []
    for name, column in [("default", "prediction_0_5"), ("youden", "prediction_youden")]:
        threshold = float(thresholds[name])
        assert np.isfinite(threshold)
        tn, fp, fn, tp = map(int, confusion_matrix(y, predictions[column], labels=[0, 1]).ravel())
        threshold_rows.append({
            "year": year, "model": model_name, "threshold_name": name, "threshold": threshold,
            "n": len(y), "n_positive": int(y.sum()),
            "sensitivity": _safe_ratio(tp, tp + fn), "specificity": _safe_ratio(tn, tn + fp),
            "ppv": _safe_ratio(tp, tp + fp), "npv": _safe_ratio(tn, tn + fn),
            "f1": _safe_ratio(2 * tp, 2 * tp + fp + fn),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "predicted_positive_rate": _safe_ratio(tp + fp, len(y)),
        })
        assert tn + fp + fn + tp == len(source)
    assert joblib.hash(pipeline) == before_pipeline
    assert _hash_tuning_frames(source) == before_source
    groups = get_calibration_bins(y, probability).assign(year=year, model=model_name)
    return predictions, probability_metrics, threshold_rows, groups


def get_binary_group_metrics(y, probability, prediction):
    """Tính metric nhị phân cho một nhóm; metric không xác định trả NA."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    labels = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = np.asarray(prediction, dtype=int)
    assert labels.ndim == 1 and labels.shape == probability.shape == prediction.shape
    assert np.isin(labels, [0, 1]).all() and np.isin(prediction, [0, 1]).all()
    assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
    n, positive = len(labels), int(labels.sum())
    negative = n - positive
    tn = int(((labels == 0) & (prediction == 0)).sum())
    fp = int(((labels == 0) & (prediction == 1)).sum())
    fn = int(((labels == 1) & (prediction == 0)).sum())
    tp = int(((labels == 1) & (prediction == 1)).sum())
    return {
        "n": n, "n_positive": positive, "n_negative": negative,
        "positive_rate": _safe_ratio(positive, n), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "tpr": _safe_ratio(tp, positive), "fpr": _safe_ratio(fp, negative),
        "specificity": _safe_ratio(tn, negative), "ppv": _safe_ratio(tp, tp + fp),
        "npv": _safe_ratio(tn, tn + fn), "f1": _safe_ratio(2 * tp, 2 * tp + fp + fn),
        "roc_auc": float(roc_auc_score(labels, probability)) if positive and negative else np.nan,
        "average_precision": float(average_precision_score(labels, probability)) if positive else np.nan,
        "brier": float(brier_score_loss(labels, probability)) if n else np.nan,
        "predicted_positive_rate": _safe_ratio(tp + fp, n),
    }


def get_sex_metric_gaps(group_metrics):
    """Trả chênh lệch Nam − Nữ và equalized odds gap, đơn vị điểm phần trăm."""
    rows = []
    for (year, model), groups in group_metrics.groupby(["year", "model"], sort=False):
        groups = groups.set_index("sex_group")
        assert groups.index.is_unique and {"Nam", "Nữ"}.issubset(groups.index)
        row = {"year": year, "model": model}
        for metric in ["tpr", "fpr", "ppv", "predicted_positive_rate"]:
            male, female = groups.loc["Nam", metric], groups.loc["Nữ", metric]
            row[f"{metric}_gap_pp"] = float((male - female) * 100) if pd.notna(male) and pd.notna(female) else np.nan
        gaps = [row["tpr_gap_pp"], row["fpr_gap_pp"]]
        row["equalized_odds_gap_pp"] = max(map(abs, gaps)) if np.isfinite(gaps).all() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def get_transformer_feature_map(preprocessor):
    """Ánh xạ cột sau biến đổi về biến gốc từ cấu trúc ColumnTransformer đã fit."""
    names = preprocessor.get_feature_names_out()
    rows = []
    for name, transformer, columns in preprocessor.transformers_:
        output = preprocessor.output_indices_[name]
        if output.start == output.stop:
            continue
        assert all(isinstance(column, str) for column in columns)
        encoder = transformer.named_steps.get("onehot")
        if encoder is None:
            assert output.stop - output.start == len(columns)
            pairs = [(column, None) for column in columns]
        else:
            assert encoder.drop_idx_ is None
            assert encoder.min_frequency is None and encoder.max_categories is None
            assert len(encoder.categories_) == len(columns)
            pairs = [(column, category) for column, categories in zip(columns, encoder.categories_)
                     for category in categories]
            assert len(pairs) == output.stop - output.start
        for position, (column, category) in zip(range(output.start, output.stop), pairs):
            rows.append({"transformed_index": position, "transformed_feature": names[position],
                         "original_feature": column, "transformer": name, "category": category})
    mapping = pd.DataFrame(rows).sort_values("transformed_index").reset_index(drop=True)
    np.testing.assert_array_equal(mapping["transformed_index"], np.arange(len(names)))
    assert set(mapping["original_feature"]) == set(preprocessor.feature_names_in_)
    return mapping


def explain_xgboost_pipeline(pipeline, X):
    """Tính TreeSHAP raw margin và cộng có dấu về biến gốc, không fit lại pipeline."""
    import joblib
    from scipy.special import expit
    from xgboost import DMatrix

    before, original = joblib.hash(pipeline), X.copy(deep=True)
    preprocessor, classifier = pipeline.named_steps["preprocessor"], pipeline.named_steps["classifier"]
    assert X.columns.tolist() == pipeline.feature_names_in_.tolist() and X.index.is_unique
    mapping = get_transformer_feature_map(preprocessor)
    transformed = preprocessor.transform(X)
    assert transformed.shape == (len(X), len(mapping)) and np.isfinite(transformed).all()
    booster = classifier.get_booster()
    assert booster.num_features() == len(mapping)
    matrix = DMatrix(transformed, feature_names=mapping["transformed_feature"].tolist())
    contributions = booster.predict(matrix, pred_contribs=True, approx_contribs=False)
    margin = booster.predict(matrix, output_margin=True).astype(float)
    probability = pipeline.predict_proba(X)[:, 1].astype(float)
    assert contributions.shape == (len(X), len(mapping) + 1) and np.isfinite(contributions).all()
    shap_transformed = pd.DataFrame(contributions[:, :-1].astype(float), index=X.index,
                                    columns=mapping["transformed_feature"])
    shap_transformed.columns.name = None
    base = contributions[:, -1].astype(float)
    grouped = pd.DataFrame({column: shap_transformed.iloc[:, mapping.loc[
        mapping["original_feature"].eq(column), "transformed_index"
    ]].sum(axis=1) for column in X.columns}, index=X.index)
    reconstructed = base + shap_transformed.to_numpy().sum(axis=1)
    reconstructed_grouped = base + grouped.to_numpy().sum(axis=1)
    np.testing.assert_allclose(reconstructed, margin, rtol=0, atol=1e-5)
    np.testing.assert_allclose(reconstructed_grouped, margin, rtol=0, atol=1e-5)
    np.testing.assert_allclose(expit(margin), probability, rtol=0, atol=1e-7)
    np.testing.assert_allclose(grouped.to_numpy().sum(axis=1), shap_transformed.to_numpy().sum(axis=1),
                               rtol=0, atol=1e-12)
    pd.testing.assert_frame_equal(X, original)
    assert joblib.hash(pipeline) == before
    scores = pd.DataFrame({"base_value": base, "raw_margin": margin, "probability": probability,
                           "sum_shap": grouped.sum(axis=1),
                           "additivity_error": reconstructed_grouped - margin}, index=X.index)
    checks = pd.DataFrame([
        {"check": "base + transformed SHAP vs raw margin", "max_abs_error": float(np.max(abs(reconstructed - margin))),
         "absolute_tolerance": 1e-5},
        {"check": "base + grouped SHAP vs raw margin", "max_abs_error": float(np.max(abs(reconstructed_grouped - margin))),
         "absolute_tolerance": 1e-5},
        {"check": "sigmoid(raw margin) vs predict_proba", "max_abs_error": float(np.max(abs(expit(margin) - probability))),
         "absolute_tolerance": 1e-7},
    ])
    return {"transformed": pd.DataFrame(transformed, index=X.index, columns=shap_transformed.columns),
            "mapping": mapping, "shap_transformed": shap_transformed,
            "shap_grouped": grouped, "scores": scores, "checks": checks}


def _safe_ratio(numerator, denominator):
    """Chia hai số; mẫu số 0 trả NA."""
    return float(numerator / denominator) if denominator else np.nan


def get_model_metrics(y, score, probability, prediction):
    """Tính bảy metric dùng chung cho baseline và nested CV."""
    from sklearn.metrics import (
        average_precision_score, brier_score_loss, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score,
    )

    tn, fp, _, _ = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "ROC-AUC": float(roc_auc_score(y, score)),
        "Average precision": float(average_precision_score(y, score)),
        "Sensitivity": float(recall_score(y, prediction, zero_division=0)),
        "Specificity": float(tn / (tn + fp)),
        "Precision": float(precision_score(y, prediction, zero_division=0)),
        "F1": float(f1_score(y, prediction, zero_division=0)),
        "Brier score": float(brier_score_loss(y, probability)),
    }


def save_png(figure, path):
    """Xuất hình PNG 300 dpi và trả đường dẫn."""
    path = Path(path)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    return path
