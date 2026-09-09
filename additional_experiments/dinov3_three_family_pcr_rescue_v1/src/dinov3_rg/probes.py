"""Matched outer-fold probes for radiomics transfer and FTV retention."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .contracts import FOLDS, PRIMARY_PATIENTS, load_protocol
from .data import FoldTargets, load_fold_frame, validate_state_archive


def safe_spearman(truth: np.ndarray, prediction: np.ndarray) -> float:
    a = np.asarray(truth, dtype=np.float64); b = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3 or np.ptp(a[valid]) <= 0 or np.ptp(b[valid]) <= 0:
        return float("nan")
    return float(spearmanr(a[valid], b[valid]).statistic)


def _state_payload(path: Path) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    patient_ids, state = validate_state_archive(path)
    with np.load(path, allow_pickle=False) as payload:
        direct = np.asarray(payload["radiomics_prediction"], dtype=np.float32)
    if direct.shape != (808, 4, 16): raise ValueError("direct radiomics head archive contract failed")
    return patient_ids, state, direct


def _rows(
    patient_ids: tuple[str, ...], state: np.ndarray, target: np.ndarray,
    target_mask: np.ndarray, visits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs=[]; ys=[]; pids=[]; visit_ids=[]
    for patient_index, patient_id in enumerate(patient_ids):
        for visit in range(visits):
            if not bool(target_mask[patient_index, visit]): continue
            one_hot = np.zeros(visits, dtype=np.float32); one_hot[visit] = 1
            xs.append(np.concatenate((state[patient_index, visit], one_hot)))
            ys.append(target[patient_index, visit]); pids.append(patient_id); visit_ids.append(visit)
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(pids), np.asarray(visit_ids)


def _fit_matched_ridge(
    x: np.ndarray, y: np.ndarray, patient_ids: np.ndarray,
    split_by_patient: dict[str, str], alphas: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, float]:
    split = np.asarray([split_by_patient[str(value)] for value in patient_ids])
    train = split == "train"; validation = split == "val"; test = split == "test"
    if not train.any() or not validation.any() or not test.any():
        raise RuntimeError("matched probe split is empty")
    scaler = StandardScaler().fit(x[train])
    transformed = scaler.transform(x)
    candidates=[]
    for alpha in alphas:
        model = Ridge(alpha=float(alpha)).fit(transformed[train], y[train])
        prediction = model.predict(transformed[validation])
        mse = float(np.mean(np.square(prediction - y[validation])))
        candidates.append((mse, -float(alpha), float(alpha)))
    selected = min(candidates)[2]
    refit = train | validation
    model = Ridge(alpha=selected).fit(transformed[refit], y[refit])
    return model.predict(transformed[test]), test, selected


def evaluate_matched_probes(
    *, seeds: Iterable[int], arms: Iterable[str],
    state_path: Callable[[int, int, str], Path], target_root: str | Path,
    fold_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one pooled-OOF row per seed/arm and public fold diagnostics."""
    folds = load_fold_frame() if fold_frame is None else fold_frame.copy()
    alphas = tuple(float(x) for x in load_protocol()["probe"]["ridge_alphas"])
    result=[]; diagnostics=[]
    for seed in map(int, seeds):
        for arm in map(str, arms):
            observed_ids=[]; rad_truth=[]; rad_mask=[]; rad_direct=[]; rad_probe=[]
            ftv_truth=[]; ftv_mask=[]; ftv_probe=[]; held_states=[]
            primary_ids: set[str] | None = None
            for fold in FOLDS:
                targets = FoldTargets.load(Path(target_root) / f"fold_{fold}_targets.private.npz")
                if primary_ids is None: primary_ids = set(targets.patient_ids)
                elif set(targets.patient_ids) != primary_ids: raise ValueError("target cohort differs across folds")
                ids, state, direct = _state_payload(state_path(seed, fold, arm))
                state_lookup = {value: i for i, value in enumerate(ids)}
                target_lookup = {value: i for i, value in enumerate(targets.patient_ids)}
                split_frame = folds.loc[folds["fold"].eq(fold)]
                split_by_patient = dict(zip(split_frame["patient_id"].astype(str), split_frame["split"].astype(str)))
                ordered_target_ids = tuple(targets.patient_ids)
                state_indices = np.asarray([state_lookup[value] for value in ordered_target_ids])
                aligned_state = state[state_indices]; aligned_direct = direct[state_indices]
                rad_x, rad_y, rad_pid, rad_visit = _rows(
                    ordered_target_ids, aligned_state, targets.radiomics, targets.radiomics_mask, 3
                )
                rad_prediction, rad_test, rad_alpha = _fit_matched_ridge(
                    rad_x, rad_y, rad_pid, split_by_patient, alphas
                )
                ftv_x, ftv_y, ftv_pid, ftv_visit = _rows(
                    ordered_target_ids, aligned_state, targets.ftv, targets.ftv_mask, 4
                )
                ftv_prediction, ftv_test, ftv_alpha = _fit_matched_ridge(
                    ftv_x, ftv_y, ftv_pid, split_by_patient, alphas
                )
                test_ids = tuple(
                    split_frame.loc[split_frame["split"].eq("test"), "patient_id"].astype(str)
                )
                eligible = tuple(value for value in test_ids if value in target_lookup)
                ti = np.asarray([target_lookup[value] for value in eligible])
                si = np.asarray([state_lookup[value] for value in eligible])
                observed_ids.extend(eligible); rad_truth.append(targets.radiomics[ti])
                rad_mask.append(targets.radiomics_mask[ti]); rad_direct.append(direct[si])
                ftv_truth.append(targets.ftv[ti]); ftv_mask.append(targets.ftv_mask[ti]); held_states.append(state[si])
                fold_rad_probe = np.zeros((len(eligible), 4, 16), np.float32)
                fold_ftv_probe = np.zeros((len(eligible), 4), np.float32)
                eligible_lookup = {value: i for i, value in enumerate(eligible)}
                for pred, pid, visit in zip(rad_prediction, rad_pid[rad_test], rad_visit[rad_test]):
                    fold_rad_probe[eligible_lookup[str(pid)], int(visit)] = pred
                for pred, pid, visit in zip(ftv_prediction, ftv_pid[ftv_test], ftv_visit[ftv_test]):
                    fold_ftv_probe[eligible_lookup[str(pid)], int(visit)] = pred
                rad_probe.append(fold_rad_probe); ftv_probe.append(fold_ftv_probe)
                diagnostics.append({"seed": seed, "fold": fold, "arm": arm,
                                    "radiomics_alpha": rad_alpha, "ftv_alpha": ftv_alpha,
                                    "held_out_patients": len(eligible)})
            if primary_ids is None or len(primary_ids) != PRIMARY_PATIENTS:
                raise RuntimeError("primary cohort contract failed")
            if len(observed_ids) != PRIMARY_PATIENTS or len(set(observed_ids)) != PRIMARY_PATIENTS or set(observed_ids) != primary_ids:
                raise RuntimeError("each primary patient must enter OOF exactly once")
            rt=np.concatenate(rad_truth); rm=np.concatenate(rad_mask); rd=np.concatenate(rad_direct); rp=np.concatenate(rad_probe)
            ft=np.concatenate(ftv_truth); fm=np.concatenate(ftv_mask); fp=np.concatenate(ftv_probe); st=np.concatenate(held_states)
            direct_rhos=[]; probe_rhos=[]
            for visit in range(3):
                for component in range(16):
                    valid=rm[:,visit]
                    direct_rhos.append(safe_spearman(rt[:,visit,component][valid], rd[:,visit,component][valid]))
                    probe_rhos.append(safe_spearman(rt[:,visit,component][valid], rp[:,visit,component][valid]))
            static=[safe_spearman(ft[:,v][fm[:,v]], fp[:,v][fm[:,v]]) for v in range(4)]
            delta=[]
            for visit in range(3):
                valid=fm[:,visit]&fm[:,visit+1]
                delta.append(safe_spearman((ft[:,visit+1]-ft[:,visit])[valid], (fp[:,visit+1]-fp[:,visit])[valid]))
            values={
                "seed": seed, "arm": arm, "held_out_patients": len(observed_ids),
                "direct_head_radiomics_macro_spearman": float(np.nanmean(direct_rhos)),
                "matched_probe_radiomics_macro_spearman": float(np.nanmean(probe_rhos)),
                "matched_probe_static_ftv_macro_spearman": float(np.nanmean(static)),
                "matched_probe_delta_ftv_macro_spearman": float(np.nanmean(delta)),
                "state_mean_sd": float(st.reshape(-1,192).std(0).mean()),
            }
            if not np.isfinite(list(values.values())[2:]).all(): raise RuntimeError("probe metrics non-finite")
            result.append(values)
    return pd.DataFrame(result), pd.DataFrame(diagnostics)


__all__ = ["evaluate_matched_probes", "safe_spearman"]
