from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Profile projection/loading does not execute tensor code.  Keep this focused
# firewall suite runnable in lightweight audit environments without PyTorch.
try:
    import torch as _torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType("torch")
    torch_utils_stub = types.ModuleType("torch.utils")
    torch_data_stub = types.ModuleType("torch.utils.data")
    torch_data_stub.Dataset = object
    torch_utils_stub.data = torch_data_stub
    torch_stub.utils = torch_utils_stub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.utils"] = torch_utils_stub
    sys.modules["torch.utils.data"] = torch_data_stub

from crps import data as profile_data  # noqa: E402
from crps.contracts import (  # noqa: E402
    EXPECTED_TRAINING_PROFILE_PATIENTS,
    FORBIDDEN_TRAINING_COLUMN_TOKENS,
    LOCKED_ISPY1_PROFILE_SHA256,
    LOCKED_ISPY2_PROFILE_SHA256,
    LOCKED_TECHNICAL_ELIGIBILITY_SHA256,
    PCR_LABEL_ACCESS,
    PROFILE_SOURCE_USECOLS,
    TECHNICAL_ELIGIBILITY_USECOLS,
    TRAINING_PROFILE_COLUMNS,
    assert_representation_config,
    file_sha256,
    load_json,
)
from crps.data import (  # noqa: E402
    ConditionSpec,
    ProfiledStageBDataset,
    TrainingProfile,
    load_training_profiles,
)


SCRIPT = ROOT / "scripts" / "build_training_profiles.py"
SPEC = importlib.util.spec_from_file_location("goal_f_profile_projection", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
projection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(projection)

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _write_csv(path: Path, frame: pd.DataFrame) -> str:
    frame.to_csv(path, index=False)
    return file_sha256(path)


def _profile_frame(patient_ids: list[str], *, source: str) -> pd.DataFrame:
    count = len(patient_ids)
    payload = {
        "patient_id": patient_ids,
        "label_pcr": [f"SECRET_PCR_VALUE_{source}_{index}" for index in range(count)],
        "label_hr": [index % 2 for index in range(count)],
        "label_her2": [(index + 1) % 2 for index in range(count)],
        "label_mp": [index % 2 for index in range(count)],
        "arm": [f"therapy-{source}" for _ in range(count)],
        "rcb_class": [f"SECRET_RCB_{index}" for index in range(count)],
        "response_outcome": [f"SECRET_OUTCOME_{index}" for index in range(count)],
    }
    if source == "ispy1":
        order = (
            "patient_id",
            "label_pcr",
            "arm",
            "label_hr",
            "label_her2",
            "label_mp",
            "rcb_class",
            "response_outcome",
        )
        return pd.DataFrame(payload).loc[:, list(order)]
    return pd.DataFrame(payload)


def _assert_count_or_hash_leaves(test: unittest.TestCase, value: object) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _assert_count_or_hash_leaves(test, nested)
        return
    if isinstance(value, bool):
        test.fail("public provenance contains a boolean rather than a count/hash")
    if isinstance(value, int):
        return
    test.assertIsInstance(value, str)
    test.assertIsNotNone(SHA256_PATTERN.fullmatch(str(value)))


class ProjectionBoundaryTests(unittest.TestCase):
    def test_projection_uses_exact_usecols_and_drops_every_outcome_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ispy2 = root / "ispy2.csv"
            ispy1 = root / "ispy1.csv"
            eligibility = root / "eligibility.private.csv"
            ispy2_sha = _write_csv(
                ispy2,
                _profile_frame(["PATIENT_ALPHA", "PATIENT_BETA"], source="ispy2"),
            )
            ispy1_sha = _write_csv(
                ispy1,
                _profile_frame(["PATIENT_GAMMA"], source="ispy1"),
            )
            eligibility_sha = _write_csv(
                eligibility,
                pd.DataFrame(
                    {
                        "patient_id": [
                            "PATIENT_ALPHA",
                            "PATIENT_BETA",
                            "PATIENT_GAMMA",
                        ],
                        "eligible": [True, False, True],
                        "outcome_must_not_be_read": [1, 0, 1],
                    }
                ),
            )
            calls: list[tuple[str, ...]] = []
            original_read_csv = pd.read_csv

            def audited_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
                calls.append(tuple(kwargs.get("usecols", ())))
                return original_read_csv(*args, **kwargs)

            with mock.patch.object(
                projection.pd, "read_csv", side_effect=audited_read_csv
            ):
                frame, provenance = projection.project_training_profiles(
                    ispy2_source=ispy2,
                    ispy2_sha256=ispy2_sha,
                    ispy1_source=ispy1,
                    ispy1_sha256=ispy1_sha,
                    technical_eligibility=eligibility,
                    technical_eligibility_sha256=eligibility_sha,
                    expected_patient_count=2,
                )

            self.assertEqual(
                calls,
                [
                    PROFILE_SOURCE_USECOLS,
                    PROFILE_SOURCE_USECOLS,
                    TECHNICAL_ELIGIBILITY_USECOLS,
                ],
            )
            self.assertEqual(tuple(frame.columns), TRAINING_PROFILE_COLUMNS)
            self.assertEqual(
                frame["patient_id"].tolist(), ["PATIENT_ALPHA", "PATIENT_GAMMA"]
            )

            private_output = root / "training_profiles.private.csv"
            public_output = root / "training_profiles_provenance.json"
            written = projection.write_projection(
                frame,
                provenance,
                private_output=private_output,
                provenance_output=public_output,
            )
            private_text = private_output.read_text(encoding="utf-8")
            public_text = public_output.read_text(encoding="utf-8")
            self.assertEqual(
                private_text.splitlines()[0], ",".join(TRAINING_PROFILE_COLUMNS)
            )
            for forbidden in (
                "label_pcr",
                "secret_pcr",
                "rcb_class",
                "secret_rcb",
                "response_outcome",
                "secret_outcome",
            ):
                self.assertNotIn(forbidden, private_text.casefold())
                self.assertNotIn(forbidden, public_text.casefold())
            for private_value in (
                "PATIENT_ALPHA",
                "PATIENT_GAMMA",
                "therapy-ispy1",
                "therapy-ispy2",
                str(ispy1.resolve()),
                str(ispy2.resolve()),
            ):
                self.assertNotIn(private_value, public_text)
            self.assertEqual(stat.S_IMODE(private_output.stat().st_mode), 0o600)
            self.assertEqual(written["private_manifest_sha256"], file_sha256(private_output))
            _assert_count_or_hash_leaves(self, json.loads(public_text))

    def test_every_hash_is_verified_before_any_source_is_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ispy2 = root / "ispy2.csv"
            ispy1 = root / "ispy1.csv"
            eligibility = root / "eligibility.private.csv"
            ispy2_sha = _write_csv(ispy2, _profile_frame(["P1"], source="ispy2"))
            _write_csv(ispy1, _profile_frame(["P2"], source="ispy1"))
            eligibility_sha = _write_csv(
                eligibility,
                pd.DataFrame({"patient_id": ["P1"], "eligible": [True]}),
            )
            with mock.patch.object(projection.pd, "read_csv") as read_csv:
                with self.assertRaisesRegex(ValueError, "I-SPY1.*SHA-256 mismatch"):
                    projection.project_training_profiles(
                        ispy2_source=ispy2,
                        ispy2_sha256=ispy2_sha,
                        ispy1_source=ispy1,
                        ispy1_sha256="0" * 64,
                        technical_eligibility=eligibility,
                        technical_eligibility_sha256=eligibility_sha,
                        expected_patient_count=1,
                    )
            read_csv.assert_not_called()

    def test_projection_rejects_duplicate_source_or_eligibility_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ispy2 = root / "ispy2.csv"
            ispy1 = root / "ispy1.csv"
            eligibility = root / "eligibility.private.csv"
            ispy2_sha = _write_csv(
                ispy2, _profile_frame(["DUPLICATE", "DUPLICATE"], source="ispy2")
            )
            ispy1_sha = _write_csv(ispy1, _profile_frame(["P2"], source="ispy1"))
            eligibility_sha = _write_csv(
                eligibility,
                pd.DataFrame({"patient_id": ["DUPLICATE"], "eligible": [True]}),
            )
            with self.assertRaisesRegex(ValueError, "duplicate patient IDs"):
                projection.project_training_profiles(
                    ispy2_source=ispy2,
                    ispy2_sha256=ispy2_sha,
                    ispy1_source=ispy1,
                    ispy1_sha256=ispy1_sha,
                    technical_eligibility=eligibility,
                    technical_eligibility_sha256=eligibility_sha,
                    expected_patient_count=1,
                )

    def test_formal_cli_cannot_replace_preregistered_hashes_or_count(self) -> None:
        valid = argparse.Namespace(
            ispy2_sha256=LOCKED_ISPY2_PROFILE_SHA256,
            ispy1_sha256=LOCKED_ISPY1_PROFILE_SHA256,
            technical_eligibility_sha256=LOCKED_TECHNICAL_ELIGIBILITY_SHA256,
            expected_patient_count=EXPECTED_TRAINING_PROFILE_PATIENTS,
        )
        projection._assert_preregistered_locks(valid)
        for field, value in (
            ("ispy2_sha256", "0" * 64),
            ("ispy1_sha256", "1" * 64),
            ("technical_eligibility_sha256", "2" * 64),
            ("expected_patient_count", EXPECTED_TRAINING_PROFILE_PATIENTS + 1),
        ):
            with self.subTest(field=field):
                changed = argparse.Namespace(**vars(valid))
                setattr(changed, field, value)
                with self.assertRaises(PermissionError):
                    projection._assert_preregistered_locks(changed)

    def test_builder_atomically_binds_manifest_hash_and_preflights_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "representation.json"
            private_output = root / "training_profiles.private.csv"
            public_output = root / "training_profiles_provenance.json"
            config = load_json(ROOT / "configs" / "representation.json")
            config["profiles"]["training_manifest_path"] = str(private_output)
            config["profiles"]["training_manifest_sha256"] = None
            config_path.write_text(json.dumps(config), encoding="utf-8")

            count = EXPECTED_TRAINING_PROFILE_PATIENTS
            frame = pd.DataFrame(
                {
                    "patient_id": [f"P{index:06d}" for index in range(count)],
                    "label_hr": [index % 2 for index in range(count)],
                    "label_her2": [(index + 1) % 2 for index in range(count)],
                    "label_mp": [0 for _ in range(count)],
                    "arm": ["A" for _ in range(count)],
                },
                columns=TRAINING_PROFILE_COLUMNS,
            )
            provenance = {
                "schema_version": 1,
                "source_file_count": 2,
                "source_row_counts": {"ispy2": 808, "ispy1": 215},
                "source_sha256": {
                    "ispy2": LOCKED_ISPY2_PROFILE_SHA256,
                    "ispy1": LOCKED_ISPY1_PROFILE_SHA256,
                },
                "eligibility_manifest_sha256": (
                    LOCKED_TECHNICAL_ELIGIBILITY_SHA256
                ),
                "eligibility_candidate_count": 948,
                "eligibility_selected_count": count,
                "eligibility_excluded_count": 1,
                "projected_patient_count": count,
                "projected_patient_order_sha256": "a" * 64,
            }
            written = projection.write_projection_and_bind(
                frame,
                provenance,
                private_output=private_output,
                provenance_output=public_output,
                representation_config=config_path,
            )
            sealed = load_json(config_path)
            self.assertEqual(
                sealed["profiles"]["training_manifest_sha256"],
                written["private_manifest_sha256"],
            )
            self.assertEqual(file_sha256(private_output), written["private_manifest_sha256"])

            conflict_private = root / "conflict.private.csv"
            conflict_public = root / "conflict.json"
            sealed["profiles"]["training_manifest_path"] = str(conflict_private)
            sealed["profiles"]["training_manifest_sha256"] = "f" * 64
            config_path.write_text(json.dumps(sealed), encoding="utf-8")
            with self.assertRaisesRegex(PermissionError, "conflicts"):
                projection.write_projection_and_bind(
                    frame,
                    provenance,
                    private_output=conflict_private,
                    provenance_output=conflict_public,
                    representation_config=config_path,
                )
            self.assertFalse(conflict_private.exists())
            self.assertFalse(conflict_public.exists())


class FormalTrainingManifestTests(unittest.TestCase):
    def _valid_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "patient_id": "000001",
                    "label_hr": 1,
                    "label_her2": 0,
                    "label_mp": 1,
                    "arm": "A",
                },
                {
                    "patient_id": "000002",
                    "label_hr": 0,
                    "label_her2": 1,
                    "label_mp": 0,
                    "arm": "B",
                },
            ],
            columns=TRAINING_PROFILE_COLUMNS,
        )

    def test_loader_accepts_only_exact_private_schema_and_count_hash_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.private.csv"
            digest = _write_csv(path, self._valid_frame())
            calls: list[dict[str, object]] = []
            original_read_csv = pd.read_csv

            def audited_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
                calls.append(dict(kwargs))
                return original_read_csv(*args, **kwargs)

            with mock.patch.object(
                profile_data.pd, "read_csv", side_effect=audited_read_csv
            ):
                profiles, spec, provenance = load_training_profiles(
                    path,
                    digest,
                    expected_patient_ids=("000001", "000002"),
                )
            self.assertEqual(tuple(profiles), ("000001", "000002"))
            self.assertEqual(spec.arm_vocab, {"A": 0, "B": 1})
            self.assertEqual(calls[0].get("nrows"), 0)
            self.assertNotIn("usecols", calls[0])
            self.assertEqual(tuple(calls[1]["usecols"]), TRAINING_PROFILE_COLUMNS)
            _assert_count_or_hash_leaves(self, provenance)
            serialized = json.dumps(provenance, sort_keys=True)
            self.assertNotIn(str(path.resolve()), serialized)
            self.assertNotIn("000001", serialized)
            self.assertNotIn("arm_vocab", serialized)

    def test_loader_rejects_extra_outcome_reordered_and_duplicate_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = self._valid_frame()
            variants = {
                "outcome": valid.assign(label_pcr=[1, 0]),
                "response": valid.assign(response_score=[0.1, 0.2]),
                "benign_extra": valid.assign(age=[50, 60]),
                "reordered": valid.loc[
                    :, ["patient_id", "arm", "label_hr", "label_her2", "label_mp"]
                ],
            }
            for name, frame in variants.items():
                with self.subTest(name=name):
                    path = root / f"{name}.private.csv"
                    digest = _write_csv(path, frame)
                    with self.assertRaises(PermissionError):
                        load_training_profiles(path, digest)

            duplicate_header = root / "duplicate_header.private.csv"
            duplicate_header.write_text(
                "patient_id,label_hr,label_her2,label_mp,arm,arm\n"
                "P1,1,0,1,A,B\n",
                encoding="utf-8",
            )
            with self.assertRaises(PermissionError):
                load_training_profiles(duplicate_header, file_sha256(duplicate_header))

            duplicate_rows = root / "duplicate_rows.private.csv"
            duplicated = pd.concat((valid.iloc[[0]], valid.iloc[[0]]), ignore_index=True)
            duplicate_rows_sha = _write_csv(duplicate_rows, duplicated)
            with self.assertRaisesRegex(ValueError, "patient-unique"):
                load_training_profiles(duplicate_rows, duplicate_rows_sha)

    def test_loader_fails_closed_on_suffix_hash_cohort_and_firewall(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public_path = root / "profiles.csv"
            private_path = root / "profiles.private.csv"
            public_sha = _write_csv(public_path, self._valid_frame())
            private_sha = _write_csv(private_path, self._valid_frame())
            with self.assertRaisesRegex(ValueError, "owner-private"):
                load_training_profiles(public_path, public_sha)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_training_profiles(private_path, "0" * 64)
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                load_training_profiles(
                    private_path,
                    private_sha,
                    expected_patient_ids=("000001",),
                )
            with mock.patch.object(profile_data, "PCR_LABEL_ACCESS", "ALLOWED"):
                with self.assertRaisesRegex(PermissionError, "firewall"):
                    load_training_profiles(private_path, private_sha)

    def test_profiled_dataset_rejects_any_outcome_bearing_base_item(self) -> None:
        class UnsafeBase:
            patient_ids = ("P1",)
            transformed_ftv: dict[str, object] = {}

            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int) -> dict[str, object]:
                del index
                return {
                    "patient_id": "P1",
                    "image": object(),
                    "ftv_target": object(),
                    "ftv_mask": object(),
                    "label_pcr": 1,
                }

        profiles = {"P1": TrainingProfile("P1", 1, 0, 1, "A")}
        dataset = ProfiledStageBDataset(UnsafeBase(), profiles, ConditionSpec({"A": 0}, ()))
        with self.assertRaisesRegex(PermissionError, "schema"):
            dataset[0]


class StaticFirewallTests(unittest.TestCase):
    def test_config_and_contract_keep_the_forbidden_sentinel_and_source_locks(self) -> None:
        config = load_json(ROOT / "configs" / "representation.json")
        assert_representation_config(config)
        self.assertEqual(PCR_LABEL_ACCESS, "FORBIDDEN")
        self.assertEqual(
            tuple(config["profiles"]["forbidden_column_tokens"]),
            FORBIDDEN_TRAINING_COLUMN_TOKENS,
        )
        self.assertEqual(
            config["profiles"]["technical_eligibility_sha256"],
            LOCKED_TECHNICAL_ELIGIBILITY_SHA256,
        )

        changed = json.loads(json.dumps(config))
        changed["profiles"]["ispy2_sha256"] = "0" * 64
        with self.assertRaisesRegex(PermissionError, "I-SPY2"):
            assert_representation_config(changed)

    def test_projection_ast_contains_only_the_two_exact_csv_allowlists(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_csv"
        ]
        self.assertEqual(len(calls), 2)
        usecols_names: list[str] = []
        for call in calls:
            keyword = next((item for item in call.keywords if item.arg == "usecols"), None)
            self.assertIsNotNone(keyword)
            self.assertIsInstance(keyword.value, ast.Name)
            usecols_names.append(keyword.value.id)
        self.assertEqual(
            usecols_names,
            ["PROFILE_SOURCE_USECOLS", "TECHNICAL_ELIGIBILITY_USECOLS"],
        )

    def test_representation_runtime_has_no_static_outcome_field_access(self) -> None:
        paths = [ROOT / "src" / "crps" / "data.py"]
        for name in ("model.py", "losses.py", "training.py", "objective.py"):
            candidate = ROOT / "src" / "crps" / name
            if candidate.exists():
                paths.append(candidate)
        paths.extend(
            path
            for path in (ROOT / "scripts").glob("*.py")
            if "train" in path.stem.casefold() and path.name != SCRIPT.name
        )
        forbidden_fields = {
            "pcr",
            "label_pcr",
            "pcr_label",
            "raw_pcr",
            "rcb",
            "rcb_class",
            "outcome",
            "outcome_label",
        }
        forbidden_literal_fields = forbidden_fields.difference({"pcr"})
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Name, ast.Attribute)):
                    token = node.id if isinstance(node, ast.Name) else node.attr
                    self.assertNotIn(token.casefold(), forbidden_fields, str(path))
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertNotIn(
                        node.value.strip().casefold(), forbidden_literal_fields, str(path)
                    )
                if isinstance(node, ast.Subscript) and isinstance(
                    node.slice, ast.Constant
                ):
                    if isinstance(node.slice.value, str):
                        self.assertNotIn(
                            node.slice.value.strip().casefold(), forbidden_fields, str(path)
                        )
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"get", "pop", "setdefault"}
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    self.assertNotIn(
                        node.args[0].value.strip().casefold(), forbidden_fields, str(path)
                    )


if __name__ == "__main__":
    unittest.main()
