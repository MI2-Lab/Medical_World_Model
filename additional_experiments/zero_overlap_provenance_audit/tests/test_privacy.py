from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = EXPERIMENT_ROOT / "scripts" / "audit_public_artifacts.py"
SPEC = importlib.util.spec_from_file_location("zero_overlap_privacy", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
privacy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(privacy)


def _chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _png(*extra_chunks: tuple[bytes, bytes]) -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    pixel = zlib.compress(b"\x00\x00\x00\x00")
    middle = b"".join(_chunk(kind, payload) for kind, payload in extra_chunks)
    return (
        privacy.PNG_SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + middle
        + _chunk(b"IDAT", pixel)
        + _chunk(b"IEND", b"")
    )


class PrivacyScannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for name in privacy.PUBLIC_DIRECTORIES:
            (self.root / name).mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, relative: str, content: str | bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    def _findings(self) -> set[str]:
        result = privacy.scan_public_artifacts(self.root)
        return {item["finding"] for item in result["privacy_findings"]}

    def test_allows_expected_alias_relative_paths_distances_and_citation_urls(self) -> None:
        self._write(
            "reports/final_report.md",
            "# Audit\n\nCASE_ZERO_OVERLAP_001 separation: 123.4 mm.\n\n"
            "See [DICOM standard](https://dicom.nema.org/medical/dicom/current/"
            "output/html/part03.html). Relative artifact: `metrics/summary.json`.\n",
        )
        self._write(
            "metrics/summary.json",
            json.dumps(
                {
                    "case_alias": "CASE_ZERO_OVERLAP_001",
                    "minimum_separation_mm": 123.4,
                    "source_center_ras_t0_relative_mm": [0.0, -12.3, 5.5],
                    "scanner_model": "Public Example 1.2.3",
                    "citation": "https://doi.org/10.1000/example/path",
                }
            ),
        )
        self._write(
            "manifests/public_summary.csv",
            "case_alias,visit_pair,separation_mm\n"
            "CASE_ZERO_OVERLAP_001,T0-T1,123.4\n",
        )
        self._write(
            "figures/safe.png",
            _png(
                (b"tEXt", b"Title\x00Zero-overlap provenance audit"),
                (b"tEXt", b"Description\x00CASE_ZERO_OVERLAP_001 public schematic"),
            ),
        )

        result = privacy.scan_public_artifacts(self.root)

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["privacy_findings"], [])

    def test_rejects_each_forbidden_text_class_without_echoing_secret(self) -> None:
        fixtures = {
            "patient_identifier": "failed patient I-SPY2-12345",
            "dicom_uid": "series UID 1.2.840.113619.2.55.3.604688123.9",
            "absolute_path": "source was /home/researcher/raw/scan.dcm",
            "private_filename": "details are in case_identity_private.json",
            "raw_coordinate_vector": "source_center_ras: [12.0, -34.5, 78.9]",
            "unexpected_case_alias": "CASE_ZERO_OVERLAP_002",
        }
        for index, (expected, secret) in enumerate(fixtures.items()):
            with self.subTest(expected=expected):
                path = self._write(f"reports/finding_{index}.md", secret)
                result = privacy.scan_public_artifacts(self.root)
                names = {item["finding"] for item in result["privacy_findings"]}
                self.assertEqual(result["status"], "FAIL")
                self.assertIn(expected, names)
                self.assertNotIn(secret, json.dumps(result, sort_keys=True))
                path.unlink()

    def test_urls_exempt_only_absolute_path_syntax_not_sensitive_values(self) -> None:
        self._write(
            "reports/unsafe_urls.md",
            "\n".join(
                (
                    "https://example.org/I-SPY2-12345",
                    "https://example.org/1.2.840.113619.2.55.3.604688123.9",
                    "https://example.org/case_identity_private.json",
                    "https://example.org/CASE_ZERO_OVERLAP_002",
                )
            ),
        )

        findings = self._findings()

        self.assertIn("patient_identifier", findings)
        self.assertIn("dicom_uid", findings)
        self.assertIn("private_filename", findings)
        self.assertIn("unexpected_case_alias", findings)
        self.assertNotIn("absolute_path", findings)

    def test_json_and_csv_identifier_and_coordinate_fields_fail_closed(self) -> None:
        self._write(
            "metrics/unsafe.json",
            '{"patient_id": "P12345", "source_center_ras": [1, 2, 3]}',
        )
        self._write(
            "manifests/unsafe.csv",
            "patient_id,source_center_ras,result\nP12345,redacted,FAIL\n",
        )

        findings = self._findings()

        self.assertIn("identifier_field", findings)
        self.assertIn("raw_coordinate_field", findings)

    def test_explicit_private_directories_and_artifacts_are_not_opened(self) -> None:
        self._write(
            "reports/private/identity.md",
            "I-SPY2-12345 /home/researcher/raw 1.2.840.113619.2.55.3",
        )
        self._write(
            "metrics/case_private.json",
            '{"patient_id":"P12345","ipp":[1,2,3]}',
        )
        self._write("reports/public.md", "CASE_ZERO_OVERLAP_001: audit pending.\n")

        paths = [path.relative_to(self.root).as_posix() for path in privacy.public_paths(self.root)]
        result = privacy.scan_public_artifacts(self.root)

        self.assertEqual(paths, ["reports/public.md"])
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["private_artifacts_ignored"])

    def test_png_text_metadata_is_scanned(self) -> None:
        payload = b"Comment\x00patient I-SPY2-12345"
        self._write("figures/unsafe_metadata.png", _png((b"tEXt", payload)))

        result = privacy.scan_public_artifacts(self.root)

        self.assertEqual(result["status"], "FAIL")
        self.assertIn("patient_identifier", self._findings())
        self.assertNotIn("I-SPY2-12345", json.dumps(result, sort_keys=True))

    def test_png_unknown_chunk_byte_strings_are_scanned(self) -> None:
        self._write(
            "figures/unsafe_bytes.png",
            _png((b"raNd", b"exported from /data/private_workspace/raw/scan.dcm")),
        )

        self.assertIn("absolute_path", self._findings())

    def test_identifier_bearing_public_filename_is_redacted_in_output(self) -> None:
        secret_filename = "reports/patient_12345_summary.md"
        self._write(secret_filename, "audit pending\n")

        result = privacy.scan_public_artifacts(self.root)
        serialized = json.dumps(result, sort_keys=True)

        self.assertEqual(result["status"], "FAIL")
        self.assertIn("generic_patient_identifier", self._findings())
        self.assertNotIn(secret_filename, serialized)
        self.assertIn("REDACTED_PUBLIC_PATH_", serialized)

    def test_malformed_json_and_png_are_findings(self) -> None:
        self._write("metrics/broken.json", "{")
        self._write("figures/broken.png", privacy.PNG_SIGNATURE + b"truncated")

        findings = self._findings()

        self.assertIn("malformed_json", findings)
        self.assertIn("malformed_png", findings)

    def test_unexpected_public_file_type_fails_closed(self) -> None:
        self._write("reports/unscannable.bin", b"opaque")

        self.assertIn("unsupported_public_artifact_type", self._findings())

    def test_existing_gate_is_excluded_so_repeat_scans_are_current(self) -> None:
        self._write(
            "metrics/public_artifact_privacy_gate.json",
            '{"patient_id":"this old output must never scan itself"}',
        )
        self._write("reports/public.md", "CASE_ZERO_OVERLAP_001\n")

        result = privacy.scan_public_artifacts(self.root)

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(list(result["scanned_files_sha256"]), ["reports/public.md"])


if __name__ == "__main__":
    unittest.main()
