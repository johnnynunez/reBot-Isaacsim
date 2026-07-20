#!/usr/bin/env python3
"""Regression tests for dynamic evidence and VALIDATION.md generation."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from dynamic_evidence_contract import dynamic_evidence_problems  # noqa: E402


class ValidationEvidenceTests(unittest.TestCase):
    def load_report(self, engine):
        path = PACKAGE_DIR / f"evidence/physics_fidelity_dynamic_{engine}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def copy_minimal_package(self, destination):
        destination.mkdir(parents=True)
        (destination / "scripts").mkdir()
        (destination / "evidence").mkdir()
        for name in (
            "dynamic_evidence_contract.py",
            "make_validation_md.py",
            "validate_dynamic_physics.py",
        ):
            shutil.copy2(SCRIPT_DIR / name, destination / "scripts" / name)
        for source in (PACKAGE_DIR / "evidence").rglob("*.json"):
            target = destination / "evidence" / source.relative_to(PACKAGE_DIR / "evidence")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for source in PACKAGE_DIR.rglob("*"):
            if source.is_file() and source.suffix.lower() in {".usd", ".usda", ".usdc"}:
                target = destination / source.relative_to(PACKAGE_DIR)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

    def run_generator(self, package):
        return subprocess.run(
            [sys.executable, str(package / "scripts/make_validation_md.py")],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_early_failure_removes_preexisting_pass_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dynamic.json"
            output.write_text('{"passed": true, "marker": "stale"}\n', encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "validate_dynamic_physics.py"),
                    str(Path(directory) / "missing.usda"),
                    "newton",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_current_reports_satisfy_shared_contract(self):
        for engine in ("newton", "physx"):
            with self.subTest(engine=engine):
                self.assertEqual(dynamic_evidence_problems(self.load_report(engine), engine), [])

    def test_shared_contract_rejects_adversarial_mutations(self):
        mutations = {
            "position flag": {"positions_within_limits": False},
            "step count": {"physics_steps_advanced": 1},
            "published hold metric": {"max_angular_hold_error_rad": 1.518},
            "passive motion": {"passive_motion_joint3_rad": 0.0},
            "widened tolerance": {"hold_error_tolerances": [10.0] * 8},
        }
        original = self.load_report("newton")
        for name, values in mutations.items():
            with self.subTest(mutation=name):
                report = deepcopy(original)
                report.update(values)
                self.assertNotEqual(dynamic_evidence_problems(report, "newton"), [])

    def test_generator_rejects_contradictory_pass_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / PACKAGE_DIR.name
            self.copy_minimal_package(package)
            path = package / "evidence/physics_fidelity_dynamic_newton.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report.update(
                {
                    "positions_within_limits": False,
                    "physics_steps_advanced": 1,
                    "max_angular_hold_error_rad": 1.518,
                    "passive_motion_joint3_rad": 0.0,
                    "passed": True,
                }
            )
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            result = self.run_generator(package)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((package / "VALIDATION.md").exists())

    def test_generator_fails_when_a_versioned_baseline_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / PACKAGE_DIR.name
            self.copy_minimal_package(package)
            (package / "evidence/baselines/gt_pj_newasset_newton.json").unlink()
            result = self.run_generator(package)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((package / "VALIDATION.md").exists())

    def test_generator_is_clean_checkout_reproducible(self):
        generator_source = (SCRIPT_DIR / "make_validation_md.py").read_text(encoding="utf-8")
        self.assertNotIn("/home/", generator_source)
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / PACKAGE_DIR.name
            self.copy_minimal_package(package)
            result = self.run_generator(package)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                (package / "VALIDATION.md").read_bytes(),
                (PACKAGE_DIR / "VALIDATION.md").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
