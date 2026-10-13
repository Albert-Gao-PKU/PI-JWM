from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pi_jwm.step5_2_training_loop_v1 import Step52Trainer
from pi_jwm.step5_6a_formal_training_config_v1 import formal_training_config_v1
from run_research_geo_temporal_count_v1 import (
    BASELINE_SOURCE_COMMIT, DEFAULT_MANIFEST, ROOT, deployment_git_commit,
    make_config, train,
)


class FakeTrainer:
    """Bookkeeping-only double: does not run an optimizer or touch real data."""
    state_snapshot = Step52Trainer.state_snapshot
    update_validation_state = Step52Trainer.update_validation_state

    def __init__(self, config, initial=None):
        self.config, self.initial = config, initial
        self.saved, self.steps = [], []

    def load_checkpoint(self, path):
        return self.initial

    def train_step(self, global_step, epoch):
        self.steps.append(global_step)
        return {"L_Total": 1.0}

    def save_checkpoint(self, path, state):
        self.saved.append((path.name, dict(state)))


class ResearchTrainingEntryTests(unittest.TestCase):
    def test_deployment_git_commit_returns_real_head(self):
        commit = "1234567890abcdef1234567890abcdef12345678"
        with patch("run_research_geo_temporal_count_v1.subprocess.check_output", return_value=commit + "\n") as probe:
            self.assertEqual(deployment_git_commit(), commit)
        probe.assert_called_once_with(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.PIPE,
        )
        self.assertNotEqual(commit, BASELINE_SOURCE_COMMIT)

    def test_deployment_without_git_repository_records_none(self):
        error = subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])
        with patch("run_research_geo_temporal_count_v1.subprocess.check_output", side_effect=error):
            self.assertIsNone(deployment_git_commit())

    def test_deployment_without_git_executable_records_none(self):
        with patch("run_research_geo_temporal_count_v1.subprocess.check_output", side_effect=FileNotFoundError("git")):
            self.assertIsNone(deployment_git_commit())

    def test_research_formal_config_matches_baseline_except_encoder(self):
        args = SimpleNamespace(variant="full", batch_size=8, steps=5520, mode="train", device="cuda")
        actual = asdict(make_config(args))
        expected = asdict(formal_training_config_v1(DEFAULT_MANIFEST).training)
        actual.pop("encoder")
        expected.pop("encoder")
        self.assertEqual(actual, expected)

    def test_formal_early_stopping_preserves_counter(self):
        args = SimpleNamespace(variant="full", batch_size=8, steps=5520, mode="train", device="cuda", resume=None)
        trainer = FakeTrainer(make_config(args))
        with tempfile.TemporaryDirectory() as folder:
            args.output = Path(folder)
            with patch("run_research_geo_temporal_count_v1.validate_full", return_value={"L_Val": 1.0}), redirect_stdout(io.StringIO()):
                train(trainer, args, {})
            result = json.loads((args.output / "complete.json").read_text())
        self.assertEqual(result["status"], "early_stopped")
        self.assertEqual(result["global_step"], 4416)
        self.assertEqual(sum(name == "best.pt" for name, _ in trainer.saved), 1)
        self.assertEqual(trainer.saved[-1][1]["early_stopping_counter"], 3)

    def test_resume_starts_at_next_completed_step(self):
        args = SimpleNamespace(variant="full", batch_size=8, steps=1106, mode="train", device="cuda", resume=Path("checkpoint.pt"))
        trainer = FakeTrainer(make_config(args), initial={"global_step": 1104, "best_l_val": 1.0, "early_stopping_counter": 1})
        with tempfile.TemporaryDirectory() as folder:
            args.output = Path(folder)
            with patch("run_research_geo_temporal_count_v1.validate_full", return_value={"L_Val": 2.0}), redirect_stdout(io.StringIO()):
                train(trainer, args, {})
        self.assertEqual(trainer.steps, [1104, 1105])
        self.assertEqual(trainer.saved[-1][1]["early_stopping_counter"], 2)


if __name__ == "__main__":
    unittest.main()
