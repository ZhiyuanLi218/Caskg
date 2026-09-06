from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path

_alfworld_data = Path.home() / ".local" / "share" / "caskg" / "alfworld"
if _alfworld_data.is_dir():
    os.environ.setdefault("ALFWORLD_DATA", str(_alfworld_data))

from ablation_experiments.runners import run_alfworld_main_parity as runner


class AlfWorldMainParityTests(unittest.TestCase):
    def test_evaluator_game_order_matches_textworld_seed_1234(self) -> None:
        self.assertEqual(
            runner._evaluator_game_order([str(index) for index in range(6)]),
            ["2", "1", "5", "0", "4", "3"],
        )

    def test_result_is_normalized_to_historical_episode_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "output"
            output_dir.mkdir()
            original = runner.base.eval_single_game

            def fake_eval_single_game(game_index, *_args):
                result_path = output_dir / f"idx_{game_index}.json"
                result_path.write_text(
                    json.dumps(
                        {
                            "name": "task",
                            "task_done": True,
                            "reward": True,
                            "steps": 1,
                            "messages": [],
                            "retrieval_query": "query",
                        }
                    ),
                    encoding="utf-8",
                )
                return {"task_done": True}

            runner.base.eval_single_game = fake_eval_single_game
            try:
                outcome = runner._run_one(
                    game_index=26,
                    episode_id=0,
                    worker_args=argparse.Namespace(),
                    config={},
                    split="eval_in_distribution",
                    output_dir=str(output_dir),
                    log_dir=str(Path(temporary) / "logs"),
                    pass_index=1,
                )
            finally:
                runner.base.eval_single_game = original

            self.assertEqual(outcome["status"], "success")
            self.assertTrue((output_dir / "idx_0.json").is_file())
            self.assertFalse((output_dir / "idx_26.json").exists())


if __name__ == "__main__":
    unittest.main()
