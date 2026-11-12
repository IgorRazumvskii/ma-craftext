"""Regression checks for dataset selection, runnable without JAX or a text model.

The real loader function and real dataset constructors are executed from their
ASTs. Only the unavailable JAX/Flax data containers and package imports are
replaced. CRAFTEXT_SOURCE_ROOT can point to source fixtures outside this tree.
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get("CRAFTEXT_SOURCE_ROOT", PACKAGE_ROOT))
SCENARIOS_ROOT = SOURCE_ROOT / "craftext/dataset/scenarious"
LOADER_PATH = Path(
    os.environ.get(
        "CRAFTEXT_LOADER_PATH",
        PACKAGE_ROOT / "craftext/environment/scenarious/loader.py",
    )
)


def execute_nodes(path, select, namespace):
    parsed = ast.parse(path.read_text())
    nodes = [node for node in parsed.body if select(node)]
    for node in nodes:
        if isinstance(node, ast.ClassDef):
            node.decorator_list = []
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec")
    exec(code, namespace)


def real_easy_scenarios(dataset):
    namespace = {
        "Achievements": lambda **kwargs: SimpleNamespace(**kwargs),
        "TargetState": lambda **kwargs: SimpleNamespace(**kwargs),
    }
    execute_nodes(
        SOURCE_ROOT / "craftext/environment/craftext_constants.py",
        lambda node: isinstance(node, ast.ClassDef)
        and node.name in {"Achievement", "AchievementState", "Scenarios"},
        namespace,
    )
    execute_nodes(
        SCENARIOS_ROOT / dataset / "instructions.py",
        lambda node: (
            isinstance(node, ast.FunctionDef) and node.name == "create_target_state"
        )
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "easy" for target in node.targets)
        ),
        namespace,
    )
    return namespace["easy"]


class ScenarioLoaderTests(unittest.TestCase):
    def setUp(self):
        self.modules = {
            f"craftext.dataset.scenarious.{name}.instructions": SimpleNamespace(
                easy=real_easy_scenarios(name), empty={}
            )
            for name in (
                "jax_conditional_achievements",
                "jax_conditional_achievements_reward_progression",
            )
        }
        self.import_module = Mock(side_effect=self.modules.__getitem__)
        namespace = {
            "os": os,
            "importlib": SimpleNamespace(import_module=self.import_module),
            "get_default_scenario_path": lambda: str(SCENARIOS_ROOT),
        }
        execute_nodes(
            LOADER_PATH,
            lambda node: isinstance(node, ast.FunctionDef) and node.name == "load_scenarios",
            namespace,
        )
        self.load_scenarios = namespace["load_scenarios"]

    def config(self, name):
        data = yaml.safe_load(
            (SOURCE_ROOT / "craftext/dataset/configs" / f"{name}.yaml").read_text()
        )
        return SimpleNamespace(**data)

    def test_easy_train_keeps_unrestricted_wood_goal_in_both_directory_orders(self):
        directories = [
            "jax_conditional_achievements",
            "jax_conditional_achievements_reward_progression",
        ]
        for order in (directories, directories[::-1]):
            with self.subTest(directory_order=order), patch.object(os, "listdir", return_value=order):
                self.import_module.reset_mock()
                scenarios = self.load_scenarios(self.config("easy_train"))
                self.assertEqual(len(scenarios), 57)
                mask = scenarios["COLLECT_WOOD"]["arguments"].achievements.achievement_mask
                self.assertEqual(mask, (1,) + (0,) * 21)
                self.import_module.assert_called_once_with(
                    "craftext.dataset.scenarious.jax_conditional_achievements.instructions"
                )

    def test_explicit_strict_train_retains_strict_constraints(self):
        scenarios = self.load_scenarios(self.config("easy_strict_train"))
        mask = scenarios["COLLECT_WOOD"]["arguments"].achievements.achievement_mask
        self.assertEqual(mask, (1,) + (-1,) * 21)

    def test_unknown_dataset_fails_before_import(self):
        config = self.config("easy_train")
        config.dataset_key = "jax_conditional"
        with self.assertRaisesRegex(ValueError, "dataset does not exist"):
            self.load_scenarios(config)
        self.import_module.assert_not_called()

    def test_missing_and_empty_subsets_fail_explicitly(self):
        for subset in ("missing", "empty"):
            with self.subTest(subset=subset):
                config = self.config("easy_train")
                config.subset_key = subset
                with self.assertRaises(ValueError):
                    self.load_scenarios(config)


if __name__ == "__main__":
    unittest.main()
