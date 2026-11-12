"""Dependency-light regression tests for MAPPO batching.

These execute the actual training function's batching statements with NumPy,
without importing the environment, model downloads, or external logging.
They check shape/index contracts, not JAX compilation or learning quality.

Run: python -m unittest discover -s tests -p 'test_mappo_batching.py' -v
Set MAPPO_SOURCE to test the same cases against another my_mappo.py.
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


SOURCE = Path(os.environ.get(
    "MAPPO_SOURCE", str(Path(__file__).resolve().parents[1] / "baselines/my_mappo.py")
))
TREE = ast.parse(SOURCE.read_text())


def function_node(name):
    return next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name == name)


def tree_map(fn, value):
    if isinstance(value, dict):
        return {key: tree_map(fn, item) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(tree_map(fn, item) for item in value))
    if isinstance(value, tuple):
        return tuple(tree_map(fn, item) for item in value)
    if isinstance(value, list):
        return [tree_map(fn, item) for item in value]
    return fn(value)


def load_function(name, namespace):
    module = ast.Module(body=[function_node(name)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace[name]


class BatchingTests(unittest.TestCase):
    def test_one_environment_keeps_vmap_axis(self):
        namespace = {"jnp": np, "action": np.array([[2, 3, 4]]),
                     "env": SimpleNamespace(agents=["a", "b", "c"], num_agents=3),
                     "config": {"NUM_ENVS": 1}}
        load_function("unbatchify_actions", namespace)
        statements = [node for node in function_node("_env_step").body
                      if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "env_act" for t in node.targets)]
        exec(compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])),
                     str(SOURCE), "exec"), namespace)
        for action in namespace["env_act"].values():
            self.assertEqual(action.shape, (1,))

    def run_epoch_batching(self, num_envs, num_steps, num_minibatches):
        from typing import NamedTuple
        class Transition(NamedTuple):
            obs: np.ndarray
            info: dict
        num_actors = 3 * num_envs
        config = {"NUM_ACTORS": num_actors, "NUM_MINIBATCHES": num_minibatches}
        obs = np.arange(num_steps * num_actors * 2).reshape(num_steps, num_actors, 2)
        env_info = {"steps": np.zeros((num_steps, num_envs)),
                    "user_info": {"success": np.zeros((num_steps, num_envs, 3))}}
        traj = Transition(obs=obs, info=env_info)
        initial = (np.zeros((num_actors, 4)), np.ones((num_actors, 4)))
        advantages = np.arange(num_steps * num_actors).reshape(num_steps, num_actors)
        captured = {}
        def scan(_fn, state, minibatches):
            captured["minibatches"] = minibatches
            return state, {}
        namespace = {"np": np, "jnp": np, "config": config,
                     "jax": SimpleNamespace(
                         tree=SimpleNamespace(map=tree_map),
                         random=SimpleNamespace(split=lambda _: (1, 2),
                                                permutation=lambda _, n: np.arange(n)[::-1]),
                         lax=SimpleNamespace(scan=scan))}
        update_epoch = load_function("_update_epoch", namespace)
        output, _ = update_epoch((None, initial, traj, advantages, advantages + 1, 0), None)
        batch = captured["minibatches"]
        minibatch_actors = num_actors // num_minibatches
        self.assertEqual(batch[0].shape, (num_minibatches, 1, minibatch_actors, 4))
        self.assertEqual(batch[2].obs.shape, (num_minibatches, num_steps, minibatch_actors, 2))
        self.assertEqual(batch[2].info, {})
        self.assertEqual(batch[3].shape, (num_minibatches, num_steps, minibatch_actors))
        self.assertEqual(output[1][0].shape, initial[0].shape)
        self.assertIs(output[2].info, env_info)
        expected = obs[:, ::-1].reshape(num_steps, num_minibatches, minibatch_actors, 2).swapaxes(0, 1)
        np.testing.assert_array_equal(batch[2].obs, expected)

    def test_environment_info_is_not_indexed_by_actor_permutation(self):
        self.run_epoch_batching(num_envs=2, num_steps=4, num_minibatches=3)

    def test_single_step_and_single_actor_per_minibatch(self):
        self.run_epoch_batching(num_envs=1, num_steps=1, num_minibatches=3)

    def test_single_actor_carry_keeps_batch_axis_in_loss(self):
        # Execute the real model.apply argument expression from each loss.
        for name in ("_actor_loss_fn", "_critic_loss_fn"):
            call = next(node for node in ast.walk(function_node(name))
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "apply")
            carry_arg = call.args[1]
            carry = eval(compile(ast.Expression(carry_arg), str(SOURCE), "eval"),
                         {"init_hstate": np.zeros((1, 1, 4))})
            self.assertEqual(carry.shape, (1, 4))

    def test_invalid_recurrent_batch_sizes_fail_early(self):
        namespace = {"np": np, "WorldStateWrapper": lambda env: env,
                     "LogWrapper": lambda env: env,
                     "build_actor_action_mask": lambda env, num_envs: None}
        make_train = load_function("make_train", namespace)
        base = {"NUM_ENVS": 2, "NUM_STEPS": 4, "NUM_MINIBATCHES": 3,
                "UPDATE_EPOCHS": 2, "TOTAL_TIMESTEPS": 16,
                "CLIP_EPS": 0.2, "SCALE_CLIP_EPS": False}
        for changes in ({"NUM_MINIBATCHES": 4}, {"NUM_ENVS": 0},
                        {"NUM_STEPS": -1}, {"UPDATE_EPOCHS": 0},
                        {"TOTAL_TIMESTEPS": 2}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                make_train(dict(base, **changes), SimpleNamespace(num_agents=3))


if __name__ == "__main__":
    unittest.main()
