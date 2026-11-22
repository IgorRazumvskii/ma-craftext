"""Regression: capture the post-action state, even when step auto-resets."""
import ast
from pathlib import Path
import sys
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines"))
from trajectory_parquet import ShardWriter, iter_parquet

source = ast.parse((ROOT / "baselines/collect_expert_trajectories_mappo.py").read_text())
names = {"batchify_obs", "unbatchify_actions", "build_rollout_fn", "terminal_snapshot_data"}
nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
ns = {"jax": jax, "jnp": jnp, "np": np,
      "build_actor_action_mask": lambda env, n: jnp.ones((n * env.num_agents, 1), bool)}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "collector", "exec"), ns)


class Env:
    agents = ["a", "b"]
    num_agents = 2

    def step(self, key, state, action):
        del key, action
        final = state + 1
        done = final >= 2
        # Model an automatic reset to a recognizably different state.
        returned = jnp.where(done, 0, final)
        obs = {a: jnp.array([returned], dtype=jnp.float32) for a in self.agents}
        post = {a: jnp.array([final], dtype=jnp.float32) for a in self.agents}
        info = {"terminal_state": {"map": jnp.array([final, 9])}, "terminal_obs": post,
                "user_info": {"Instructions/instruction_done": jnp.full((2,), done),
                              "Instructions/instruction_idx": jnp.zeros(2),
                              "Instructions/task_idx": jnp.zeros(2)}}
        return obs, returned, {a: jnp.float32(0) for a in self.agents}, {**{a: done for a in self.agents}, "__all__": done}, info


class Actor:
    def apply(self, params, hidden, inputs):
        del params, inputs
        class Policy:
            probs = jnp.ones((1, 2, 1))
        return hidden, Policy()


class TerminalCollectionTest(unittest.TestCase):
    def test_snapshot_is_pre_reset_and_survives_further_steps_and_parquet(self):
        env = Env()
        rollout = ns["build_rollout_fn"](env, Actor(), {}, 1, 4, True, False, 1)
        obs = {a: jnp.zeros((1, 1)) for a in env.agents}
        carry = (jnp.zeros((1,), jnp.int32), obs, jnp.zeros(2, bool), jnp.zeros((2, 1)),
                 jax.random.PRNGKey(0), jnp.ones(1, bool), {"map": jnp.zeros((1, 2), jnp.int32)}, obs)
        carry, transitions = rollout(carry)
        self.assertEqual(np.asarray(transitions["valid"]).ravel().tolist(), [True, True, False, False])
        self.assertEqual(int(carry[0][0]), 0)  # reset has already happened
        data = ns["terminal_snapshot_data"](jax.device_get(carry[-2]), jax.device_get(carry[-1]), env.agents, 0, 17)
        np.testing.assert_array_equal(data["state/map"], [[2, 9]])
        np.testing.assert_array_equal(data["terminal_obs"], [[[2], [2]]])
        with tempfile.TemporaryDirectory() as directory:
            writer = ShardWriter(directory, 1000)
            writer.write(data)
            writer.flush()
            restored = next(iter_parquet(writer.parts[0]["path"]))
        np.testing.assert_array_equal(restored["state/map"], [[2, 9]])
        np.testing.assert_array_equal(restored["trajectory_id"], [17])


if __name__ == "__main__":
    unittest.main()
