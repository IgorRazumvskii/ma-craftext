"""Focused CPU regressions using real JAX/Flax and AST-selected source code.

Only imports of rendering assets, CUDA text models and external logging are
avoided. These are unit tests, not a full Craftax rollout/training benchmark.
MA_SOURCE_ROOT can select an unmodified checkout for before/after comparison.
"""
import ast
from enum import Enum
import functools
import os
from pathlib import Path
import pickle
import re
import tempfile
from types import SimpleNamespace
from typing import Dict, Sequence, NamedTuple
import unittest

import distrax
import flax.linen as nn
from flax import struct
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

ROOT = Path(os.environ.get("MA_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
FIXTURES = Path(os.environ.get("MA_FIXTURE_ROOT", ROOT))
ENV = Path("craftax/craftax_together_talens")


def tree(rel):
    path = ROOT / rel
    if not path.exists():
        path = FIXTURES / rel
    return ast.parse(path.read_text())


def execute(nodes, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "<actual-source>", "exec"), namespace)


def load_defs(rel, names, namespace):
    execute([n for n in tree(rel).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names], namespace)


def statement_name(node):
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.FunctionDef):
        return node.name
    return None


def section(function, start, stop):
    nodes = next(n for n in tree(ENV / 'game_logic/actions.py').body
                 if isinstance(n, ast.FunctionDef) and n.name == function).body
    i = next(i for i, n in enumerate(nodes) if statement_name(n) == start)
    j = next(i for i, n in enumerate(nodes) if i > next(k for k, x in enumerate(nodes)
             if statement_name(x) == start) and statement_name(n) == stop)
    return nodes[i:j]


def env_namespace():
    ns = dict(jax=jax, jnp=jnp, np=np, Enum=Enum, struct=struct)
    load_defs(ENV / 'constants.py', {'Action', 'Achievement', 'BlockType', 'ItemType'}, ns)
    execute([n for n in tree(ENV / 'game_logic/constants.py').body
             if not isinstance(n, (ast.Import, ast.ImportFrom))], ns)
    load_defs(ENV / 'craftax_state.py', {'Inventory'}, ns)
    load_defs(ENV / 'game_logic/trade.py', {'trade_materials', 'make_request'}, ns)
    # make_request's one local import is replaced with the actual source constant.
    make_req = next(n for n in tree(ENV / 'game_logic/trade.py').body
                    if isinstance(n, ast.FunctionDef) and n.name == 'make_request')
    make_req.body = [n for n in make_req.body if not isinstance(n, ast.ImportFrom)]
    execute([n for n in tree(ENV / 'constants.py').body
             if statement_name(n) == 'REQUEST_MAX_DURATION'], ns)
    execute([make_req], ns)
    load_defs(ENV / 'game_logic/actions.py', {'add_new_growing_plant', 'update_plants_with_eat'}, ns)
    load_defs(ENV / 'game_logic/core.py', {'update_plants'}, ns)
    return ns


def state_fixture(ns):
    inv = {name: jnp.zeros((3,), dtype=jnp.int32)
           for name in ns['Inventory'].__dataclass_fields__}
    inv['armour'] = jnp.zeros((3, 4), dtype=jnp.int32)
    inv['potions'] = jnp.zeros((3, 6), dtype=jnp.int32)
    fields = dict(inventory=ns['Inventory'](**inv), achievements=jnp.zeros((3, 100), bool),
                  player_specialization=jnp.array([1, 2, 3]), player_alive=jnp.ones(3, bool),
                  request_type=jnp.zeros(3, jnp.int32), request_duration=jnp.zeros(3, jnp.int32),
                  player_food=jnp.ones(3), player_drink=jnp.ones(3),
                  player_hunger=jnp.zeros(3), player_thirst=jnp.zeros(3),
                  trade_count=0, food_trade_count=0, drink_trade_count=0,
                  map=jnp.full((1, 8, 8), ns['BlockType'].GRASS.value, jnp.int32),
                  player_level=0, growing_plants_positions=jnp.zeros((3, 2), jnp.int32),
                  growing_plants_age=jnp.zeros(3, jnp.int32), growing_plants_mask=jnp.zeros(3, bool))
    State = struct.dataclass(type('UnitState', (), {'__annotations__': {name: object for name in fields}}))
    return State(**fields)


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.ns = env_namespace()
        self.state = state_fixture(self.ns)

    def test_request_mask_and_observation_index(self):
        ns = self.ns
        self.assertEqual(ns['get_action_dim'](3), 61)
        self.assertEqual(ns['REQUEST_OBSERVATION_DIM'], 16)
        masks = ns['get_team_action_masks'](jnp.array([1, 2, 3]), 61)
        np.testing.assert_array_equal(masks[:, 60], [False, True, False])
        self.assertEqual(int(ns['get_request_observation_index'](60)), 15)
        np.testing.assert_array_equal(ns['get_request_observation_index'](jnp.arange(54, 60)), np.arange(9, 15))

    def test_sapling_gather_request_transfer_plant_grow_eat(self):
        ns, state = self.ns, self.state
        static = SimpleNamespace(player_count=3, max_growing_plants=1)
        positions = jnp.array([[2, 2], [3, 3], [4, 4]])
        # Execute the actual sapling-gather block until the miner finds a sapling.
        gather = section('do_action', 'is_block_grass', 'is_block_water')
        for seed in range(100):
            local = dict(ns, state=state, static_params=static, block_position=positions,
                         is_forager=jnp.array([True, False, False]), doing_action=jnp.ones(3, bool),
                         new_inventory=state.inventory, _rng=jax.random.PRNGKey(seed))
            execute(gather, local)
            if int(local['new_inventory'].sapling[0]) > 0:
                break
        state = state.replace(inventory=local['new_inventory'])
        self.assertGreater(int(state.inventory.sapling[0]), 0)
        action = jnp.array([ns['Action'].GIVE.value, ns['REQUEST_SAPLING'], 0])
        requested = jax.jit(ns['make_request'])(state, action)
        self.assertEqual(int(requested.request_type[1]), 60)
        self.assertGreater(int(requested.request_duration[1]), 0)
        traded, count = jax.jit(lambda s, a: ns['trade_materials'](s, a, static))(requested, action)
        np.testing.assert_array_equal(traded.inventory.sapling, [1, 1, 0])
        np.testing.assert_array_equal(count, [1, 0, 0])
        self.assertEqual(int(traded.inventory.sapling.sum()), int(state.inventory.sapling.sum()))
        # Run the real plant placement block, including inventory and achievement updates.
        local = dict(ns, state=traded, static_params=static,
                     action=jnp.array([0, ns['Action'].PLACE_PLANT.value, 0]),
                     placing_block_position=positions, new_map=traded.map[0],
                     new_item_map=jnp.full((8, 8), ns['ItemType'].NONE.value, jnp.int32),
                     is_placement_in_bounds_not_in_mobs=jnp.ones(3, bool),
                     new_inventory=traded.inventory, new_achievements=traded.achievements)
        execute(section('place_block', '_player_place_plant', 'new_whole_map'), local)
        planted = traded.replace(map=traded.map.at[0].set(local['new_map']),
            inventory=local['new_inventory'], achievements=local['new_achievements'],
            growing_plants_positions=local['new_growing_plants_positions'],
            growing_plants_age=local['new_growing_plants_age'],
            growing_plants_mask=local['new_growing_plants_mask'])
        self.assertEqual(int(planted.inventory.sapling[1]), 0)
        self.assertTrue(bool(planted.achievements[1, ns['Achievement'].PLACE_PLANT.value]))
        mature = jax.jit(lambda s: jax.lax.fori_loop(0, 500, lambda _, x: ns['update_plants'](x, static), s))(planted)
        self.assertEqual(int(mature.map[0, 3, 3]), ns['BlockType'].RIPE_PLANT.value)
        local = dict(ns, state=mature, block_position=positions, doing_action=jnp.array([False, True, False]),
                     equal_block_placement=(positions[:, None] == positions[None]).all(axis=-1),
                     new_map=mature.map[0], new_achievements=mature.achievements,
                     get_max_food=lambda s: jnp.full(3, 9))
        execute(section('do_action', 'is_block_plant', 'can_mine_stalagmite'), local)
        self.assertTrue(bool(local['new_achievements'][1, ns['Achievement'].EAT_PLANT.value]))
        self.assertGreater(float(local['new_food'][1]), float(mature.player_food[1]))

    def test_sapling_transfer_requires_miner_stock_and_live_request(self):
        ns, state = self.ns, self.state
        static = SimpleNamespace(player_count=3)
        stock = jnp.array([1, 0, 1])
        state = state.replace(inventory=state.inventory.replace(sapling=stock),
                              request_type=jnp.array([0, 60, 0]), request_duration=jnp.array([0, 10, 0]))
        trade = jax.jit(lambda s, a: ns['trade_materials'](s, a, static))
        # Warrior has a sapling but is not a permitted donor.
        out, count = trade(state, jnp.array([0, 0, 53]))
        np.testing.assert_array_equal(out.inventory.sapling, stock)
        self.assertEqual(int(count.sum()), 0)
        out, count = trade(state.replace(request_duration=jnp.zeros(3, jnp.int32)), jnp.array([52, 0, 0]))
        np.testing.assert_array_equal(out.inventory.sapling, stock)
        self.assertEqual(int(count.sum()), 0)

    def test_furnace_shared_target_keeps_map_inventory_achievement_consistent(self):
        ns = self.ns
        for other in (0, 2):
            positions = jnp.array([[1, 1], [3, 3], [5, 5]]).at[other].set(jnp.array([3, 3]))
            inventory = self.state.inventory.replace(stone=jnp.array([0, 1, 0]))
            local = dict(ns, state=self.state, action=jnp.array([0, ns['Action'].PLACE_FURNACE.value, 0]),
                is_valid_placement=jnp.array([False, True, False]), placing_block_position=positions,
                equal_block_placement=(positions[:, None] == positions[None]).all(axis=-1),
                new_inventory=inventory, new_achievements=self.state.achievements, new_map=self.state.map[0])
            execute(section('place_block', 'furnace_key_down', 'stone_key_down'), local)
            self.assertEqual(int(local['new_map'][3, 3]), ns['BlockType'].FURNACE.value)
            self.assertEqual(int(local['new_inventory'].stone[1]), 0)
            self.assertTrue(bool(local['new_achievements'][1, ns['Achievement'].PLACE_FURNACE.value]))


class ModelAndEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.ns = dict(jax=jax, jnp=jnp, np=np, nn=nn, functools=functools,
                       constant=constant, orthogonal=orthogonal, distrax=distrax,
                       Sequence=Sequence, Dict=Dict, _ILLEGAL_ACTION_LOGIT=jnp.float32(-1e8))
        load_defs(Path('baselines/my_mappo.py'), {'ScannedRNN', 'ActorRNN', 'CriticRNN'}, self.ns)

    def test_recurrent_replay_matches_rollout_and_mask_has_finite_gradients(self):
        ns = self.ns
        model = ns['ActorRNN'](5, config={'FC_DIM_SIZE': 128, 'GRU_HIDDEN_DIM': 512})
        h0 = ns['ScannedRNN'].initialize_carry(1, 512)
        obs = jax.random.normal(jax.random.PRNGKey(1), (3, 1, 7))
        done = jnp.array([[False], [True], [False]])
        mask = jnp.broadcast_to(jnp.array([True, False, True, False, False]), (3, 1, 5))
        params = model.init(jax.random.PRNGKey(2), h0, (obs, done, mask))
        carry, sequence = jax.jit(model.apply)(params, h0, (obs, done, mask))
        h, probs = h0, []
        for i in range(3):
            h, pi = model.apply(params, h, (obs[i:i+1], done[i:i+1], mask[i:i+1]))
            probs.append(pi.probs)
        np.testing.assert_allclose(sequence.probs, jnp.concatenate(probs), atol=1e-6)
        np.testing.assert_allclose(carry, h, atol=1e-6)
        np.testing.assert_array_equal(sequence.probs[..., [1, 3, 4]], 0)
        self.assertTrue(np.isfinite(sequence.entropy()).all())
        grads = jax.grad(lambda p: -model.apply(p, h0, (obs, done, mask))[1].log_prob(jnp.zeros((3, 1), jnp.int32)).mean())(params)
        self.assertTrue(all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(grads)))
        critic = ns['CriticRNN'](config=model.config)
        cp = critic.init(jax.random.PRNGKey(3), h0, (obs, done))
        self.assertEqual(critic.apply(cp, h0, (obs, done))[1].shape, (3, 1))

    def test_both_checkpoint_formats_restore(self):
        ns = dict(self.ns, Path=Path, pickle=pickle)
        load_defs(Path('baselines/eval_sr.py'), {'load_checkpoint', 'restore_model_params'}, ns)
        for key in ('train_state', 'actor_train_state'):
            with tempfile.NamedTemporaryFile(suffix='.pkl') as f:
                pickle.dump({key: {'params': {'w': np.array([1., 2.])}}}, f)
                f.flush()
                checkpoint = ns['load_checkpoint'](f.name)
                np.testing.assert_array_equal(ns['restore_model_params'](checkpoint)['w'], [1., 2.])

    def test_evaluation_supplies_mask_to_real_mappo_actor(self):
        ns = self.ns
        load_defs(Path('baselines/eval_sr.py'), {'run_episode'}, ns)
        class Env:
            num_agents = 1
            agents = ['agent_0']
            def reset(self, rng, instruction_idx=-1):
                return {'agent_0': jnp.zeros(4)}, None
            def step(self, rng, state, action):
                assert int(action['agent_0']) == 2
                return {'agent_0': jnp.zeros(4)}, None, {}, {'__all__': True}, {
                    'user_info': {'Instructions/instruction_done': jnp.array([1.]),
                                  'Instructions/task_idx': jnp.array([0]),
                                  'Instructions/instruction_idx': jnp.array([0])}}
        net = ns['ActorRNN'](3, config={'FC_DIM_SIZE': 8, 'GRU_HIDDEN_DIM': 8})
        mask = jnp.array([[False, False, True]])
        params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 8)),
                          (jnp.zeros((1, 1, 4)), jnp.zeros((1, 1), bool), mask[None]))
        for greedy in (False, True):
            result = ns['run_episode'](Env(), net, params, jax.random.PRNGKey(1), 2, -1, greedy, mask)
            self.assertEqual(result['solved'], 1)
            self.assertEqual(result['steps'], 1)

    def test_full_ppo_update_on_synthetic_env_with_singleton_minibatches(self):
        # Actual make_train, actor/critic, GAE, PPO gradients, optimizer and JIT.
        # Synthetic observations replace only the expensive game/text stack.
        ns = self.ns
        env_ns = env_namespace()
        class Wrapper:
            def __init__(self, env):
                self._env = env
            def __getattr__(self, name):
                return getattr(self._env, name)
        class Env:
            num_agents = 3
            agents = ['agent_0', 'agent_1', 'agent_2']
            def action_space(self, agent):
                return SimpleNamespace(n=61)
            def observation_space(self, agent):
                return SimpleNamespace(shape=(4,))
            def reset(self, rng):
                return {a: jnp.zeros(4) for a in self.agents}, jnp.int32(0)
            def step(self, rng, state, actions):
                terminal = state >= 1
                state = jnp.where(terminal, 0, state + 1)
                obs = {a: jnp.full(4, state, jnp.float32) for a in self.agents}
                done = {a: terminal for a in self.agents}
                done['__all__'] = terminal
                info = {'returned_episode': jnp.repeat(terminal, 3),
                        'returned_episode_lengths': jnp.full(3, 2),
                        'returned_episode_returns': jnp.ones(3), 'steps': state,
                        'user_info': {'Instructions/instruction_done': jnp.repeat(terminal, 3),
                                      'Instructions/task_idx': jnp.zeros(3)}}
                return obs, state, {a: jnp.float32(terminal) for a in self.agents}, done, info
        class Experiment:
            def log_metric(self, name, value, step):
                assert np.isfinite(value).all(), (name, value)
        ns.update(JaxMARLWrapper=Wrapper, LogWrapper=lambda env: env,
                  partial=functools.partial, NamedTuple=NamedTuple,
                  Path=Path, os=os, pickle=pickle, re=re, TrainState=TrainState, optax=optax,
                  Specialization=env_ns['Specialization'], get_team_action_masks=env_ns['get_team_action_masks'])
        load_defs(Path('baselines/my_mappo.py'),
                  {'Transition', 'WorldStateWrapper', 'build_actor_action_mask',
                   'batchify', 'unbatchify_actions', '_dump_checkpoint_atomic', 'make_train'}, ns)
        with tempfile.TemporaryDirectory() as temp:
            cfg = dict(NUM_ENVS=1, NUM_STEPS=1, NUM_MINIBATCHES=3, UPDATE_EPOCHS=2,
                       TOTAL_TIMESTEPS=2, CLIP_EPS=.2, SCALE_CLIP_EPS=False,
                       FC_DIM_SIZE=8, GRU_HIDDEN_DIM=16, LR=.0002, ANNEAL_LR=True,
                       MAX_GRAD_NORM=1., GAMMA=.99, GAE_LAMBDA=.8, ENT_COEF=.01, VF_COEF=.5,
                       experiment=Experiment(), CHECKPOINT_DIR=temp, CHECKPOINT_INTERVAL=100,
                       HF_HUB_UPLOAD=False, TASK_NAMES=['synthetic'])
            train = jax.jit(jax.vmap(ns['make_train'](cfg, Env())))
            result = jax.block_until_ready(train(jax.random.split(jax.random.PRNGKey(0), 1)))
            jax.effects_barrier()
            train_states = result['runner_state'][0][0]
            for train_state in train_states:
                self.assertTrue(all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(train_state.params)))
                self.assertEqual(int(train_state.step[0]), 12)


if __name__ == '__main__':
    unittest.main()
