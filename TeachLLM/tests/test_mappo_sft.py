"""Regression checks for structured MAPPO dataset generation."""
import contextlib
import io
import json
from argparse import Namespace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data'))
import build_mappo_sft as builder
from trajectory_io import arrays_to_table


class ConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.constants = builder.CRAFTAX / 'craftax/craftax_together_talens/constants.py'
        cls.classes = builder.read_layout(cls.constants)[1]

    def observation(self, role=0):
        obs = np.zeros(8746, dtype=np.float32)
        grid = obs[:8613].reshape(9, 11, 87)
        grid[:, :, 2] = 1
        grid[:, :, 37] = 1
        grid[:, :, -1] = 1
        dash = obs[8613:8673].reshape(3, 20)
        dash[:, 1] = 1
        for i in range(3): dash[i, 2+i] = 1
        dash[0, 2:5] = 0
        dash[0, 2+role] = 1
        return obs

    def test_directions_map_unknown_and_entities(self):
        for idx, direction in enumerate(builder.OFFSCREEN_DIRECTIONS):
            obs = self.observation()
            obs[8613 + 60 + 8 + idx] = 1
            self.assertEqual(builder.describe_obs(obs, 0, [], self.classes)['teammates'][0]['offscreen_direction'], direction)
        obs = self.observation()
        grid = obs[:8613].reshape(9, 11, 87)
        grid[3, 7, 2] = 0
        grid[3, 7, 5] = 1
        grid[5, 5, 50] = 1
        grid[0, 0] = 0
        grid[1, 1, 37] = 0
        view = builder.describe_obs(obs, 0, [], self.classes)
        self.assertEqual(view['observation']['terrain_rows'][0].split()[0], 'unknown')
        self.assertEqual(len(view['observation']['terrain_rows']), 9)
        self.assertEqual(len(view['observation']['terrain_rows'][0].split()), 11)
        self.assertIn({'type': 'tree', 'relative': [-1, 2]}, view['observation']['nearest_visible_terrain'])
        self.assertIn({'kind': 'passive_mob', 'type': 'cow', 'type_id': 0, 'relative': [1, 0]}, view['observation']['visible_entities'])
        self.assertEqual(view['observation']['item_state_unknown'], [[-3, -4]])

    def test_conservative_action_filter(self):
        obs = self.observation(1)
        self.assertEqual(builder.exclusion_reason(obs, 21), 'craft_insufficient_materials')
        self.assertEqual(builder.exclusion_reason(obs, 8), 'placement_insufficient_materials')
        self.assertIsNone(builder.exclusion_reason(obs, 52))
        obs[8613+84] = np.sqrt(2)/10
        obs[8613+84+4] = np.sqrt(2)/10
        self.assertIsNone(builder.exclusion_reason(obs, 21))
        obs[8613+84+12] = 1
        self.assertEqual(builder.exclusion_reason(obs, 21), 'craft_tool_not_upgrade')
        obs[8613+126] = 1
        self.assertEqual(builder.exclusion_reason(obs, 1), 'sleeping_or_resting')
        obs[8613+1] = 0
        self.assertEqual(builder.exclusion_reason(obs, 1), 'dead')

    def prepare(self, base, output=True):
        obs = np.stack([[self.observation(a) for a in range(3)] for _ in range(3)])
        obs[1, 0, 8613+126] = 1
        data = {'trajectory_id': np.array([10]*3, dtype=np.int32),
                'step_id': np.arange(3, dtype=np.int32),
                'task_idx': np.array([0]*3, dtype=np.int32),
                'instruction_idx': np.array([0]*3, dtype=np.int32),
                'instruction_done': np.array([0, 0, 1], dtype=np.float32),
                'done_all': np.array([False, False, True]),
                'actions': np.array([[1, 21, 52], [3, 1, 3], [1, 1, 1]], dtype=np.int32), 'obs': obs}
        parts = base / 'parts'
        parts.mkdir()
        pq.write_table(arrays_to_table(data), parts / 'part_000001.parquet')
        return Namespace(mode='both', parts=parts, scenario_file=builder.DEFAULT_SCENARIOS,
                         constants_file=self.constants, output=base / 'output' if output else None,
                         hf_repo_id=None if output else 'user/dataset', hf_use_empty_repo=False,
                         hf_public=False, hf_shard_mib=1, source_hf_repo_id=None, source_hf_revision=None,
                         max_shards=0, skip_unreadable=False, seed=42, val_fraction=.05, test_fraction=.05, history=5)

    def test_end_to_end_roles_history_and_team_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.prepare(Path(tmp))
            with contextlib.redirect_stdout(io.StringIO()): builder.build(args)
            report = json.loads((args.output / 'report.json').read_text())
            records = {mode: [(split, json.loads(line)) for split in ('train', 'validation', 'test')
                              for line in (args.output / mode / f'{split}.jsonl').read_text().splitlines()]
                       for mode in ('individual', 'team')}
            self.assertEqual(len(records['individual']), 7)
            self.assertEqual(len(records['team']), 1)
            self.assertEqual(report['team_steps_skipped'], 2)
            self.assertEqual({s for rows in records.values() for s, _ in rows}, {builder.split_for(10, 42, .05, .05)})
            agent = next(row for _, row in records['individual'] if row['metadata']['step_id'] == 1 and row['metadata']['agent_id'] == 2)
            self.assertEqual(json.loads(agent['messages'][1]['content'])['selected_action_history'], ['GIVE_TO_TEAMMATE_1'])
            self.assertEqual(agent['messages'][2]['content'], '<action>UP</action>')
            team = records['team'][0][1]
            self.assertEqual(set(json.loads(team['messages'][1]['content'])['agents']), {'agent_0', 'agent_1', 'agent_2'})
            self.assertEqual(team['messages'][2]['content'].count('<action>'), 3)
            self.assertTrue(team['messages'][2]['content'].startswith('<actions>'))

    def test_hub_writes_configured_modes_without_overwriting(self):
        class FakeHub:
            instance = None
            def __init__(self):
                self.files = {}
                FakeHub.instance = self
            def create_repo(self, **kwargs):
                self.create = kwargs
            def upload_file(self, path_or_fileobj, path_in_repo, **kwargs):
                self.files[path_in_repo] = path_or_fileobj.read() if hasattr(path_or_fileobj, 'read') else path_or_fileobj
            def list_repo_files(self, **kwargs):
                return ['.gitattributes', 'data/existing.jsonl']
        with tempfile.TemporaryDirectory() as tmp:
            args = self.prepare(Path(tmp), output=False)
            with patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(HfApi=FakeHub)}), contextlib.redirect_stdout(io.StringIO()):
                builder.build(args)
                hub = FakeHub.instance
                self.assertFalse(hub.create['exist_ok'])
                self.assertTrue(any(p.startswith('data/individual/') for p in hub.files))
                self.assertTrue(any(p.startswith('data/team/') for p in hub.files))
                self.assertIn(b'config_name: team', hub.files['README.md'])
                args.hf_use_empty_repo = True
                with self.assertRaisesRegex(ValueError, 'not empty'):
                    builder.build(args)
                self.assertEqual(FakeHub.instance.files, {})


if __name__ == '__main__':
    unittest.main()
