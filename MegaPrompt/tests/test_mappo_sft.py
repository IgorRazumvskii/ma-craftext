"""Regression tests for MAPPO observation decoding and SFT filtering."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from argparse import Namespace

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_mappo_sft as builder
from trajectory_parquet import arrays_to_table


class DatasetConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classes = builder.read_layout()[1]

    def observation(self, role=0):
        obs = np.zeros(8746, dtype=np.float32)
        grid = obs[:8613].reshape(9, 11, 87)
        grid[:, :, 2] = 1  # GRASS
        grid[:, :, -1] = 1  # illuminated
        dashboard = obs[8613:8673].reshape(3, 20)
        dashboard[:, 1] = 1  # all alive
        for i in range(3): dashboard[i, 2+i] = 1
        dashboard[0, 2:5] = 0
        dashboard[0, 2+role] = 1
        return obs

    def test_all_offscreen_directions_match_renderer_encoding(self):
        for idx, expected in enumerate(('west', 'east', 'north', 'northwest',
                                        'northeast', 'south', 'southwest', 'southeast')):
            with self.subTest(idx=idx):
                obs = self.observation()
                obs[8613 + 60 + 8 + idx] = 1  # first teammate
                text, _, _ = builder.describe_obs(obs, 0, self.classes)
                self.assertIn(f'Teammate 1 off-screen direction: {expected}', text)

    def test_map_and_entity_have_correct_relative_location(self):
        obs = self.observation()
        grid = obs[:8613].reshape(9, 11, 87)
        grid[3, 7, 2] = 0
        grid[3, 7, 5] = 1  # TREE
        grid[5, 5, 37 + 5 + 8] = 1  # passive mob type 0 = cow
        grid[0, 0, -1] = 0
        grid[0, 0, :86] = 0
        text, role, alive = builder.describe_obs(obs, 0, self.classes)
        self.assertIn('?? GR GR', text)
        self.assertIn('TR=tree', text)
        self.assertIn('tree: 1 north, 2 east', text)
        self.assertIn('1 south: cow (passive mob, type 0)', text)
        self.assertEqual(role, 'MINER')
        self.assertTrue(alive)

    def test_conservative_filter(self):
        obs = self.observation(1)
        self.assertEqual(builder.exclusion_reason(obs, 21), 'craft_insufficient_materials')
        self.assertEqual(builder.exclusion_reason(obs, 8), 'placement_insufficient_materials')
        self.assertIsNone(builder.exclusion_reason(obs, 52))  # simultaneous team request is possible
        obs[8613+84] = np.sqrt(2)/10  # two wood
        obs[8613+84+4] = np.sqrt(2)/10  # two diamonds
        self.assertIsNone(builder.exclusion_reason(obs, 21))
        self.assertIsNone(builder.exclusion_reason(obs, 8))
        obs[8613+84+12] = 1.0  # highest-tier sword
        self.assertEqual(builder.exclusion_reason(obs, 21), 'craft_tool_not_upgrade')
        obs[8613+126] = 1
        self.assertEqual(builder.exclusion_reason(obs, 1), 'sleeping_or_resting')
        obs[8613+1] = 0
        self.assertEqual(builder.exclusion_reason(obs, 1), 'dead')

    def test_end_to_end_episode_splits_and_filters(self):
        obs = np.stack([[self.observation(a) for a in range(3)] for _ in range(2)])
        obs[1, 0, 8613+126] = 1  # sleeping agent's UP is ignored
        obs[1, 1, 8613+84] = .1  # builder has one wood
        data = {
            'trajectory_id': np.array([10, 10], dtype=np.int32),
            'step_id': np.array([0, 1], dtype=np.int32),
            'task_idx': np.array([0, 0], dtype=np.int32),
            'instruction_idx': np.array([0, 0], dtype=np.int32),
            'instruction_done': np.array([0, 1], dtype=np.float32),
            'done_all': np.array([False, True]),
            'actions': np.array([[1, 21, 52], [3, 11, 3]], dtype=np.int32),
            'obs': obs,
        }
        with tempfile.TemporaryDirectory() as temp:
            parts = Path(temp) / 'parts'
            parts.mkdir()
            pq.write_table(arrays_to_table(data), parts / 'part_000001.parquet')
            output = Path(temp) / 'out'
            args = Namespace(parts=parts, scenario_file=builder.DEFAULT_SCENARIOS,
                             constants_file=builder.CRAFTAX/'craftax/craftax_together_talens/constants.py',
                             output=output, hf_repo_id=None, hf_use_empty_repo=False,
                             hf_public=False, hf_shard_mib=1, source_hf_repo_id=None,
                             source_hf_revision=None, max_shards=0, skip_unreadable=False,
                             seed=42, val_fraction=.05, test_fraction=.05, history=5)
            with contextlib.redirect_stdout(io.StringIO()): builder.build(args)
            report = json.loads((output / 'report.json').read_text())
            records = [json.loads(line) for split in ('train', 'validation', 'test')
                       for line in (output / f'{split}.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 4)
            self.assertEqual(report['skipped_by_reason'],
                             {'craft_insufficient_materials': 1, 'sleeping_or_resting': 1})
            self.assertEqual({builder.split_for(x['metadata']['trajectory_id'], 42, .05, .05)
                              for x in records}, {builder.split_for(10, 42, .05, .05)})
            record = next(x for x in records if x['metadata']['step_id']==1 and x['metadata']['agent_id']==2)
            self.assertIn('GIVE_TO_TEAMMATE_1', record['messages'][0]['content'].split('## Your previous policy-selected actions')[1].split('## Actions')[0])
            self.assertEqual(record['messages'][1]['content'], '<action>UP</action>')
            self.assertTrue(all(x['metadata']['format_version'] == 'mappo_sft_v2' for x in records))


if __name__ == '__main__':
    unittest.main()
