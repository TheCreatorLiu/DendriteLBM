"""Opt-in end-to-end tests: DENDRITE_RUN_SIM_TESTS=1, optional DENDRITE_TEST_DEVICE."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('DENDRITE_RUN_SIM_TESTS') == '1', 'Set DENDRITE_RUN_SIM_TESTS=1 for simulation checks')
class SimulationTests(unittest.TestCase):
    def test_all_models_and_flow_states(self):
        import numpy as np
        root = Path(__file__).resolve().parents[1]
        device = os.environ.get('DENDRITE_TEST_DEVICE', 'cpu')
        for model in ('thermal', 'solutal', 'thermosolutal'):
            for flow in ('off', 'on'):
                with self.subTest(model=model, flow=flow), tempfile.TemporaryDirectory() as folder:
                    out = Path(folder) / 'run'
                    args = [sys.executable, str(root / (model+'.py')), '--smoke', '--flow', flow, '--device', device, '--no-png', '--output', str(out)]
                    result = subprocess.run(args, capture_output=True, text=True, timeout=300)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(json.loads((out / 'run_status.json').read_text())['status'], 'completed')
                    with np.load(out / 'final_state.npz') as fields, np.load(out / 'fields_000000000.npz') as initial:
                        self.assertTrue(all(np.isfinite(fields[name]).all() for name in fields.files))
                        self.assertGreater(float(np.max(np.abs(fields['psi'] - initial['psi']))), 0)
                        speed = float(np.max(np.abs(fields['velocity'])))
                        if flow == 'off':
                            self.assertEqual(speed, 0)
                        else:
                            self.assertGreater(speed, 0)
                    with (out / 'tip_velocity.csv').open() as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertGreaterEqual(len(rows), 3)
                    self.assertTrue(all(math_isfinite(row['velocity_star_west']) for row in rows[1:]))


def math_isfinite(value):
    import math
    return math.isfinite(float(value))
