"""Numerical measurement conventions and valid multirate configurations."""
from pathlib import Path
import contextlib
import io
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import axis_tip, configure


class RuntimeTests(unittest.TestCase):
    def test_linear_crossing_and_missing_interface(self):
        self.assertAlmostEqual(axis_tip([-1, -.5, .5, 1, .5, -.5, -1], 3, 1), 4.5)
        self.assertAlmostEqual(axis_tip([-1, -.5, .5, 1, .5, -.5, -1], 3, -1), 1.5)
        self.assertTrue(math.isnan(axis_tip([1, 1, 1], 1, 1)))
        self.assertTrue(math.isnan(axis_tip([-1, -1, -1], 1, 1)))

    def test_paper_transport_sweep_keeps_phase_and_base_time(self):
        for model in ('thermal', 'solutal'):
            for factor in (15, 30, 45):
                cfg = configure(model, ['--flow', 'on', '--transport-factor', str(factor)])
                self.assertAlmostEqual(cfg.nF, 1/15)
                self.assertEqual(cfg.nG, 1)
                transport = cfg.nT if model == 'thermal' else cfg.nH
                self.assertAlmostEqual(transport / cfg.nF, factor)

    def test_final_times(self):
        self.assertEqual(configure('thermal', ['--flow', 'on']).steps, 240000)
        self.assertEqual(configure('thermal', []).steps, 16000)
        self.assertEqual(configure('solutal', ['--flow', 'on']).steps, 450000)
        self.assertEqual(configure('solutal', []).steps, 50000)
        cfg = configure('thermosolutal', [])
        time_star = cfg.steps * .004 / (2 * .554)**2
        self.assertEqual(cfg.steps, 1074206)
        self.assertGreaterEqual(time_star, 3500 - 1e-9)
        self.assertLess(time_star - 3500, .004 / (2 * .554)**2)

    def test_invalid_intervals_and_grid_are_rejected(self):
        for args in (['--tip-every', '0'], ['--transport-factor', '0'], ['--steps', '-1'], ['--nx', '15'], ['--nx', '65'], ['--flow', 'on', '--tip-every', '16']):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                configure('thermal', args)


if __name__ == '__main__':
    unittest.main()
