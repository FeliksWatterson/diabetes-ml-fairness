"""Small reference calculations for the research-extension statistics."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import (
    get_brier_ci, get_decision_curve, get_delong_auc_ci,
    get_disparate_impact, get_paired_auc_test, get_subgroup_auc_tests,
)


class ResearchStatisticsTests(unittest.TestCase):
    def test_delong_against_pairwise_u_statistic_with_ties(self):
        y = np.array([1, 0, 1, 0, 1, 0, 1])
        score = np.array([.8, .2, .5, .5, .1, .8, .8])
        positive, negative = score[y == 1], score[y == 0]
        kernel = (positive[:, None] > negative).astype(float)
        kernel += .5 * (positive[:, None] == negative)
        expected_variance = (np.var(kernel.mean(axis=1), ddof=1) / len(positive)
                             + np.var(kernel.mean(axis=0), ddof=1) / len(negative))
        result = get_delong_auc_ci(y, score)
        self.assertAlmostEqual(result['roc_auc'], kernel.mean())
        self.assertAlmostEqual(result['roc_auc'], roc_auc_score(y, score))
        self.assertAlmostEqual(result['auc_variance'], expected_variance)
        self.assertLessEqual(result['auc_ci_lower'], result['roc_auc'])
        self.assertGreaterEqual(result['auc_ci_upper'], result['roc_auc'])

    def test_auc_missing_class_small_class_and_degenerate_scores(self):
        self.assertTrue(np.isnan(get_delong_auc_ci([0, 0], [.1, .2])['roc_auc']))
        small = get_delong_auc_ci([0, 0, 1], [.1, .2, .9])
        self.assertEqual(small['roc_auc'], 1)
        self.assertTrue(np.isnan(small['auc_ci_lower']))
        tied = get_delong_auc_ci([0, 0, 1, 1], [.5] * 4)
        self.assertEqual(tied['roc_auc'], .5)
        self.assertEqual(tied['auc_variance'], 0)
        with self.assertRaises(ValueError):
            get_delong_auc_ci([0, 2], [.1, .8])

    def test_paired_variance_uses_within_person_covariance(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        a = np.array([.1, .6, .3, .8, .4, .7])
        b = np.array([.2, .5, .7, .9, .1, .6])
        result = get_paired_auc_test(y, a, b)
        pos, neg = y == 1, y == 0
        kernels = []
        for score in [a, b]:
            kernels.append((score[pos, None] > score[neg]).astype(float)
                           + .5 * (score[pos, None] == score[neg]))
        difference = kernels[0] - kernels[1]
        variance = difference.mean(axis=1).var(ddof=1) / 3 + difference.mean(axis=0).var(ddof=1) / 3
        self.assertAlmostEqual(result['difference_se'] ** 2, variance)
        self.assertAlmostEqual(result['auc_difference'], roc_auc_score(y, a) - roc_auc_score(y, b))
        same = get_paired_auc_test(y, a, a)
        self.assertEqual(same['auc_difference'], 0)
        self.assertEqual(same['p_value'], 1)

    def test_independent_test_and_prespecified_bonferroni(self):
        frame = pd.DataFrame([
            dict(year=2017, model='LR', attribute='age', group=g, roc_auc=a, auc_variance=.01, n=100)
            for g, a in [('young', .8), ('middle', .7), ('old', .6)]
        ])
        tests = get_subgroup_auc_tests(frame, {'age': 'young'}, {'age': ['young', 'middle', 'old']})
        self.assertTrue(tests.bonferroni_family_size.eq(2).all())
        self.assertAlmostEqual(tests.iloc[0].auc_difference, -.1)
        self.assertAlmostEqual(tests.iloc[0].difference_se, np.sqrt(.02))
        np.testing.assert_allclose(tests.p_bonferroni, np.minimum(1, 2 * tests.p_value))

    def test_brier_mean_loss_and_empty_case(self):
        y, p = [0, 1, 1, 0], [.1, .8, .4, .3]
        loss = (np.array(p) - y) ** 2
        result = get_brier_ci(y, p)
        self.assertAlmostEqual(result['brier'], loss.mean())
        self.assertAlmostEqual(result['brier_se'], loss.std(ddof=1) / 2)
        self.assertTrue(np.isnan(get_brier_ci([], [])['brier_ci_lower']))
        with self.assertRaises(ValueError):
            get_brier_ci([1], [1.1])

    def test_dca_ties_and_baselines_by_hand(self):
        curve = get_decision_curve([0, 1, 1, 0], [.1, .5, .9, .5], [.5, .9])
        self.assertEqual(curve.iloc[0].tp, 2)
        self.assertEqual(curve.iloc[0].fp, 1)
        self.assertAlmostEqual(curve.iloc[0].net_benefit, .25)
        self.assertAlmostEqual(curve.iloc[0].screen_all_net_benefit, 0)
        self.assertAlmostEqual(curve.iloc[1].net_benefit, .25)
        self.assertTrue(curve.better_than_both.all())
        with self.assertRaises(ValueError):
            get_decision_curve([0, 1], [.2, .8], [0, .5])
        with self.assertRaises(ValueError):
            get_decision_curve([0, 1], [.2, .8], [.5, .5])

    def test_di_zero_reference_and_missing_group(self):
        frame = pd.DataFrame([
            dict(year=2017, model='LR', attribute='sex', group=g,
                 predicted_positive_rate=rate, n=n)
            for g, rate, n in [('Male', .2, 100), ('Female', .1, 100), ('Missing', np.nan, 0)]
        ])
        args = ({'sex': 'Male'}, {'sex': ['Male', 'Female']})
        di = get_disparate_impact(frame, *args).set_index('group')
        self.assertEqual(di.loc['Female', 'disparate_impact_ratio'], .5)
        self.assertFalse(di.loc['Missing', 'primary_comparison'])
        self.assertTrue(np.isnan(di.loc['Missing', 'disparate_impact_ratio']))
        frame.loc[frame.group.eq('Male'), 'predicted_positive_rate'] = 0
        self.assertTrue(get_disparate_impact(frame, *args).disparate_impact_ratio.isna().all())


if __name__ == '__main__':
    unittest.main()
