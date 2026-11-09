"""Reference checks for post-hoc analysis, including ties and paired resampling."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset_shift import (
    _metrics_from_counts, _score_blocks, bootstrap_scores, distribution_tables,
    notebook_training_source, prevalence_sensitivity, uncertainty_tables,
    self_report_sensitivity,
)


def fixture():
    frame = pd.DataFrame({
        'age':[18.,39.,40.,59.,60.,80.,30.,50.], 'bmi':[17.,20.,24.,27.,32.,35.,np.nan,26.],
        'sex':pd.array([1,2,1,2,1,2,1,2],dtype='Int64'),
        'race_ethnicity':pd.array(['White NH','Black NH','Hispanic','Other']*2,dtype='string'),
        'smoking_status':['never','current']*4,'physical_activity':[0,1]*4,
        'heart_attack':[0,1]*4,'stroke':[0,1]*4,'y_true':[0,1]*4,
    })
    return {'internal':'a','external':'b','cohorts':{'a':frame,'b':frame.copy()},
            'predictions':{(d,m):frame.assign(score=[.1,.3,.3,.8,.4,.7,.6,.6])
                           for d in ['a','b'] for m in ['LR','XGBoost']}}


class ShiftTests(unittest.TestCase):
    def test_count_metrics_match_sklearn_with_ties_and_zero_weights(self):
        rng=np.random.default_rng(14)
        for _ in range(30):
            y=rng.integers(0,2,40)
            score=rng.choice([.1,.4,.8],40)
            weights=rng.integers(0,5,40)
            if not weights[y==0].sum() or not weights[y==1].sum():
                continue
            actual=_metrics_from_counts(y,*_score_blocks(y,score),weights)
            expected=[roc_auc_score(y,score,sample_weight=weights),
                      average_precision_score(y,score,sample_weight=weights)]
            np.testing.assert_allclose(actual,expected,atol=1e-12)

    def test_bootstrap_matches_explicit_resampling_and_pairing(self):
        y=np.array([0,1,0,1,0,1])
        score=np.array([.2,.2,.7,.9,.4,.6])
        result=bootstrap_scores(y,np.column_stack([score,score]),30,16)
        np.testing.assert_allclose(result[0],result[1],equal_nan=True)
        rng=np.random.default_rng(16)
        for b in range(30):
            ix=rng.integers(len(y),size=len(y))
            if np.unique(y[ix]).size==2:
                np.testing.assert_allclose(result[0,b],[roc_auc_score(y[ix],score[ix]),
                                                       average_precision_score(y[ix],score[ix])])
            else:
                assert np.isnan(result[:,b]).all()
        self.assertTrue(np.isnan(bootstrap_scores([],np.empty((0,2)),10)).all())
        self.assertTrue(np.isnan(bootstrap_scores([1,1],[.2,.7],10)).all())

    def test_common_prevalence_constant_score_has_ap_reference(self):
        bundle=fixture()
        for f in bundle['predictions'].values():
            f['score']=.5
        table=prevalence_sensitivity(bundle,.17)['prevalence_sensitivity']
        np.testing.assert_allclose(table.loc[table['status'].eq('ok'),'ap_standardized'],.17)
        with self.assertRaises(ValueError):
            prevalence_sensitivity(bundle,1)

    def test_shift_identical_cohorts_and_all_missing(self):
        result=distribution_tables(fixture())
        shifts=result['input_shift'].query("status == 'ok'")
        for col in ['ks_statistic','wasserstein','standardized_mean_difference','total_variation','delta_missing_rate']:
            np.testing.assert_allclose(shifts[col].dropna(),0)
        missing_bmi=result['input_shift'].query("attribute == 'bmi' and group == 'Missing' and feature == 'bmi'")
        self.assertTrue(missing_bmi['status'].ne('ok').all())

    def test_ci_model_difference_is_exactly_zero_for_identical_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            result=uncertainty_tables(fixture(),directory,n_boot=40,progress=None)
            # Resume with cached replicates must produce identical tables.
            second=uncertainty_tables(fixture(),directory,n_boot=40,progress=None)
            for key in result:
                pd.testing.assert_frame_equal(result[key],second[key])
        valid=result['paired_model_delta_ci'].query("status == 'ok'")
        np.testing.assert_allclose(valid[['estimate','ci_low','ci_high']],0)

    def test_training_fingerprint_ignores_only_marked_extension(self):
        old="code_source = '\\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')"
        new="code_source = __import__('src.dataset_shift', fromlist=['notebook_training_source']).notebook_training_source(notebook)"
        nb={'cells':[{'cell_type':'code','source':[old+'\nN_TRIALS = 20'],'metadata':{}}]}
        expected=notebook_training_source(nb)
        changed=copy.deepcopy(nb)
        changed['cells'][0]['source']=[new+'\nN_TRIALS = 20']
        changed['cells'].append({'cell_type':'code','source':['additional_analysis()'],'metadata':{'analysis_extension':True}})
        self.assertEqual(notebook_training_source(changed),expected)
        changed['cells'][0]['source']=[new+'\nN_TRIALS = 21']
        self.assertNotEqual(notebook_training_source(changed),expected)

    def test_alternative_labels_join_by_id_and_compare_common_people(self):
        bundle=fixture()
        for model in ['LR','XGBoost']:
            bundle['predictions'][('b',model)]=bundle['predictions'][('b',model)].assign(
                SEQN=np.arange(8),cycle=['2015-2016']*4+['2017-2018']*4)
        before={key:frame.copy(deep=True) for key,frame in bundle['predictions'].items()}
        def raw(path,format):
            offset=0 if '2015-2016' in str(path) else 4
            # Deliberately reverse subject order to catch positional joins.
            return pd.DataFrame({'SEQN':np.arange(4)+offset,'DIQ010':[1,2,3,9]}).iloc[::-1]
        with tempfile.TemporaryDirectory() as directory:
            for cycle,suffix in [('2015-2016','I'),('2017-2018','J')]:
                path=Path(directory)/f'data/raw/nhanes/{cycle}/DIQ_{suffix}.xpt'
                path.parent.mkdir(parents=True); path.touch()
            with patch('src.dataset_shift.pd.read_sas',side_effect=raw):
                tables,_=self_report_sensitivity(bundle,directory)
        coverage=tables['label_coverage'].query("dataset == 'b'").iloc[0]
        self.assertEqual(coverage['n_included'],6)
        self.assertEqual(coverage['n_excluded_unknown_self_report'],2)
        rows=tables['label_sensitivity'].query("dataset == 'b' and attribute == 'overall' and model == 'LR'")
        self.assertEqual(set(rows['n']),{6})
        self.assertEqual(rows.set_index('endpoint').loc['self_report','n_positive'],2)
        for key in before:
            pd.testing.assert_frame_equal(before[key],bundle['predictions'][key])


if __name__=='__main__':
    unittest.main()
