"""Post-hoc uncertainty, input-shift and endpoint sensitivity; never fits models.

Bootstrap uses ordinary person-level sampling and fixed saved scores. It does
not estimate retraining uncertainty or complex-survey sampling uncertainty.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import average_precision_score, roc_auc_score

from .utils import get_fairness_groups, get_fairness_group_specs

FEATURES = ['age', 'sex', 'race_ethnicity', 'bmi', 'smoking_status',
            'physical_activity', 'heart_attack', 'stroke']
NUMERIC = ['age', 'bmi']
MODELS = ['LR', 'XGBoost']


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def notebook_training_source(notebook):
    """Exclude appended analyses from run identity, retaining the legacy hash.

    Canonicalize only the fingerprint-call line to its original spelling.
    Every other byte of each original code cell still contributes to identity.
    Analysis code has its own hash in the extension manifest.
    """
    old = "code_source = '\\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')"
    new = "code_source = __import__('src.dataset_shift', fromlist=['notebook_training_source']).notebook_training_source(notebook)"
    return '\n'.join(''.join(c['source']).replace(new, old) for c in notebook['cells']
                     if c['cell_type'] == 'code' and not c.get('metadata', {}).get('analysis_extension'))


def group_labels(frame):
    labels, specs = get_fairness_groups(frame)
    if not frame['sex'].dropna().isin([1, 2]).all():
        raise ValueError('Unexpected sex code')
    labels['sex'] = frame['sex'].map({1: 'Nam', 2: 'Nữ'}).astype('string').fillna('Missing')
    return labels, {'sex': ['Nam', 'Nữ'], **specs}


def masks(frame):
    labels, specs = group_labels(frame)
    yield 'overall', 'All', np.ones(len(frame), dtype=bool)
    for attribute, groups in specs.items():
        for group in groups + ['Missing']:
            yield attribute, group, labels[attribute].eq(group).to_numpy(dtype=bool)


def load_saved_analysis(project, direction, reverse_run='79dce7d27030baa2'):
    """Join by subject keys, check saved labels/features and align both models."""
    project = Path(project)
    paths = {'nhanes': project / 'data/processed/nhanes_2015_2018.parquet',
             'brfss': project / 'data/processed/brfss_2017.parquet'}
    nhanes, brfss = (pd.read_parquet(paths[name]) for name in ['nhanes', 'brfss'])
    brfss = brfss.assign(source_row_id=brfss.index.to_numpy(copy=True))
    if direction == '07':
        internal, external = 'NHANES_OOF', 'BRFSS_2017'
        predictions = {}
        for model in MODELS:
            paths[f'internal_{model}'] = project / f'results/nhanes_to_brfss/tuning/3d15f3dbb58256ec/oof_{model}.parquet'
            paths[f'external_{model}'] = project / f'results/nhanes_to_brfss/external_validation/54c463bbe9079503/brfss_2017_{model}_predictions.parquet'
            for dataset, name, source, keys in [
                (internal, f'internal_{model}', nhanes, ['SEQN', 'cycle']),
                (external, f'external_{model}', brfss, ['year', 'source_row_id']),
            ]:
                saved = pd.read_parquet(paths[name]).rename(columns={'diabetes_label': 'y_true', 'probability': 'score'})
                if dataset == internal:
                    assert saved['run_id'].eq('3d15f3dbb58256ec').all()
                    assert set(saved['outer_fold']) == {1, 2, 3, 4, 5}
                joined = saved.merge(source[keys + FEATURES + ['diabetes_label']], on=keys,
                                     suffixes=('_saved', ''), validate='one_to_one', indicator=True)
                assert len(joined) == len(saved) == len(source) and joined['_merge'].eq('both').all()
                assert joined['y_true'].eq(joined['diabetes_label']).all()
                for feature in FEATURES:
                    if feature + '_saved' in joined:
                        pd.testing.assert_series_equal(joined[feature + '_saved'], joined[feature], check_names=False)
                predictions[(dataset, model)] = joined[keys + FEATURES + ['y_true', 'score']].sort_values(keys).reset_index(drop=True)
    elif direction == '08':
        internal, external = 'BRFSS_internal_test', 'NHANES_external'
        directory = project / 'results/brfss_to_nhanes/holdout_validation' / reverse_run
        paths['run_manifest'] = directory / 'manifest.json'
        manifest = json.loads(paths['run_manifest'].read_text(encoding='utf-8'))
        assert manifest['run_id'] == reverse_run and manifest['same_fitted_pipeline_for_both_tests']
        for category in ['source_sha256', 'model_sha256', 'output_sha256']:
            for name, digest in manifest[category].items():
                assert file_hash(project / name) == digest, name
        predictions = {}
        for dataset, source, keys in [(internal, brfss, ['year', 'source_row_id']),
                                      (external, nhanes, ['SEQN', 'cycle'])]:
            for model in MODELS:
                paths[f'{dataset}_{model}'] = directory / 'predictions' / f'{dataset}_{model}.parquet'
                saved = pd.read_parquet(paths[f'{dataset}_{model}']).sort_values(keys).reset_index(drop=True)
                assert saved['run_id'].eq(reverse_run).all() and saved['model'].eq(model).all()
                assert saved['dataset'].eq(dataset).all()
                joined = saved.merge(source[keys + FEATURES + ['diabetes_label']], on=keys,
                                     suffixes=('_saved', ''), validate='one_to_one', indicator=True)
                assert len(joined) == len(saved) and joined['_merge'].eq('both').all()
                assert joined['y_true'].eq(joined['diabetes_label']).all()
                for feature in FEATURES:
                    pd.testing.assert_series_equal(joined[feature + '_saved'], joined[feature], check_names=False)
                predictions[(dataset, model)] = joined[keys + FEATURES + ['y_true', 'score']].sort_values(keys).reset_index(drop=True)
        paths['membership'] = directory / 'split_membership.parquet'
        membership = pd.read_parquet(paths['membership'])
        actual_ids = set(predictions[(internal, 'LR')]['source_row_id'])
        expected_ids = set(membership.loc[membership['partition'].eq('internal_test'), 'source_row_id'])
        assert actual_ids == expected_ids
    else:
        raise ValueError('direction must be 07 or 08')
    cohorts = {}
    for dataset in [internal, external]:
        lr, xg = (predictions[(dataset, model)] for model in MODELS)
        pd.testing.assert_frame_equal(lr.drop(columns='score'), xg.drop(columns='score'))
        assert lr['y_true'].isin([0, 1]).all() and lr['y_true'].notna().all()
        for frame in [lr, xg]:
            assert np.isfinite(frame['score'].to_numpy(dtype=float)).all()
        cohorts[dataset] = lr.drop(columns='score').copy()
    return {'predictions': predictions, 'cohorts': cohorts, 'internal': internal,
            'external': external, 'paths': paths, 'direction': direction}


def _score_blocks(y, scores):
    """Sorted tied-score bins: bootstrap can reuse the order without re-sorting."""
    order = np.argsort(-np.asarray(scores, dtype=float), kind='stable')
    sorted_score = np.asarray(scores)[order]
    first = np.r_[True, sorted_score[1:] != sorted_score[:-1]]
    bins = np.empty(len(order), dtype=int)
    bins[order] = np.cumsum(first) - 1
    return bins, int(first.sum())


def _metrics_from_counts(y, bins, n_bins, counts):
    positive = np.bincount(bins, weights=counts * y, minlength=n_bins)
    negative = np.bincount(bins, weights=counts * (1 - y), minlength=n_bins)
    p, n = positive.sum(), negative.sum()
    if p == 0 or n == 0:
        return np.array([np.nan, np.nan])
    tp, fp = positive.cumsum(), negative.cumsum()
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    ap = np.sum(positive / p * precision)
    auc = np.sum(negative / n * (tp - positive / 2) / p)
    return np.array([auc, ap])


def bootstrap_scores(y, scores, n_boot=1000, seed=42):
    """Ordinary IID percentile bootstrap; models share sampled persons.

    Return model x replicate x [AUC, AP], retaining failed one-class replicates.
    """
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores[:, None]
    if scores.shape[0] != len(y) or not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid labels/scores or unaligned rows')
    if n_boot < 2:
        raise ValueError('At least two bootstrap replicates required')
    result = np.full((scores.shape[1], n_boot, 2), np.nan)
    if np.unique(y).size != 2:
        return result
    blocks = [_score_blocks(y, scores[:, model]) for model in range(scores.shape[1])]
    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        counts = np.bincount(rng.integers(len(y), size=len(y)), minlength=len(y))
        for model, (bins, n_bins) in enumerate(blocks):
            result[model, b] = _metrics_from_counts(y, bins, n_bins, counts)
    return result


def _interval(samples, confidence):
    valid = samples[np.isfinite(samples)]
    # Do not report an interval based on very few successful replicates.
    required = max(2, int(np.ceil(.9 * len(samples))))
    if len(valid) < required:
        return np.nan, np.nan, len(valid), 'insufficient_valid_replicates'
    lo, hi = np.quantile(valid, [(1-confidence)/2, (1+confidence)/2])
    return float(lo), float(hi), len(valid), 'ok'


def uncertainty_tables(bundle, output_dir, n_boot=1000, confidence=.95, seed=42, progress=print):
    """CI per cohort/group, external-minus-internal, and paired XGBoost-minus-LR."""
    if not 0 < confidence < 1:
        raise ValueError('confidence must lie between zero and one')
    output_dir = Path(output_dir)
    cache = output_dir / 'bootstrap_cache'
    cache.mkdir(parents=True, exist_ok=True)
    rows, delta_rows, model_rows = [], [], []
    internal, external = bundle['internal'], bundle['external']
    group_masks = {d: {(a, g): mask for a, g, mask in masks(frame)}
                   for d, frame in bundle['cohorts'].items()}
    for k, (attribute, group) in enumerate(group_masks[internal]):
        samples, points, counts = {}, {}, {}
        for d, dataset in enumerate([internal, external]):
            mask = group_masks[dataset][(attribute, group)]
            frame = bundle['predictions'][(dataset, 'LR')].loc[mask]
            y = frame['y_true'].to_numpy(dtype=int)
            scores = np.column_stack([bundle['predictions'][(dataset, m)].loc[mask, 'score'] for m in MODELS])
            identity = hashlib.sha256(y.tobytes() + scores.tobytes() + f'{n_boot}:{seed}:{k}:{d}'.encode()).hexdigest()
            cached = cache / (identity + '.npz')
            if cached.exists():
                with np.load(cached) as saved:
                    samples[dataset] = saved['samples']
            else:
                samples[dataset] = bootstrap_scores(y, scores, n_boot, seed + k*2 + d)
                np.savez_compressed(cached, samples=samples[dataset])
            counts[dataset] = {'n': len(y), 'n_positive': int(y.sum()), 'n_negative': int(len(y)-y.sum())}
            points[dataset] = np.array([_metrics_from_counts(y, *_score_blocks(y, scores[:, m]), np.ones(len(y)))
                                        if len(y) and np.unique(y).size == 2 else [np.nan, np.nan]
                                        for m in range(len(MODELS))])
            for m, model in enumerate(MODELS):
                for j, metric in enumerate(['auc_roc', 'ap']):
                    low, high, valid, status = _interval(samples[dataset][m, :, j], confidence)
                    rows.append({'dataset': dataset, 'model': model, 'attribute': attribute, 'group': group,
                                 **counts[dataset], 'metric': metric, 'estimate': points[dataset][m, j],
                                 'ci_low': low, 'ci_high': high, 'n_boot_valid': valid,
                                 'n_boot': n_boot, 'confidence': confidence, 'status': status})
            for j, metric in enumerate(['auc_roc', 'ap']):
                low, high, valid, status = _interval(samples[dataset][1, :, j] - samples[dataset][0, :, j], confidence)
                model_rows.append({'dataset': dataset, 'attribute': attribute, 'group': group,
                                   **counts[dataset], 'metric': metric, 'contrast': 'XGBoost - LR',
                                   'estimate': points[dataset][1, j]-points[dataset][0, j],
                                   'ci_low': low, 'ci_high': high, 'n_boot_valid': valid, 'n_boot': n_boot,
                                   'confidence': confidence, 'status': status})
        for m, model in enumerate(MODELS):
            for j, metric in enumerate(['auc_roc', 'ap']):
                low, high, valid, status = _interval(samples[external][m, :, j] - samples[internal][m, :, j], confidence)
                delta_rows.append({'model': model, 'attribute': attribute, 'group': group, 'metric': metric,
                                   'internal': internal, 'external': external,
                                   **{key+'_internal': value for key, value in counts[internal].items()},
                                   **{key+'_external': value for key, value in counts[external].items()},
                                   'estimate': points[external][m, j]-points[internal][m, j],
                                   'ci_low': low, 'ci_high': high, 'n_boot_valid': valid, 'n_boot': n_boot,
                                   'confidence': confidence, 'status': status})
        if progress:
            progress(f'CI {attribute}/{group}: done ({n_boot} replicates, cached)', flush=True)
    return {'metric_ci': pd.DataFrame(rows), 'external_internal_delta_ci': pd.DataFrame(delta_rows),
            'paired_model_delta_ci': pd.DataFrame(model_rows)}


def distribution_tables(bundle):
    """Descriptive shift only: no p-values or automatic causal interpretation."""
    internal, external = bundle['internal'], bundle['external']
    frames = bundle['cohorts']
    gm = {d: {(a,g): m for a,g,m in masks(f)} for d,f in frames.items()}
    numeric_rows, category_rows, shift_rows, missing_rows, composition_rows = [], [], [], [], []
    for attribute, group in gm[internal]:
        subsets = {d: frames[d].loc[gm[d][(attribute,group)]] for d in frames}
        for dataset, frame in subsets.items():
            composition_rows.append({'dataset': dataset, 'attribute': attribute, 'group': group,
                                     'n': len(frame), 'fraction_of_cohort': len(frame)/len(frames[dataset]),
                                     'n_positive': int(frame['y_true'].sum()),
                                     'positive_rate': frame['y_true'].mean()})
            for feature in FEATURES:
                missing_rows.append({'dataset': dataset, 'attribute': attribute, 'group': group,
                                     'feature': feature, 'n': len(frame), 'n_missing': int(frame[feature].isna().sum()),
                                     'missing_rate': frame[feature].isna().mean()})
        for feature in FEATURES:
            a, b = (subsets[d][feature] for d in [internal,external])
            base = {'attribute': attribute, 'group': group, 'feature': feature,
                    'n_internal': len(a), 'n_external': len(b),
                    'delta_missing_rate': b.isna().mean() - a.isna().mean()}
            if feature in NUMERIC:
                values = {}
                for dataset, series in [(internal,a), (external,b)]:
                    observed = series.dropna().to_numpy(dtype=float)
                    values[dataset] = observed
                    qs = np.quantile(observed,[.1,.25,.5,.75,.9]) if len(observed) else [np.nan]*5
                    numeric_rows.append({'dataset': dataset, 'attribute': attribute, 'group': group,
                                         'feature': feature, 'n_observed': len(observed),
                                         'mean': observed.mean() if len(observed) else np.nan,
                                         'sd': observed.std(ddof=1) if len(observed)>1 else np.nan,
                                         **dict(zip(['p10','p25','median','p75','p90'], qs))})
                x, y = values[internal], values[external]
                if len(x) and len(y):
                    pooled_sd = np.sqrt((x.var(ddof=1)+y.var(ddof=1))/2) if min(len(x),len(y))>1 else np.nan
                    shift_rows.append({**base, 'kind': 'numeric', 'ks_statistic': ks_2samp(x,y,method='exact' if min(len(x),len(y))<2 else 'asymp').statistic,
                                       'wasserstein': wasserstein_distance(x,y),
                                       'standardized_mean_difference': (y.mean()-x.mean())/pooled_sd if pooled_sd>0 else np.nan,
                                       'total_variation': np.nan, 'status': 'ok'})
                else:
                    shift_rows.append({**base,'kind':'numeric','status':'no_observed_values_in_one_or_both_cohorts'})
            else:
                ac, bc = (s.astype('string').fillna('Missing') for s in [a,b])
                categories = sorted(set(ac) | set(bc))
                tv = 0.0
                for category in categories:
                    pa = ac.eq(category).mean() if len(ac) else np.nan
                    pb = bc.eq(category).mean() if len(bc) else np.nan
                    tv += abs(pb-pa)/2
                    for dataset, s, rate in [(internal,ac,pa),(external,bc,pb)]:
                        category_rows.append({'dataset':dataset,'attribute':attribute,'group':group,
                                              'feature':feature,'category':category,'n':len(s),
                                              'count':int(s.eq(category).sum()),'fraction':rate})
                shift_rows.append({**base,'kind':'categorical','total_variation':tv if len(a) and len(b) else np.nan,
                                   'status':'ok' if len(a) and len(b) else 'empty_cohort'})
    return {'numeric_summary':pd.DataFrame(numeric_rows), 'categorical_distribution':pd.DataFrame(category_rows),
            'input_shift':pd.DataFrame(shift_rows),'missingness':pd.DataFrame(missing_rows),
            'group_composition':pd.DataFrame(composition_rows)}


def prevalence_sensitivity(bundle, reference_prevalence=.15):
    """Class reweighting to one common prevalence within every group and cohort.

    Weight sums of positive/negative classes become p_ref and 1-p_ref.
    Scores, labels and ranking are unchanged. This is a sensitivity analysis,
    not a correction of covariate shift or a causal adjustment.
    """
    if not 0 < reference_prevalence < 1:
        raise ValueError('reference_prevalence must lie between zero and one')
    rows=[]
    for (dataset,model), frame in bundle['predictions'].items():
        for attribute,group,mask in masks(frame):
            part=frame.loc[mask]
            y=part['y_true'].to_numpy(dtype=int)
            score=part['score'].to_numpy(dtype=float)
            p,n=int(y.sum()),int(len(y)-y.sum())
            weights=np.where(y==1,reference_prevalence/max(p,1),(1-reference_prevalence)/max(n,1))
            rows.append({'dataset':dataset,'model':model,'attribute':attribute,'group':group,
                         'n':len(y),'n_positive':p,'n_negative':n,'observed_prevalence':y.mean() if len(y) else np.nan,
                         'reference_prevalence':reference_prevalence,
                         'ap_observed':average_precision_score(y,score) if p and n else np.nan,
                         'ap_standardized':average_precision_score(y,score,sample_weight=weights) if p and n else np.nan,
                         'auc_roc':roc_auc_score(y,score) if p and n else np.nan,
                         'status':'ok' if p and n else 'not_two_classes'})
    table=pd.DataFrame(rows)
    _,primary=get_fairness_group_specs()
    primary={'sex':['Nam','Nữ'],**primary}
    macro=[]
    for (dataset,model),part in table.groupby(['dataset','model'],sort=False):
        for attribute,groups in primary.items():
            selected=part.loc[part['attribute'].eq(attribute)].set_index('group').reindex(groups)
            valid=selected['ap_standardized'].notna()
            macro.append({'dataset':dataset,'model':model,'attribute':attribute,
                          'reference_prevalence':reference_prevalence,'n_groups':len(groups),'n_groups_valid':int(valid.sum()),
                          'map_observed':selected['ap_observed'].mean() if valid.all() else np.nan,
                          'map_standardized':selected['ap_standardized'].mean() if valid.all() else np.nan})
    return {'prevalence_sensitivity':table,'prevalence_macro_ap':pd.DataFrame(macro)}


def self_report_sensitivity(bundle, project):
    """Frozen-score endpoint sensitivity; no training on the alternative label.

    NHANES DIQ010: yes=1, no/borderline=0; other/missing excluded.
    BRFSS DIABETE3: yes=1, pregnancy-only/no/prediabetes=0, as notebook 01.
    Similar self-report endpoints, not proven identical case definitions.
    """
    project=Path(project)
    paths={}
    raw=[]
    for cycle,suffix in [('2015-2016','I'),('2017-2018','J')]:
        path=project/f'data/raw/nhanes/{cycle}/DIQ_{suffix}.xpt'
        paths[cycle]=path
        if not path.exists():
            return {'label_sensitivity_status':pd.DataFrame([{'status':'unavailable','reason':f'Missing {path}'}])},paths
        part=pd.read_sas(path,format='xport')[['SEQN','DIQ010']]
        part['cycle']=cycle
        raw.append(part)
    labels=pd.concat(raw,ignore_index=True)
    assert not labels.duplicated(['SEQN','cycle']).any()
    transition_rows,coverage_rows=[],[]
    altered={'predictions':{},'cohorts':{},'internal':bundle['internal'],'external':bundle['external']}
    for (dataset,model),frame in bundle['predictions'].items():
        if 'SEQN' in frame:
            joined=frame.merge(labels,on=['SEQN','cycle'],how='left',validate='one_to_one',indicator=True)
            assert joined['_merge'].eq('both').all()
            joined['alternative_y']=joined['DIQ010'].map({1:1,2:0,3:0}).astype('Int64')
        else:
            joined=frame.copy()
            joined['alternative_y']=joined['y_true'].astype('Int64')
        if model=='LR':
            coverage_rows.append({'dataset':dataset,'n_original':len(joined),'n_included':int(joined['alternative_y'].notna().sum()),
                                  'n_excluded_unknown_self_report':int(joined['alternative_y'].isna().sum()),
                                  'n_label_changed':int(joined['alternative_y'].ne(joined['y_true']).fillna(False).sum())})
            counts=joined.assign(alternative_y=joined['alternative_y'].astype('string').fillna('Excluded')).groupby(
                ['y_true','alternative_y'],dropna=False).size()
            transition_rows.extend({'dataset':dataset,'original_y':a,'alternative_y':b,'n':int(n)} for (a,b),n in counts.items())
        # Also evaluate original labels on this same restricted cohort.
        included=joined.loc[joined['alternative_y'].notna()].copy().reset_index(drop=True)
        original=included.copy()
        included['y_true']=included['alternative_y'].astype(int)
        for name,part in [('original_on_common_subset',original),('self_report',included)]:
            altered['predictions'][(dataset+'__'+name,model)]=part
    result_rows=[]
    for (dataset,model),frame in altered['predictions'].items():
        for attribute,group,mask in masks(frame):
            part=frame.loc[mask]
            y=part['y_true'].to_numpy(dtype=int)
            valid=np.unique(y).size==2
            result_rows.append({'dataset':dataset.split('__')[0],'endpoint':dataset.split('__')[1],
                                'model':model,'attribute':attribute,'group':group,'n':len(y),
                                'n_positive':int(y.sum()),'n_negative':int(len(y)-y.sum()),
                                'positive_rate':y.mean() if len(y) else np.nan,
                                'auc_roc':roc_auc_score(y,part['score']) if valid else np.nan,
                                'ap':average_precision_score(y,part['score']) if valid else np.nan,
                                'status':'ok' if valid else 'not_two_classes'})
    return {'label_sensitivity':pd.DataFrame(result_rows),'label_coverage':pd.DataFrame(coverage_rows),
            'label_transition':pd.DataFrame(transition_rows)},paths
