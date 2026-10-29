"""Consume Group 3 KronoDroid artifacts; select QDA on validation, then test once.

No preprocessing or ranking is learned here except a TRAIN-only constant filter.
Use --validate-only for read-only contract checks without fitting or output files.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)


class DataContractError(ValueError):
    """An input or setting differs from the declared experiment."""


def require(condition, message):
    if not condition:
        raise DataContractError(message)


def repository_root(config_path):
    for parent in config_path.resolve().parents:
        if (parent / '.git').exists():
            return parent
    raise DataContractError('Cannot locate the repository root from config path.')


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    root = repository_root(path)
    s = config['selection']
    require(s['metric'] == 'macro_f1', 'Selection metric must be macro_f1.')
    require(s['candidate_tie'] == 'larger_reg_param', 'Unsupported candidate tie rule.')
    require(s['feature_set_tie'] == 'configured_order', 'Unsupported feature-set tie rule.')
    require(s['primary'] == s['feature_sets'][0], 'Primary must lead feature-set order.')
    require(len(set(s['feature_sets'])) == len(s['feature_sets']), 'Duplicate feature sets.')
    require(set(s['feature_sets']) <= set(config['contract']['top_k_sets']), 'Unknown feature set.')
    require(s['zero_variance'] == 0, 'Only exactly constant TRAIN features may be dropped.')
    require(s['variance_ddof'] == 0, 'Variance definition must use ddof=0.')
    grid = config['qda']['reg_param_grid']
    require(bool(grid) and len(set(grid)) == len(grid), 'Empty or duplicate reg_param grid.')
    require(all(not isinstance(x, bool) and np.isfinite(x) and 0 <= x <= 1 for x in grid),
            'Invalid reg_param grid.')
    for name, expected in config['qda']['expected_defaults'].items():
        actual = QuadraticDiscriminantAnalysis().get_params()[name]
        require(actual == expected, f'sklearn default mismatch: {name}={actual!r}.')
    for name in config['output']['filenames'].values():
        require(Path(name).name == name and name not in {'.', '..'}, 'Unsafe output filename.')
    require(len(set(config['output']['filenames'].values())) == len(config['output']['filenames']),
            'Output filenames must be distinct.')
    for name in s['feature_sets']:
        require(Path(name).name == name and name not in {'.', '..'}, 'Unsafe feature-set name.')
    return config, root


def sha256_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_feature_lists(frame, rankings, config):
    c = config['contract']
    features = [x for x in frame.columns if x not in {c['split_column'], c['label_column']}]
    require(not set(features).intersection(c['forbidden_features']), 'Metadata in feature columns.')
    require(not any(x.endswith(c['forbidden_suffix']) for x in features), 'Unexpected scaled suffix.')
    require(set(rankings) == {'rankings', 'top_k_sets'}, 'Ranking JSON keys differ.')
    require(set(rankings['rankings']) == set(c['ranking_methods']), 'Ranking methods differ.')
    require(set(rankings['top_k_sets']) == set(c['top_k_sets']), 'Top-K keys differ.')
    for method, columns in rankings['rankings'].items():
        require(len(columns) == len(features) and set(columns) == set(features),
                f'{method}: ranking is not the complete unique feature set.')
    for name, spec in c['top_k_sets'].items():
        columns = rankings['top_k_sets'][name]
        require(len(columns) == spec['k'] and len(set(columns)) == len(columns),
                f'{name}: incorrect length or repeated features.')
        require(set(columns) <= set(features), f'{name}: missing feature columns.')
        require(columns == rankings['rankings'][spec['method']][:spec['k']],
                f'{name}: feature order differs from ranking prefix.')
    return features


def validate_splits(frame, config):
    c = config['contract']
    require(frame.index.name == c['index_name'], 'Unexpected index name.')
    require(frame.index.is_unique and not frame.index.hasnans, 'Missing or duplicate sample IDs.')
    require(all(isinstance(x, str) and re.fullmatch(c['id_pattern'], x) for x in frame.index),
            'Index contains invalid SHA256 IDs.')
    require(frame[c['split_column']].value_counts().to_dict() == c['split_counts'],
            'Split counts differ from the contract.')
    require(set(c['split_names'].values()) == set(c['split_counts']), 'Configured splits differ.')
    ids = {}
    for role, split in c['split_names'].items():
        part = frame.loc[frame[c['split_column']] == split]
        require(part[c['label_column']].value_counts().to_dict() == c['label_counts'][split],
                f'{split}: label counts differ.')
        ids[role] = set(part.index)
    for role, values in ids.items():
        for other, other_values in ids.items():
            if role != other:
                require(values.isdisjoint(other_values), f'ID overlap: {role}/{other}.')


def load_inputs(config, root):
    paths = {name: (root / value).resolve() for name, value in config['inputs'].items()}
    before = {name: sha256_file(path) for name, path in paths.items()}
    frame = pd.read_parquet(paths['features'])
    manifest = pd.read_parquet(paths['manifest'])
    rankings = json.loads(paths['rankings'].read_text(encoding='utf-8'))
    group3 = json.loads(paths['group3_metrics'].read_text(encoding='utf-8'))
    c = config['contract']
    require(frame.columns.is_unique, 'Duplicate feature-table column names.')
    features = validate_feature_lists(frame, rankings, config)
    require(len(frame) == c['rows'] and len(features) == c['feature_count'], 'Input dimensions differ.')
    validate_splits(frame, config)
    require(all(pd.api.types.is_numeric_dtype(frame[x]) for x in features), 'Non-numeric features.')
    require(np.isfinite(frame[features].to_numpy()).all(), 'Missing or infinite feature values.')
    require(manifest[c['manifest_id']].is_unique, 'Duplicate manifest IDs.')
    require(set(manifest[c['manifest_id']]) == set(frame.index), 'Manifest IDs differ.')
    aligned = manifest.set_index(c['manifest_id']).loc[frame.index]
    require((aligned[c['split_column']].to_numpy() == frame[c['split_column']].to_numpy()).all(),
            'Manifest split assignments differ.')
    require((aligned[c['manifest_label']].to_numpy() == frame[c['label_column']].to_numpy()).all(),
            'Manifest labels differ.')
    require(aligned[c['source_column']].eq(c['source_value']).all(), 'Unexpected source dataset.')
    require(aligned[c['canonical_label_column']].tolist() == frame[c['label_column']].map(c['labels']).tolist(),
            'Benign/Malware label mapping differs.')
    after = {name: sha256_file(path) for name, path in paths.items()}
    require(before == after, 'Inputs changed while being read.')
    report = {
        'input_files': {name: {'path': config['inputs'][name], 'sha256': digest}
                        for name, digest in before.items()},
        'rows': len(frame), 'feature_count': len(features), 'index_name': frame.index.name,
        'split_counts': c['split_counts'], 'label_counts': c['label_counts'],
        'group3_metrics': group3,
    }
    return frame, rankings, report


def guard_output_path(path, root):
    """Resolve paths, including symlinks, before enforcing immutable Group 3 roots.

    Protected roots are deliberately fixed safety invariants, not YAML overrides.
    """
    path = Path(path).resolve()
    for protected in ('runs/kronodroid', 'runs/real'):
        boundary = (root / protected).resolve()
        require(not path.is_relative_to(boundary), f'Refusing protected output path: {path}')
    return path


def new_run_directory(config, root):
    base = guard_output_path(root / config['output']['directory'], root)
    run_id = datetime.now(timezone.utc).strftime(config['output']['timestamp_format'])
    require(bool(run_id) and Path(run_id).name == run_id and run_id not in {'.', '..'},
            'Unsafe timestamp directory name.')
    run_dir = guard_output_path(base / run_id, root)
    require(run_dir.is_relative_to((root / 'runs/group1_qda/kronodroid').resolve()),
            'Output must stay inside runs/group1_qda/kronodroid/.')
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path, payload, config):
    path.write_text(json.dumps(payload, indent=config['output']['json_indent'], allow_nan=False)
                    + '\n', encoding='utf-8')


def warning_records(caught):
    return [{'category': x.category.__name__, 'message': str(x.message),
             'filename': x.filename, 'lineno': x.lineno} for x in caught]


def select_features(frame, columns, config):
    c = config['contract']
    require(not set(columns).intersection(c['forbidden_features']), 'Forbidden columns in X.')
    require(set(columns) <= set(frame.columns), 'Missing selected features.')
    train = frame.loc[frame[c['split_column']] == c['split_names']['train'], columns]
    variance = train.var(ddof=config['selection']['variance_ddof'])
    dropped = variance.index[variance == config['selection']['zero_variance']].tolist()
    retained = [x for x in columns if x not in dropped]
    require(bool(retained), 'No features remain after dropping TRAIN constants.')
    return retained, dropped


def candidate_key(result):
    return result['val_macro_f1'], result['reg_param']


def choose_feature_set(winners, ordered_names):
    # max preserves the first configured entry on an exact cross-set score tie.
    return max(ordered_names, key=lambda name: winners[name]['candidate']['val_macro_f1'])


def search(frame, columns, directory, config):
    c, names = config['contract'], config['output']['filenames']
    retained, dropped = select_features(frame, columns, config)
    train = frame.loc[frame[c['split_column']] == c['split_names']['train']]
    val = frame.loc[frame[c['split_column']] == c['split_names']['val']]
    X_train, y_train = train[retained], train[c['label_column']]
    X_val, y_val = val[retained], val[c['label_column']]
    results, best = [], None
    started = time.perf_counter()
    for reg in config['qda']['reg_param_grid']:
        model = QuadraticDiscriminantAnalysis(reg_param=reg)
        caught = []
        result = {'reg_param': reg}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                fit_started = time.perf_counter()
                model.fit(X_train, y_train)
                fit_seconds = time.perf_counter() - fit_started
                prediction = model.predict(X_val)
                result.update({
                    'val_macro_f1': float(f1_score(y_val, prediction, average='macro',
                                                 zero_division=config['metrics']['zero_division'])),
                    'val_accuracy': float(accuracy_score(y_val, prediction)),
                    'fit_seconds': fit_seconds, 'status': 'PASS',
                })
            result['warnings'] = warning_records(caught)
            results.append(result)
            if best is None or candidate_key(result) > candidate_key(best['candidate']):
                best = {'model': model, 'candidate': result, 'features': retained, 'dropped': dropped}
        except Exception as exc:
            result.update({'status': 'FAIL', 'error': f'{type(exc).__name__}: {exc}',
                           'warnings': warning_records(caught)})
            results.append(result)
            write_json(directory / names['search'], results, config)
            raise
        write_json(directory / names['search'], results, config)
        print(json.dumps({'feature_set': directory.name, **result}), flush=True)
    best['search_seconds'] = time.perf_counter() - started
    return best


def evaluate(model, part, features, config):
    """One predict and one predict_proba call; all metrics reuse those arrays."""
    m = config['metrics']
    y = part[config['contract']['label_column']]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        prediction = model.predict(part[features])
        probabilities = model.predict_proba(part[features])
        require(np.isfinite(probabilities).all(), 'Non-finite predicted probabilities.')
        require(((probabilities >= 0) & (probabilities <= 1)).all(), 'Invalid probabilities.')
        require(np.allclose(probabilities.sum(axis=1), 1, atol=m['probability_atol'],
                            rtol=m['probability_rtol']), 'Probabilities do not sum to one.')
        positive_column = list(model.classes_).index(m['positive_label'])
        probability = probabilities[:, positive_column]
        metrics = {
            'accuracy': float(accuracy_score(y, prediction)),
            'balanced_accuracy': float(balanced_accuracy_score(y, prediction)),
            'malware_precision': float(precision_score(y, prediction, pos_label=m['positive_label'], zero_division=m['zero_division'])),
            'malware_recall': float(recall_score(y, prediction, pos_label=m['positive_label'], zero_division=m['zero_division'])),
            'malware_f1': float(f1_score(y, prediction, pos_label=m['positive_label'], zero_division=m['zero_division'])),
            'macro_f1': float(f1_score(y, prediction, average='macro', zero_division=m['zero_division'])),
            'roc_auc': float(roc_auc_score(y == m['positive_label'], probability)),
            'confusion_matrix': confusion_matrix(y, prediction, labels=m['confusion_labels']).tolist(),
            'confusion_matrix_labels': m['confusion_labels'], 'n_samples': len(part),
        }
    return metrics, prediction, probability, warning_records(caught)


def save_confusion(metrics, path, config):
    import matplotlib
    matplotlib.use(config['plot']['backend'])
    import matplotlib.pyplot as plt
    import seaborn as sns
    p = config['plot']
    fig, ax = plt.subplots(figsize=p['figsize'])
    labels = [config['contract']['labels'][x] for x in config['metrics']['confusion_labels']]
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt=p['format'], cmap=p['cmap'],
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(title=p['title'], xlabel=p['xlabel'], ylabel=p['ylabel'])
    fig.tight_layout()
    fig.savefig(path, dpi=p['dpi'])
    plt.close(fig)


def package_versions():
    return {'python': platform.python_version(), 'packages': dict(sorted(
        ((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()),
        key=lambda item: item[0].lower()))}


def run(config, root, frame, rankings, provenance):
    np.random.seed(config['random_seed'])
    directory = new_run_directory(config, root)
    names = config['output']['filenames']
    versions = package_versions()
    config_text = yaml.safe_dump(config, sort_keys=False)

    def save_metadata(target):
        (target / names['config']).write_text(config_text, encoding='utf-8')
        write_json(target / names['versions'], versions, config)
        (target / names['seed']).write_text(str(config['random_seed']) + '\n', encoding='utf-8')

    save_metadata(directory)
    write_json(directory / names['provenance'], provenance, config)
    status = {'state': 'searching', 'test_evaluations_started': [], 'test_evaluations_completed': []}
    write_json(directory / names['status'], status, config)
    try:
        winners = {}
        started = time.perf_counter()
        for name in config['selection']['feature_sets']:
            target = guard_output_path(directory / name, root)
            target.mkdir(exist_ok=False)
            save_metadata(target)
            winners[name] = search(frame, rankings['top_k_sets'][name], target, config)
        total_search_seconds = time.perf_counter() - started
        selected = choose_feature_set(winners, config['selection']['feature_sets'])
        summary = {'selected_feature_set': selected, 'primary_feature_set': config['selection']['primary'],
                   'selection_metric': config['selection']['metric'],
                   'feature_set_tie_rule': config['selection']['feature_set_tie'],
                   'candidate_tie_rule': config['selection']['candidate_tie'],
                   'selected_before_test': True, 'refit_on_train_plus_val': False,
                   'sklearn_qda_defaults': QuadraticDiscriminantAnalysis().get_params(),
                   'feature_sets': {name: {'chosen_reg_param': w['candidate']['reg_param'],
                       'val_macro_f1': w['candidate']['val_macro_f1'],
                       'requested_features': rankings['top_k_sets'][name],
                       'retained_features': w['features'], 'dropped_features': w['dropped'],
                       'classes': w['model'].classes_.tolist(),
                       'fitted_priors': w['model'].priors_.tolist()} for name, w in winners.items()}}
        # Persist the selection and all four frozen models before the first TEST call.
        write_json(directory / names['selection'], summary, config)
        write_json(directory / names['runtime'], {'total_all_feature_sets_search_seconds': total_search_seconds}, config)
        for name, w in winners.items():
            joblib.dump({'model': w['model'], 'features': w['features'],
                         'dropped_features': w['dropped'], 'classes': w['model'].classes_.tolist()},
                        directory / name / names['model'])
            write_json(directory / name / names['runtime'], {
                'final_model_fit_seconds': w['candidate']['fit_seconds'],
                'total_feature_set_search_seconds': w['search_seconds'],
                'final_model_fit_definition': 'Winning candidate fit call only; TRAIN only, no refit.',
                'search_definition': 'All candidate fits, VAL prediction/scoring, warning capture and search logging.',
            }, config)
        status['state'] = 'frozen_before_test'
        write_json(directory / names['status'], status, config)
        for name, w in winners.items():
            target, evaluation_warnings = directory / name, {}
            for role, split in config['contract']['split_names'].items():
                part = frame.loc[frame[config['contract']['split_column']] == split]
                if role == 'test':
                    require(name not in status['test_evaluations_started'], 'Repeated TEST evaluation refused.')
                    status['test_evaluations_started'].append(name)
                    write_json(directory / names['status'], status, config)
                metrics, prediction, probability, captured = evaluate(w['model'], part, w['features'], config)
                write_json(target / names[role], metrics, config)
                evaluation_warnings[role] = captured
                write_json(target / names['warnings'], evaluation_warnings, config)
                if role == 'test':
                    columns = config['output']['prediction_columns']
                    pd.DataFrame({columns['id']: part.index, columns['truth']: part[config['contract']['label_column']].to_numpy(),
                                  columns['prediction']: prediction, columns['probability']: probability}).to_csv(
                                      target / names['predictions'], index=False)
                    save_confusion(metrics, target / names['confusion'], config)
                    status['test_evaluations_completed'].append(name)
                    write_json(directory / names['status'], status, config)
        for name, record in provenance['input_files'].items():
            require(sha256_file(root / record['path']) == record['sha256'], f'Input changed during run: {name}')
        status['state'] = 'complete'
        write_json(directory / names['status'], status, config)
    except Exception as exc:
        status.update({'state': 'failed_do_not_resume_or_retune', 'error': f'{type(exc).__name__}: {exc}'})
        write_json(directory / names['status'], status, config)
        raise
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Repository-contained YAML experiment configuration.')
    parser.add_argument('--validate-only', action='store_true', help='Read-only: no fitting or output directory.')
    args = parser.parse_args(argv)
    try:
        config, root = load_config(args.config)
        frame, rankings, provenance = load_inputs(config, root)
        if args.validate_only:
            print(json.dumps(provenance, indent=config['output']['json_indent']))
        else:
            print(f'Completed frozen-model evaluation: {run(config, root, frame, rankings, provenance)}')
    except Exception as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
