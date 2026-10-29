"""Read-only real-artifact checks and rejection tests; never fit or predict."""
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest

from groups.group1_qda.kronodroid.src import run_qda_kronodroid as qda


@pytest.fixture(autouse=True)
def forbid_model_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('These contract tests must never fit models or predict labels.')

    for method in ('fit', 'predict', 'predict_proba'):
        monkeypatch.setattr(qda.QuadraticDiscriminantAnalysis, method, forbidden)


@pytest.fixture(scope='module')
def inputs():
    config_path = Path(__file__).resolve().parents[1] / 'configs/qda_kronodroid.yaml'
    config, root = qda.load_config(config_path)
    frame, rankings, provenance = qda.load_inputs(config, root)
    return config, root, frame, rankings, provenance


def test_real_splits_do_not_overlap(inputs):
    config, _, frame, _, _ = inputs
    c = config['contract']
    partitions = {
        split: set(frame.index[frame[c['split_column']].eq(split)])
        for split in c['split_names'].values()
    }
    for left, right in combinations(partitions, 2):
        assert partitions[left].isdisjoint(partitions[right])
    assert set.union(*partitions.values()) == set(frame.index)
    qda.validate_splits(frame, config)


def test_cross_split_duplicate_id_is_rejected(inputs):
    config, _, frame, _, _ = inputs
    c = config['contract']
    changed = frame.copy()
    ids = changed.index.to_numpy(copy=True)
    train_position = next(i for i, split in enumerate(changed[c['split_column']])
                          if split == c['split_names']['train'])
    test_position = next(i for i, split in enumerate(changed[c['split_column']])
                         if split == c['split_names']['test'])
    ids[test_position] = ids[train_position]
    changed.index = pd.Index(ids, name=frame.index.name)
    with pytest.raises(qda.DataContractError, match='duplicate sample IDs'):
        qda.validate_splits(changed, config)


@pytest.mark.parametrize('name', ['mi_top30', 'mi_top50', 'mi_top100', 'et_top30', 'et_top50', 'et_top100'])
def test_feature_lists_are_subsets_and_x_excludes_metadata(inputs, name):
    config, _, frame, rankings, _ = inputs
    c = config['contract']
    requested = rankings['top_k_sets'][name]
    feature_columns = set(frame.columns) - {c['split_column'], c['label_column']}
    assert set(requested) <= feature_columns
    retained, dropped = qda.select_features(frame, requested, config)
    X = frame.loc[:, retained]
    assert not set(X.columns).intersection(c['forbidden_features'])
    assert 'sha256' not in X.columns and 'sample_id' not in X.columns
    assert len(retained) + len(dropped) == len(requested)
    assert retained == [column for column in requested if column not in dropped]
    qda.validate_feature_lists(frame, rankings, config)


@pytest.mark.parametrize('column', ['split', 'label_binary', 'sha256', 'sample_id'])
def test_select_features_rejects_metadata(inputs, column):
    config, _, frame, rankings, _ = inputs
    columns = rankings['top_k_sets'][config['selection']['primary']] + [column]
    with pytest.raises(qda.DataContractError, match='Forbidden columns in X'):
        qda.select_features(frame, columns, config)


def test_feature_list_with_missing_column_is_rejected(inputs):
    config, _, frame, rankings, _ = inputs
    changed = deepcopy(rankings)
    name = config['selection']['primary']
    changed['top_k_sets'][name][0] = 'NONEXISTENT_TEST_FEATURE'
    with pytest.raises(qda.DataContractError, match='missing feature columns'):
        qda.validate_feature_lists(frame, changed, config)


def test_reordered_feature_list_is_rejected(inputs):
    config, _, frame, rankings, _ = inputs
    changed = deepcopy(rankings)
    name = config['selection']['primary']
    changed['top_k_sets'][name].reverse()
    with pytest.raises(qda.DataContractError, match='order differs'):
        qda.validate_feature_lists(frame, changed, config)


@pytest.mark.parametrize('relative_path', [
    'runs/kronodroid', 'runs/kronodroid/artifacts/new-output',
    'runs/real', 'runs/real/reports/new-output',
    'runs/group1_qda/../kronodroid/new-output',
    'runs/group1_qda/../real/new-output',
])
def test_output_guard_refuses_protected_paths(inputs, relative_path):
    _, root, _, _, _ = inputs
    with pytest.raises(qda.DataContractError, match='Refusing protected output path'):
        qda.guard_output_path(root / relative_path, root)


@pytest.mark.parametrize('protected', ['runs/kronodroid', 'runs/real'])
def test_run_creation_refuses_protected_output_before_mkdir(inputs, monkeypatch, protected):
    config, root, _, _, _ = inputs
    changed = deepcopy(config)
    changed['output']['directory'] = protected

    def unexpected_mkdir(*args, **kwargs):
        pytest.fail('Protected path reached mkdir.')

    monkeypatch.setattr(Path, 'mkdir', unexpected_mkdir)
    with pytest.raises(qda.DataContractError, match='Refusing protected output path'):
        qda.new_run_directory(changed, root)


def test_output_guard_allows_group1_output(inputs):
    _, root, _, _, _ = inputs
    requested = root / 'runs/group1_qda/kronodroid/contract-test-no-write'
    assert qda.guard_output_path(requested, root) == requested.resolve()


def test_zero_variance_filter_uses_train_only_and_preserves_order(inputs):
    config, _, _, _, _ = inputs
    c = config['contract']
    frame = pd.DataFrame({
        c['split_column']: [c['split_names']['train'], c['split_names']['train'],
                            c['split_names']['val'], c['split_names']['test']],
        'train_constant': [3.0, 3.0, 10.0, -2.0],
        'train_variable': [1.0, 2.0, 7.0, 7.0],
    })
    retained, dropped = qda.select_features(frame, ['train_constant', 'train_variable'], config)
    assert retained == ['train_variable']
    assert dropped == ['train_constant']


def test_candidate_tie_prefers_larger_reg_param(inputs):
    config, _, _, _, _ = inputs
    candidates = [{'val_macro_f1': 0.8, 'reg_param': reg}
                  for reg in reversed(config['qda']['reg_param_grid'])]
    assert max(candidates, key=qda.candidate_key)['reg_param'] == max(config['qda']['reg_param_grid'])
    candidates.append({'val_macro_f1': 0.9, 'reg_param': min(config['qda']['reg_param_grid'])})
    assert max(candidates, key=qda.candidate_key)['val_macro_f1'] == 0.9


def test_feature_set_selection_uses_val_score_and_configured_order(inputs):
    config, _, _, _, _ = inputs
    order = config['selection']['feature_sets']
    winners = {name: {'candidate': {'val_macro_f1': 0.8}} for name in reversed(order)}
    assert qda.choose_feature_set(winners, order) == config['selection']['primary']
    winners[order[-1]]['candidate']['val_macro_f1'] = 0.9
    assert qda.choose_feature_set(winners, order) == order[-1]
