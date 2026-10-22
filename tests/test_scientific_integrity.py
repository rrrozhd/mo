"""Offline regressions for scientific run integrity and deterministic sampling."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from src import benchmark, similarity_bands
from src.chemistry import ENZYMES


ROOT = Path(benchmark.__file__).resolve().parents[1]


def test_audit_sampling_is_stable_across_python_hash_seeds():
    # A tiny loader fixture isolates Python's set order from remote data and TDC.
    program = textwrap.dedent('''
        import json
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        import pandas as pd
        from src.research import dataset_similarity

        train = ['C', 'CC', 'CCC', 'CCCC', 'CCO', 'CCN', 'CCCl', 'CCBr',
                 'c1ccccc1', 'C1CCCCC1']
        test = ['CCCO', 'CCCCO', 'CCCN', 'CCCCN', 'CO', 'CN', 'c1ccncc1',
                'c1ccccc1O', 'CC(=O)O', 'CC(=O)N']
        data = SimpleNamespace(get_split=lambda **kwargs: {
            'train': pd.DataFrame({'Drug': train}),
            'test': pd.DataFrame({'Drug': test}),
        })
        fake_tdc = SimpleNamespace(single_pred=SimpleNamespace(ADME=lambda **kwargs: data))
        with patch.dict(sys.modules, {'tdc': fake_tdc}), \
             patch.object(dataset_similarity, 'MAX_POOL', 3), patch.object(dataset_similarity, 'MAX_QUERIES', 4):
            print(json.dumps(dataset_similarity.audit_dataset('fixture', 'ADME', 'unused'), sort_keys=True))
    ''')
    outputs = []
    for seed in ('1', '2', '3'):
        result = subprocess.run(
            [sys.executable, '-c', program], cwd=ROOT,
            env={**os.environ, 'PYTHONHASHSEED': seed},
            capture_output=True, text=True, check=True,
        )
        outputs.append(json.loads(result.stdout))
    assert outputs[0]['status'] == 'ok'
    assert outputs[0]['n_pool'] == 3
    assert outputs[0]['n_queries'] == 4
    assert all(output == outputs[0] for output in outputs[1:])


@pytest.fixture
def frozen_run(tmp_path, request):
    identities = getattr(request, "param", ("m0", "m1"))
    artifacts = tmp_path / 'artifacts'
    (artifacts / 'models').mkdir(parents=True)
    rows = [
        {'molecule_id': identities[i], 'smiles': smiles, 'scaffold': '', 'split': 'test',
         **{enzyme: i for enzyme in ENZYMES}}
        for i, smiles in enumerate(('CCO', 'CCN'))
    ]
    # Preserve one absent endpoint label through verify_run's CSV parsing.
    rows[0][ENZYMES[0]] = None
    pd.DataFrame(rows).to_csv(artifacts / 'test.csv', index=False)
    pd.DataFrame([{'smiles': 'CCCC', 'split': 'train'}]).to_csv(
        artifacts / 'manifest.csv', index=False)
    (artifacts / 'manifest.json').write_text('{}')
    for enzyme in ENZYMES:
        (artifacts / 'models' / f'{enzyme}.joblib').write_bytes(b'fixture; never deserialized')
    run = tmp_path / 'run'
    benchmark.prepare(artifacts, run)
    for i in range(2):
        benchmark.append_record(run / 'predictions.jsonl', {
            'molecule_id': identities[i], 'arm': 'classifier', 'status': 'ok',
            'scores': {enzyme: float(i) for enzyme in ENZYMES},
        })
    return artifacts, run


def test_decomposition_accepts_unchanged_frozen_run_and_sparse_labels(frozen_run):
    artifacts, run = frozen_run
    cohort, metrics, contrasts = similarity_bands.decompose(run, artifacts, bootstrap=0)
    assert len(cohort) == 2
    assert cohort.novelty.between(0, 1).all()
    assert contrasts.empty
    whole = metrics.loc[metrics.band == 'all']
    sparse = whole.loc[whole.enzyme == ENZYMES[0]]
    assert sparse.n_evaluable.eq(1).all()
    assert sparse.undefined_reason.eq('requires_both_classes').all()


@pytest.mark.parametrize('frozen_run', [('123', '124')], indirect=True)
def test_decomposition_preserves_numeric_text_identities(frozen_run):
    artifacts, run = frozen_run
    cohort, metrics, _ = similarity_bands.decompose(run, artifacts, bootstrap=0)
    assert set(cohort.molecule_id.astype(str)) == {'123', '124'}
    # At least the fully observed endpoints keep both predictions in every band.
    assert metrics.loc[metrics.band == 'all', 'n_evaluable'].max() == 2


@pytest.mark.parametrize('changed', ['manifest', 'cohort'])
def test_decomposition_refuses_changed_frozen_inputs(frozen_run, changed):
    artifacts, run = frozen_run
    if changed == 'manifest':
        path = artifacts / 'manifest.csv'
        pd.DataFrame([{'smiles': 'CCO', 'split': 'train'},
                      {'smiles': 'CCN', 'split': 'train'}]).to_csv(path, index=False)
        message = 'Trained artifacts changed'
    else:
        path = run / 'cohort.csv'
        frame = pd.read_csv(path, keep_default_na=False)
        frame.loc[0, ENZYMES[-1]] = 1
        frame.to_csv(path, index=False)
        message = 'Frozen cohort changed'
    predictions_before = (run / 'predictions.jsonl').read_bytes()
    with pytest.raises(ValueError, match=message):
        similarity_bands.decompose(run, artifacts, bootstrap=0)
    assert (run / 'predictions.jsonl').read_bytes() == predictions_before


@pytest.mark.parametrize('present_band', ['<0.4', '0.4-0.6', '>=0.6'])
def test_decomposition_figure_accepts_empty_similarity_bands(tmp_path, monkeypatch, present_band):
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path / 'matplotlib'))
    from src.research import figures

    data = tmp_path / 'metrics.csv'
    output = tmp_path / 'figure.png'
    pd.DataFrame([
        {'enzyme': 'CYP3A4', 'arm': 'classifier', 'metric': 'auc', 'band': band,
         'value': 0.8, 'ci_low': 0.75, 'ci_high': 0.9}
        for band in ('all', present_band)
    ]).to_csv(data, index=False)
    try:
        assert figures.decomposition(data, output) == output
        assert output.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    finally:
        figures.plt.close('all')


def test_endpoint_subset_is_shared_by_evaluation_and_llm_prompt():
    program = textwrap.dedent('''
        import pandas as pd
        from src import benchmark
        from src.chemistry import ENZYMES
        from src.evaluate import evaluate

        assert ENZYMES == ('CYP3A4',)
        cohort = pd.DataFrame({'molecule_id': ['m0', 'm1'], 'scaffold': ['s0', 's1'],
                               'CYP3A4': [0, 1]})
        predictions = [{'molecule_id': f'm{i}', 'arm': 'classifier', 'status': 'ok',
                        'scores': {'CYP3A4': float(i)}} for i in range(2)]
        metrics, contrasts, _ = evaluate(cohort, predictions, bootstrap=0)
        assert set(metrics.enzyme) == {'CYP3A4'}
        assert metrics.value.eq(1.0).all()
        assert contrasts.empty
        assert benchmark.SCORE_SCHEMA['schema']['required'] == ['CYP3A4']
        assert 'CYP3A4' in benchmark.LLM_PROMPT
        assert 'CYP1A2' not in benchmark.LLM_PROMPT
    ''')
    subprocess.run([sys.executable, '-c', program], cwd=ROOT,
                   env={**os.environ, 'ENDPOINTS': 'CYP3A4'},
                   capture_output=True, text=True, check=True)


def test_decomposition_figure_refuses_to_silently_choose_an_arm(tmp_path, monkeypatch):
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path / 'matplotlib'))
    from src.research import figures

    path = tmp_path / 'metrics.csv'
    pd.DataFrame([{'arm': arm, 'metric': 'auc', 'enzyme': 'CYP3A4', 'band': 'all',
                   'value': value, 'ci_low': value - .05, 'ci_high': value + .05}
                  for arm, value in [('classifier', .9), ('smiles', .7)]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match='one prediction arm'):
        figures.decomposition(path, tmp_path / 'figure.png')
