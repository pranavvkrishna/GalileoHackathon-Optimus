"""Verify the matched experiment or launch B with a complete local audit trail."""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ('HISTORY', 'HORIZON', 'NY', 'NX', 'WIDTH', 'EPOCHS', 'BATCH_SIZE',
            'LEARNING_RATE', 'WEIGHT_DECAY', 'VORT_SCALE', 'WINDOW_SUBSAMPLE', 'KNN')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--data-dir', type=Path, required=True)
    cli.add_argument('--output-dir', type=Path)
    cli.add_argument('--seed', type=int, default=0)
    cli.add_argument('--train', action='store_true', help='launch B; default is preflight only')
    args = cli.parse_args()
    data = args.data_dir.resolve()
    package = ROOT / 'variants/B/pitching_airfoil_wake'
    a = load(ROOT / 'variants/A/pitching_airfoil_wake/baseline/run_baseline.py', 'control')
    b = load(package / 'baseline/run_baseline.py', 'intervention')
    settings = {key: getattr(b, key) for key in SETTINGS}
    assert all(getattr(a, key) == value for key, value in settings.items()), 'A/B settings differ'
    source = json.loads((package / 'data_source.json').read_text())
    for item in source['files']:
        path = data / item['name']
        assert path.stat().st_size == item['size'], f'Incomplete file: {path}'
        md5 = hashlib.md5(usedforsecurity=False)
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                md5.update(chunk)
        assert md5.hexdigest() == item['md5'], f'Checksum mismatch: {path}'
    counts = {}
    for name in ('train', 'validation'):
        with np.load(data / (name + '.npz'), allow_pickle=False) as f:
            counts[name] = {'cases': len(f['case_id']), 'declared_windows': len(f['win_start'])}
    report = {'status': 'preflight_passed', 'settings': settings, 'data_counts': counts,
              'seed': args.seed, 'augmentation_seed': args.seed + 1_000_003,
              'augmentation_probability': 0.5, 'python': platform.python_version(),
              'torch': torch.__version__, 'numpy': np.__version__,
              'cuda_available': torch.cuda.is_available(),
              'grid_audit': 'Organizer permits reflection; exact sampled coordinates unconfirmed',
              'source_sha256': digest(package / 'baseline/run_baseline.py')}
    print(json.dumps(report, indent=2), flush=True)
    if not args.train:
        return
    if not torch.cuda.is_available():
        raise SystemExit('Training requires a CUDA GPU for this launcher; preflight completed.')
    if args.output_dir is None:
        cli.error('--train requires --output-dir pointing to a fresh directory')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'source/pitching_airfoil_wake'
    shutil.copytree(package, snapshot, ignore=shutil.ignore_patterns('*.pt', '__pycache__'))
    shutil.copy2(ROOT / 'variants/A/pitching_airfoil_wake/baseline/run_baseline.py', output / 'source/control.py')
    script = snapshot / 'baseline/run_baseline.py'
    initial = output / 'initial_model.pt'
    with (output / 'train.log').open('w') as log:
        subprocess.run([sys.executable, '-u', str(script), 'init', '--seed', str(args.seed),
                        '--output', str(initial)], stdout=log, stderr=subprocess.STDOUT, check=True)
        command = [sys.executable, '-u', str(script), 'train', '--train-only', '--mirror',
                   '--data_dir', str(data), '--seed', str(args.seed), '--init_model', str(initial),
                   '--output_dir', str(output)]
        report.update(status='training_started', command=command, initial_sha256=digest(initial))
        (output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    report.update(status='training_complete' if result.returncode == 0 else 'training_failed',
                  returncode=result.returncode)
    if result.returncode == 0:
        report['final_sha256'] = digest(output / 'final_model.pt')
    (output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    packages = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
    (output / 'environment.txt').write_text('\n'.join(packages) + '\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
