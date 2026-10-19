"""Export selected small Phase 1/2 artifacts without copying model weights.

Standard-library only. ZIP entries start with results/phase1 or results/phase2.
Existing ZIPs are never overwritten and Drive source files are never modified.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import re
import zipfile


COMMON = (
    'run_config.json', 'model_metadata.json', 'metrics.json', 'history.json',
    'verification.json', 'environment.json', 'training_curves.png',
)
FILES = {
    1: COMMON + ('loss_curve.png', 'config.json', 'training_config.json',
                 'evaluation_metrics.json'),
    2: COMMON + ('best_validation_metrics.json', 'post_training_validation.json',
                 'decoder_config.json', 'decoding_candidates.json',
                 'loss_class_weights.json', 'per_label_metrics.csv'),
}
TOKENS = re.compile(r'hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}')


def export_phases(drive_root, output, phases=(1, 2)):
    """Discover immediate run folders and export only allowlisted artifacts.

    Phase 1 discovers run_* folders, including those with only a loss plot.
    Phase 2 discovers folders with run_config.json or metrics.json. Root summary
    CSVs and coverage JSONs are included; Phase 2 labels.json is included too.
    Missing sources/runs are recorded in the manifest, not treated as results.
    """
    drive_root = Path(drive_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not drive_root.is_dir():
        raise ValueError(f'Drive root is missing: {drive_root}')
    if output.exists():
        raise FileExistsError(f'Choose a new output filename: {output}')
    phases = tuple(dict.fromkeys(phases))
    if not phases or any(p not in FILES for p in phases):
        raise ValueError('Select phase 1, phase 2, or both.')
    manifest = {'exported_at_utc': datetime.now(timezone.utc).isoformat(),
                'phases': {}, 'notes': 'Saved artifacts only. No training, checkpoint loading, or Codabench verification performed.'}
    entries = []

    def collect(file, source, target):
        if file.is_symlink() or not file.resolve().is_relative_to(source):
            raise ValueError(f'Refusing linked file: {file.name}')
        if file.stat().st_size > 25 * 1024 * 1024:
            raise ValueError(f'File exceeds 25 MiB: {file.name}')
        data = file.read_bytes()
        if file.suffix in ('.json', '.csv'):
            text = data.decode('utf-8-sig')
            if TOKENS.search(text):
                raise ValueError(f'Possible credential in {file.name}; value hidden.')
            if file.suffix == '.json':
                json.loads(text)
            data = text.replace('\r\n', '\n').encode('utf-8')
        if file.suffix == '.png' and not data.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError(f'Invalid PNG: {file.name}')
        entries.append((target, data))

    for phase in phases:
        source = drive_root / f'cs781_phase{phase}'
        info = {'source_folder': source.name, 'status': 'missing_source', 'runs': [], 'phase_files': []}
        manifest['phases'][str(phase)] = info
        if not source.is_dir():
            print(f'Phase {phase}: source folder missing; listed in manifest.')
            continue
        if source.is_symlink():
            raise ValueError(f'Refusing linked source folder: {source.name}')
        folders = sorted(p for p in source.iterdir() if p.is_dir() and not p.is_symlink() and (
            (phase == 1 and p.name.startswith('run_'))
            or (p / 'run_config.json').is_file() or (p / 'metrics.json').is_file()))
        info['status'] = 'runs_found' if folders else 'no_runs_found'
        for folder in folders:
            included = []
            for name in FILES[phase]:
                file = folder / name
                if file.is_file():
                    collect(file, source, f'results/phase{phase}/{folder.name}/{name}')
                    included.append(name)
            run = {'folder': folder.name, 'included': included,
                   'missing_artifacts': [name for name in FILES[phase] if name not in included],
                   'note': 'Availability does not establish run completion or submission acceptance.'}
            info['runs'].append(run)
            print(f'Phase {phase} / {folder.name}: {len(included)} artifact files')
        for file in sorted(source.iterdir()):
            if file.is_file() and (
                (file.name.startswith('experiment_summary_') and file.suffix == '.csv')
                or (file.name.startswith('coverage_') and file.suffix == '.json')
                or (phase == 2 and file.name == 'labels.json')):
                collect(file, source, f'results/phase{phase}/{file.name}')
                info['phase_files'].append(file.name)
    if not entries:
        raise ValueError('No selected artifacts found. Check the Drive paths and folder contents.')
    # One manifest per phase keeps each phase import self-contained.
    for phase in phases:
        info = manifest['phases'][str(phase)]
        payload = {'exported_at_utc': manifest['exported_at_utc'], **info,
                   'notes': manifest['notes']}
        entries.append((f'results/phase{phase}/export_manifest.json',
                        (json.dumps(payload, indent=2) + '\n').encode()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for target, data in entries:
            archive.writestr(target, data)
    print(f'Created {output} ({output.stat().st_size / 1024:.1f} KiB). Source files unchanged.')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive-root', required=True, help='MyDrive or local parent of cs781_phase1 and cs781_phase2')
    parser.add_argument('--output', required=True, help='New ZIP path')
    parser.add_argument('--phases', nargs='+', type=int, choices=[1, 2], default=[1, 2])
    args = parser.parse_args()
    export_phases(args.drive_root, args.output, args.phases)
