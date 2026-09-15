"""Replace workstation/server absolute paths in result JSON with portable paths."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SUFFIXES = {'.pth', '.pt', '.ckpt'}


def _portable_path(value):
    normalized = value.replace('\\', '/')
    current_root = PROJECT_ROOT.as_posix()
    if normalized.casefold().startswith(current_root.casefold() + '/'):
        return normalized[len(current_root) + 1:], 'current_project', True

    sources = (
        ('server_results_high_yield', 'server_high_yield'),
        ('reproduction_results', 'local'),
        ('server_results', 'server'),
    )
    for marker, source in sources:
        match = re.search(rf'(?:^|/){re.escape(marker)}/(.+)$', normalized)
        if match is None:
            continue
        suffix = Path(match.group(1))
        if source == 'local':
            base = (
                PROJECT_ROOT / 'checkpoints' / 'imported' / 'local'
                if suffix.suffix.lower() in CHECKPOINT_SUFFIXES
                else PROJECT_ROOT / 'results' / 'runs' / 'local'
            )
        elif source == 'server':
            base = (
                PROJECT_ROOT / 'checkpoints' / 'imported' / 'server'
                if suffix.suffix.lower() in CHECKPOINT_SUFFIXES
                else PROJECT_ROOT / 'results' / 'runs' / 'server'
            )
        else:
            base = PROJECT_ROOT / 'external-artifacts' / source
        target = base / suffix
        available = target.exists()
        if not available and source != 'server_high_yield':
            target = PROJECT_ROOT / 'external-artifacts' / source / suffix
        return target.relative_to(PROJECT_ROOT).as_posix(), source, available
    return value, 'unmapped', False


def _rewrite(value, counts):
    if isinstance(value, dict):
        rewritten = {}
        for key, item in value.items():
            if key == 'independent_ensemble_role':
                key = 'full_fusion_comparison_role'
                item = (
                    'complete iMOE + iMOE_CSR prediction-level fusion comparison; '
                    'excluded from DRC-iMOE selection and not a mathematical bound'
                )
                counts['terminology:rewritten'] += 1
            rewritten[key] = _rewrite(item, counts)
        return rewritten
    if isinstance(value, list):
        return [_rewrite(item, counts) for item in value]
    if not isinstance(value, str):
        return value
    if 'TPSL_upper_bound' in value:
        value = value.replace('TPSL_upper_bound', 'TPSL_full_fusion')
        counts['terminology:rewritten'] += 1
    if 'performance upper bound' in value:
        value = (
            'complete iMOE + iMOE_CSR prediction-level fusion comparison; '
            'excluded from DRC-iMOE selection and not a mathematical bound'
        )
        counts['terminology:rewritten'] += 1
    if not (re.match(r'^[A-Za-z]:[\\/]', value) or value.startswith('/')):
        return value
    replacement, source, available = _portable_path(value)
    changed = replacement != value
    counts[f'{source}:rewritten'] += changed
    counts[f'{source}:available'] += changed and available
    counts[f'{source}:external_missing'] += changed and not available
    return replacement


def normalize_results(results_root):
    counts = Counter()
    changed_files = 0
    for path in sorted(Path(results_root).rglob('*.json')):
        try:
            original = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            counts['invalid_json'] += 1
            continue
        rewritten = _rewrite(original, counts)
        if rewritten == original:
            continue
        path.write_text(
            json.dumps(rewritten, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        changed_files += 1
    return {'changed_files': changed_files, **dict(sorted(counts.items()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results-root',
        type=Path,
        default=PROJECT_ROOT / 'results',
    )
    parser.add_argument(
        '--report',
        type=Path,
        default=PROJECT_ROOT / 'results' / 'PATH_MIGRATION.json',
    )
    args = parser.parse_args()
    latest_run = normalize_results(args.results_root)
    history = []
    if args.report.is_file():
        existing = json.loads(args.report.read_text(encoding='utf-8'))
        history.extend(existing.get('history', [existing]))
    history.append(latest_run)
    report = {'latest_run': latest_run, 'history': history}
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
