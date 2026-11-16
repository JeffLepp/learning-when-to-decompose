"""Lossless artifact packaging and presentation QA; no fit, selection or evaluation."""
import hashlib
import json
import lzma
import re
import zlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'results/v2/router'
DEST = BASE / 'task_e'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


changes = []
for name, directory in [('v1', BASE / 'p4-v1adapter'), ('synthetic', BASE / 'synthetic16/p4')]:
    report_path = directory / 'report.json'
    original_report = report_path.read_bytes()
    report = json.loads(original_report)
    artifact = directory / 'frozen/objects.joblib'
    original = artifact.read_bytes()
    assert digest(original) == report['frozen_sha256'][artifact.name]
    payload = zlib.decompress(original)
    packed = lzma.compress(payload, preset=9)
    assert lzma.decompress(packed) == payload
    # Original compressed bytes are also exactly reconstructible. No estimator
    # is unpickled or mutated during this byte-stream transcode.
    assert zlib.compress(payload, level=3) == original
    if len(packed) < len(original):
        archive = DEST / f'{name}_p4_report_before_packaging.json.xz'
        archive.write_bytes(lzma.compress(original_report))
        artifact.write_bytes(packed)
        change = dict(name=name, before_bytes=len(original), after_bytes=len(packed),
                      original_container_sha256=digest(original), packed_container_sha256=digest(packed),
                      identical_serialized_payload_sha256=digest(payload),
                      original_p4_report_sha256=digest(original_report), original_report_archive=str(archive.relative_to(ROOT)),
                      reversible='Original container = zlib.compress(lzma.decompress(objects.joblib), level=3)')
        report['frozen_sha256'][artifact.name] = digest(packed)
        report['artifact_packaging'] = change
        write(report_path, report)
        for split in ['holdout', 'ood_length']:
            final_path = directory / f'final_{split}/report.json'
            if final_path.exists():
                final = json.loads(final_path.read_text(encoding='utf-8'))
                # Preserve the original evaluation-time provenance. That exact
                # P4 report is recoverable from the archived .json.xz above.
                assert final['p4_report_sha256'] == digest(original_report)
                final['packaged_p4_report_sha256'] = digest(report_path.read_bytes())
                final['original_p4_report_archive'] = str(archive.relative_to(ROOT))
                write(final_path, final)
        changes.append(change)
    # Presentation-only figure rerender from recorded aggregate points.
    if len(report['fixed_points']) > 4:
        fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
        for i, p in enumerate(report['fixed_points']):
            ax.scatter(p['mean_cost'], p['accuracy'], s=35, marker=['o', 's', '^', 'D'][i % 4],
                       label=f"{i}: {p['arm'].replace('qwen3:', '')}")
        for label, points in [('Fixed frontier', report['fixed_frontier'])] + [
                (k.replace('selected/', ''), p['curve']) for k, p in report['policies'].items()
                if k.startswith('selected/') and k != 'selected/overall']:
            ordered = sorted(points, key=lambda p: (p['mean_cost'], -p['accuracy']))
            frontier, best = [], -np.inf
            for p in ordered:
                if p['accuracy'] > best:
                    frontier.append(p)
                    best = p['accuracy']
            ax.plot([p['mean_cost'] for p in frontier], [p['accuracy'] for p in frontier], label=label)
        oracle = report['oracle']
        ax.scatter(oracle['mean_cost'], oracle['accuracy'], marker='*', s=120, color='black', label='Outcome oracle')
        ax.set(title='Validation (used for selection; optimistic)', xlabel='Mean total tokens per item',
               ylabel='Accuracy', ylim=(0, 1.02))
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
        ax.grid(alpha=.2)
        fig.savefig(directory / 'figures/cost_accuracy.png', dpi=150)
        plt.close(fig)
        metric = report['prediction_metrics'][report['selected_router'].removeprefix('route/')]['crossfit']
        fig, axes = plt.subplots(4, 4, figsize=(12, 10), layout='constrained', sharex=True, sharey=True)
        for ax, p, bins in zip(axes.flat, report['fixed_points'], metric['reliability_bins']):
            bins = [b for b in bins if b['n']]
            ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
            ax.plot([b['confidence'] for b in bins], [b['accuracy'] for b in bins], 'o-', markersize=3)
            ax.set(title=p['arm'].replace('qwen3:', ''), xlim=(0, 1), ylim=(0, 1))
        fig.supxlabel('Cross-fitted predicted success probability')
        fig.supylabel('Observed success fraction')
        fig.suptitle('Validation (used for selection; optimistic): selected learner, each arm')
        fig.savefig(directory / 'figures/reliability.png', dpi=150)
        plt.close(fig)
        grid = report['prediction_metrics']
        matrix = np.array([[np.nan if x is None else x for x in m['crossfit']['auroc']] for m in grid.values()])
        fig, ax = plt.subplots(figsize=(13, 10), layout='constrained')
        color = ax.imshow(matrix, aspect='auto', vmin=0, vmax=1, cmap='cividis')
        ax.set_yticks(range(len(grid)), list(grid), fontsize=8)
        ax.set_xticks(range(16), [p['arm'].replace('qwen3:', '') for p in report['fixed_points']], rotation=60, ha='right', fontsize=8)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f'{value:.2f}' if np.isfinite(value) else 'NA', ha='center', va='center', fontsize=7,
                    color='black' if value > .5 else 'white')
        ax.set(title='Validation (used for selection; optimistic): cross-fitted AUROC')
        fig.colorbar(color, ax=ax, label='AUROC')
        fig.savefig(directory / 'figures/auroc.png', dpi=150)
        plt.close(fig)

summary_paths = [BASE / 'p4-v1adapter/summary.md', BASE / 'synthetic16/p4/summary.md',
                 BASE / 'synthetic16/p4/final_holdout/summary.md', BASE / 'synthetic16/p4/final_ood_length/summary.md',
                 BASE / 'mode_a_scan/summary.md', BASE / 'mode_a_chainops/summary.md']
for path in summary_paths:
    text = path.read_text(encoding='utf-8')
    text = re.sub(r'(qwen3:[0-9.]+b)\|', r'\1\\|', text)
    path.write_text(text, encoding='utf-8')
    assert len(text.splitlines()) < 150
write(DEST / 'packaging.json', dict(lossless_payloads_verified=True, changes=changes,
      presentation='Escaped Markdown arm separators; rerendered crowded synthetic cost, reliability and AUROC figures from recorded metrics.',
      script_sha256=digest(Path(__file__).read_bytes())))
print(json.dumps(changes, indent=2))
