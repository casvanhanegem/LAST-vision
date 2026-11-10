"""Export per-run and seed-summary CSVs plus an accuracy/spike plot.

Synthetic and limited-data runs are excluded. Settings are grouped strictly;
different training protocols never get averaged into one result.
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ['accuracy', 'input_spikes_per_sample', 'input_spike_density',
           'hidden_spikes_per_sample', 'output_spikes_per_sample', 'total_spikes_per_sample',
           'synaptic_events_proxy_per_sample', 'encoder_dense_weight_macs_per_sample',
           'classifier_dense_weight_macs_per_sample', 'no_output_spike_fraction',
           'membrane_readout_accuracy']


def summarize(root, output):
    rows, groups = [], defaultdict(list)
    for path in sorted(Path(root).rglob('test_metrics.json')):
        config = json.loads(path.with_name('config.json').read_text())
        if config['fake_data'] or config.get('limit'):
            continue
        metrics = json.loads(path.read_text())
        row = {**config, **{m: metrics[m] for m in METRICS},
               'amortized_ms_per_sample': metrics['timing']['median_amortized_ms_per_sample'],
               'run': str(path.parent)}
        rows.append(row)
        excluded = {'seed', 'data_dir', 'download', 'fake_data', 'limit',
                    'encoder_checkpoint', 'encoder_checkpoint_sha256'}
        protocol = {k: v for k, v in config.items() if k not in excluded}
        environment = json.loads(path.with_name('environment.json').read_text())
        key = json.dumps({'protocol': protocol, 'environment': environment}, sort_keys=True)
        groups[key].append(row)
    if not rows:
        raise SystemExit('No complete full-data experiments found (fake/limited runs are excluded).')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    def write_csv(name, items):
        columns = sorted(set().union(*(item.keys() for item in items)))
        with (output / name).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(items)
    write_csv('per_run.csv', rows)
    summaries = []
    for key, group in groups.items():
        seeds = [r['seed'] for r in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f'Duplicate seed in same protocol: {seeds}')
        protocol = json.loads(key)['protocol']
        result = {**protocol, 'n_seeds': len(group), 'seeds': json.dumps(seeds),
                  'environment': json.dumps(json.loads(key)['environment'], sort_keys=True)}
        for metric in METRICS + ['amortized_ms_per_sample']:
            values = [r[metric] for r in group]
            result[metric + '_mean'] = statistics.mean(values)
            result[metric + '_sd'] = statistics.stdev(values) if len(values) > 1 else None
        summaries.append(result)
    write_csv('summary.csv', summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    datasets = sorted({r['dataset'] for r in summaries})
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5), squeeze=False)
    colors = {'rate': '#2563eb', 'last_fixed': '#64748b', 'last': '#d97706'}
    for ax, dataset in zip(axes[0], datasets):
        seen = set()
        for row in summaries:
            if row['dataset'] != dataset:
                continue
            name = row['encoder']
            label = name + ' / ' + row.get('readout', 'paper')
            ax.errorbar(row['input_spikes_per_sample_mean'], 100 * row['accuracy_mean'],
                        xerr=row['input_spikes_per_sample_sd'],
                        yerr=100 * (row['accuracy_sd'] or 0), fmt='o', capsize=3,
                        color=colors[name], label=label if label not in seen else None)
            ax.annotate(f"T={row['steps']}", (row['input_spikes_per_sample_mean'], 100 * row['accuracy_mean']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
            seen.add(label)
        ax.set(xlabel='Input spikes per sample', ylabel='Test accuracy (%)', title=dataset.upper())
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'accuracy_vs_input_spikes.png', dpi=200)
    plt.close(fig)
    print(f'Saved {len(rows)} runs and {len(summaries)} summaries to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='runs/main')
    parser.add_argument('--out', default='analysis')
    args = parser.parse_args()
    summarize(args.root, args.out)
