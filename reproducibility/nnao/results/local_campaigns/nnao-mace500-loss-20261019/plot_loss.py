"""Offline loss and log-energy-change plots from the read-only job-99005 snapshot."""
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json
import os
import re

import numpy as np

OUT = Path(__file__).resolve().parent
SOURCE = OUT/'snapshot'
os.environ.setdefault('MPLCONFIGDIR', str(OUT/'.mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    history = json.loads((SOURCE/'train/history.json').read_text())
    position = json.loads((SOURCE/'checkpoint-position.json').read_text())
    log = (SOURCE/'restart-single-pass-99005.log').read_text()
    assert 'DUE TO TIME LIMIT' in log
    epochs = np.array([row['epoch'] for row in history], dtype=int)
    np.testing.assert_array_equal(epochs, np.arange(597))
    online = [row for row in history if row.get('loss_kind') == 'online_pre_update_mean'
              and row.get('complete_epoch_coverage') is True
              and row.get('sample_count') == 500 and row.get('loss_hartree') is not None]
    e = np.array([row['epoch'] for row in online], dtype=int)
    energy = np.array([row['loss_hartree'] for row in online], dtype=np.float64)
    np.testing.assert_array_equal(e, np.arange(5, 597))
    assert np.isfinite(energy).all() and position['epoch'] == int(e[-1])
    assert position['update_step'] == 5969 and position['batch_cursor'] == 9
    # Check every plotted online epoch against independently parsed stdout.
    records = {int(epoch): float(value) for epoch, value in re.findall(
        r'Epoch (\d+)/1000: online mean E ([-+0-9.eE]+) Ha; samples 500/500;', log)}
    assert set(records) == set(map(int, e))
    np.testing.assert_allclose(energy, [records[int(epoch)] for epoch in e], atol=1e-12, rtol=0)
    gain = np.r_[np.nan, -np.diff(energy)*1000]  # positive means energy decreases.
    magnitude = abs(gain)
    best_index = int(np.argmin(energy))
    best = float(energy[best_index])
    gap = (energy-best)*1000  # empirical reference; not a known variational limit.
    median20 = np.full(len(e), np.nan)
    for i in range(20, len(e)):
        median20[i] = np.median(magnitude[i-19:i+1])
    online_lookup = {int(epoch): i for i, epoch in enumerate(e)}
    exported = []
    for row in history:
        epoch = row['epoch']
        i = online_lookup.get(epoch)
        exported.append(dict(epoch=epoch, update_step=row['update_step'],
            loss_kind=row.get('loss_kind', 'legacy_full_evaluation'),
            sample_count=row.get('sample_count'),
            complete_epoch_coverage=row.get('complete_epoch_coverage'),
            mean_energy_hartree=row['loss_hartree'],
            signed_epoch_energy_decrease_mEh=None if i is None or i == 0 else float(gain[i]),
            absolute_epoch_energy_change_mEh=None if i is None or i == 0 else float(magnitude[i]),
            cumulative_online_decrease_mEh=None if i is None else float((energy[0]-energy[i])*1000),
            gap_to_lowest_recorded_online_mean_mEh=None if i is None else float(gap[i]),
            rolling20_median_abs_change_mEh=None if i is None or not np.isfinite(median20[i]) else float(median20[i])))
    with (OUT/'loss-curve.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
        'axes.labelcolor': '#26313A', 'text.color': '#26313A'})
    blue, orange, grey = '#28699C', '#D87632', '#6C7780'
    boundaries = np.arange(100.5, 597, 100)

    def guides(ax):
        ax.grid(color='#E4E8EB', lw=.65)
        ax.set_axisbelow(True)
        ax.set_xlabel('Epoch')
        for boundary in boundaries:
            if ax.get_xlim()[0] <= boundary <= ax.get_xlim()[1]:
                ax.axvline(boundary, color='#BAC1C6', lw=.8, ls=':')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
    legacy = [row for row in history if row['epoch'] <= 3]
    axes[0].plot([row['epoch'] for row in legacy], [row['loss_hartree'] for row in legacy],
        color=grey, marker='o', markersize=4, lw=1.7, label='Legacy full evaluation (0-3)')
    axes[0].plot(e, energy, color=blue, lw=1.8, label='Online epoch mean (5-596)')
    axes[0].scatter(e[-1], energy[-1], color=blue, s=25, zorder=3)
    axes[0].set_xlim(-4, 604)
    axes[0].set_ylabel('Mean SCF energy per molecule (Eh)')
    axes[0].ticklabel_format(axis='y', style='plain', useOffset=False)
    axes[0].set_title('Recorded training loss', loc='left', fontsize=12)
    axes[0].legend(frameon=False, loc='upper right', fontsize=9)
    axes[0].text(.97, .27, f'Epoch 596 online mean\n{energy[-1]:.10f} Eh / molecule',
                 transform=axes[0].transAxes, ha='right', fontsize=10, color=blue)
    start_epoch = 200
    start_index = int(np.flatnonzero(e == start_epoch)[0])
    tail = e >= start_epoch
    axes[1].plot(e[tail], (energy[tail]-energy[start_index])*1000, color=blue, lw=1.8)
    axes[1].scatter(e[-1], (energy[-1]-energy[start_index])*1000, color=blue, s=25)
    axes[1].set_xlim(195, 604)
    axes[1].set_ylabel(r'$E_k-E_{200}$ (mEh / molecule)')
    axes[1].set_title('Later training: same online loss definition', loc='left', fontsize=12)
    for ax in axes:
        guides(ax)
    fig.suptitle('NNAO: 500-molecule fixed-exponent MACE training', fontsize=14, y=.98)
    fig.text(.08, .025, 'Job 99005; A100 80G, 32 CPU. Adam batch size 50; learning rate halves every 100 epochs (dotted lines).\n'
             'Epoch 4 has no recorded loss. Epoch 597 is partial and excluded. Online means combine different pre-update parameter states.',
             fontsize=8.5, color='#59636B')
    fig.subplots_adjust(left=.08, right=.98, top=.85, bottom=.19, wspace=.28)
    for extension in ['png', 'pdf']:
        fig.savefig(OUT/f'training-loss.{extension}', dpi=200, facecolor='white')
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.3), sharex=True)
    axes[0].plot(e, np.where(gap > 0, gap, np.nan), color=blue, lw=1.8)
    axes[0].set_yscale('log')
    axes[0].set_ylabel(r'$E_k-E_{\mathrm{lowest\ recorded}}$'+'\n(mEh / molecule; log scale)')
    axes[0].set_title('Gap to the lowest observed online epoch mean', loc='left', fontsize=12)
    axes[0].text(.025, .08, f'Reference: epoch {e[best_index]}. Zero gap excluded, without an epsilon floor.\n'
                 'This empirical reference is not a known converged energy or basis-set limit.',
                 transform=axes[0].transAxes, fontsize=9, color='#59636B')
    decreasing = np.isfinite(gain) & (gain > 0)
    increasing = np.isfinite(gain) & (gain < 0)
    axes[1].scatter(e[decreasing], magnitude[decreasing], color=blue, s=9, alpha=.6,
                    label='Energy decreased')
    axes[1].scatter(e[increasing], magnitude[increasing], edgecolor=orange, facecolor='none',
                    marker='^', s=19, alpha=.85, label='Energy increased')
    axes[1].plot(e, np.where(median20 > 0, median20, np.nan), color='#26313A', lw=1.8,
                 label='20-epoch median absolute change')
    axes[1].set_yscale('log')
    axes[1].set_ylabel(r'$|E_k-E_{k-1}|$'+'\n(mEh / molecule; log scale)')
    axes[1].set_title('Magnitude and sign of consecutive-epoch changes', loc='left', fontsize=12)
    axes[1].legend(frameon=False, loc='upper right', fontsize=9)
    axes[1].set_xlim(0, 604)
    for ax in axes:
        guides(ax)
    axes[0].set_xlabel('')
    fig.suptitle('Logarithmic energy trends: complete online epochs 5-596', fontsize=14, y=.98)
    fig.text(.12, .025, 'Only consecutive, complete 500-sample online records are compared. No log of a negative absolute total energy.\n'
             'Individual epoch rises are retained; no monotone envelope or exponential convergence fit is imposed.',
             fontsize=8.5, color='#59636B')
    fig.subplots_adjust(left=.12, right=.98, top=.90, bottom=.13, hspace=.25)
    for extension in ['png', 'pdf']:
        fig.savefig(OUT/f'energy-log-trends.{extension}', dpi=200, facecolor='white')
    plt.close(fig)

    windows = []
    for width in [20, 50, 100]:
        change = float((energy[-width-1]-energy[-1])*1000)
        windows.append(dict(epochs=width, start_epoch=int(e[-width-1]), end_epoch=int(e[-1]),
            net_decrease_mEh_per_molecule=change,
            mean_net_decrease_uEh_per_molecule_per_epoch=change*1000/width,
            median_absolute_change_uEh_per_molecule=float(np.median(magnitude[-width:])*1000)))
    snapshot_hashes = {str(f.relative_to(SOURCE)): hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in sorted(SOURCE.rglob('*')) if f.is_file()}
    summary = dict(captured_at=datetime.now().astimezone().isoformat(), job_id=99005,
        samples=500, batch_size=50, requested_epochs=1000, last_complete_epoch=int(e[-1]),
        current_checkpoint_update=position['update_step'], partial_epoch=597, partial_batches=9,
        online_complete_epochs=len(e), first_online_epoch=int(e[0]),
        first_online_mean_hartree=float(energy[0]), last_online_mean_hartree=float(energy[-1]),
        lowest_recorded_online_epoch=int(e[best_index]), lowest_recorded_online_mean_hartree=best,
        initial_legacy_energy_hartree=history[0]['loss_hartree'],
        initial_legacy_minus_last_online_mEh=float((history[0]['loss_hartree']-energy[-1])*1000),
        cumulative_online_decrease_mEh=float((energy[0]-energy[-1])*1000),
        positive_energy_change_epochs=int(increasing.sum()),
        negative_energy_change_epochs=int(decreasing.sum()),
        zero_change_epochs=int(np.sum(np.isfinite(gain) & (gain == 0))),
        zero_gap_excluded_epochs=e[gap == 0].tolist(), windows=windows,
        log_validation_count=len(records), snapshot_sha256=snapshot_hashes,
        note='Online means are not frozen-checkpoint full-dataset energies; no SCF or training was run.')
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'snapshot_sha256'}, indent=2))


if __name__ == '__main__':
    main()
