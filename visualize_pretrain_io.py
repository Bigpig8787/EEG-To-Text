"""Visualize pre-training reconstruction: input vs recon, MSE curves over epochs.

Run from the project root that contains pretrain_io_samples/.
Outputs: pretrain_io_samples/_viz_mse_curve.png
         pretrain_io_samples/_viz_waveform_compare.png
"""
import os
import re
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

SAMPLES_DIR = 'pretrain_io_samples'
N_CHANNELS_TO_PLOT = 6
TIME_WINDOW = (0, 2000)


def load_all():
    files = glob.glob(os.path.join(SAMPLES_DIR, 'epoch_*.pt'))
    items = []
    for f in files:
        m = re.search(r'epoch_(-?\d+)\.pt$', f)
        if not m:
            continue
        ep = int(m.group(1))
        d = torch.load(f, map_location='cpu')
        items.append((ep, d))
    items.sort(key=lambda x: x[0])
    return items


def plot_mse_curve(items, out):
    eps = [ep for ep, _ in items]
    mse_all = [float(d['mse_overall']) for _, d in items]
    mse_msk = [float(d['mse_masked']) for _, d in items]
    plt.figure(figsize=(8, 4.5))
    plt.plot(eps, mse_all, label='mse_overall', marker='.')
    plt.plot(eps, mse_msk, label='mse_masked', marker='.')
    plt.axvline(-1, color='gray', ls='--', alpha=0.5, label='baseline (untrained)')
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.title('Reconstruction MSE over epochs')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f'saved {out}')


def plot_waveforms(items, out):
    ep_first, d_first = items[0]
    ep_last, d_last = items[-1]
    inp = d_first['input'].squeeze().cpu().numpy()
    rec0 = d_first['recon'].squeeze().cpu().numpy()
    recN = d_last['recon'].squeeze().cpu().numpy()
    msk = d_first['mask'].squeeze().cpu().numpy().astype(bool)

    C = inp.shape[0]
    chans = np.linspace(0, C - 1, N_CHANNELS_TO_PLOT, dtype=int)
    t0, t1 = TIME_WINDOW

    fig, axes = plt.subplots(N_CHANNELS_TO_PLOT, 1, figsize=(11, 1.6 * N_CHANNELS_TO_PLOT), sharex=True)
    for ax, ch in zip(axes, chans):
        ax.plot(inp[ch, t0:t1], color='black', lw=0.8, label='input')
        ax.plot(rec0[ch, t0:t1], color='tab:orange', lw=0.8, alpha=0.8, label=f'recon ep{ep_first}')
        ax.plot(recN[ch, t0:t1], color='tab:blue', lw=0.8, alpha=0.9, label=f'recon ep{ep_last}')
        m = msk[ch, t0:t1]
        if m.any():
            in_seg = False
            for i, v in enumerate(m):
                if v and not in_seg:
                    s = i; in_seg = True
                elif not v and in_seg:
                    ax.axvspan(s, i, color='red', alpha=0.12)
                    in_seg = False
            if in_seg:
                ax.axvspan(s, len(m), color='red', alpha=0.12)
        ax.set_ylabel(f'ch{ch}')
        ax.grid(alpha=0.3)
    axes[0].legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel(f'time step ({t0}-{t1}); red shading = masked region')
    fig.suptitle(f'Input vs Reconstruction (baseline ep{ep_first} -> final ep{ep_last})')
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f'saved {out}')


def main():
    items = load_all()
    if not items:
        print(f'no .pt files in {SAMPLES_DIR}/')
        return
    print(f'loaded {len(items)} epochs: {items[0][0]} .. {items[-1][0]}')
    plot_mse_curve(items, os.path.join(SAMPLES_DIR, '_viz_mse_curve.png'))
    plot_waveforms(items, os.path.join(SAMPLES_DIR, '_viz_waveform_compare.png'))


if __name__ == '__main__':
    main()
