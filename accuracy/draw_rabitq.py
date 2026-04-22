import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'cm'

colors = {1: '#5bc0de', 2: '#337ab7', 3: '#e08a3c', 4: '#2d6a4f'}
rows = [("prod", r"$\mathrm{RaBitQ}_{\mathrm{prod}}$"),
        ("mse",   r"$\mathrm{RaBitQ}_{\mathrm{mse}}$")]
bits_list = [1, 2, 3, 4]

fig, axes = plt.subplots(2, 4, figsize=(16, 7))

for i, (qtype, label) in enumerate(rows):
    for j, b in enumerate(bits_list):
        ax = axes[i, j]
        err = pd.read_csv(f"./rabitq_ip_errors_{b}bit_{qtype}.csv").values.flatten()
        err = err[~np.isnan(err)]
        ax.hist(err, bins=50, alpha=0.85, edgecolor="black", linewidth=0.3, color=colors[b])
        ax.set_xlabel("Inner Product Distortion", fontsize=10)
        if j == 0:
            ax.set_ylabel("Frequency", fontsize=11)
        else:
            ax.set_ylabel("")
        ax.set_title(f"Bitwidth = {b}", fontsize=12, fontweight='bold')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.set_xlim(-0.15, 0.15)
        ax.text(0.95, 0.95, f"mean={err.mean():.4f}\nstd={err.std():.4f}\nmax={err.max():.4f}",
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    # Row label
    fig.text(0.02, 0.75 - i * 0.47, f"({'ab'[i]}) {label}", fontsize=14,
             fontweight='bold', va='center', rotation=90)

plt.tight_layout(rect=[0.03, 0.02, 1, 0.98])
plt.subplots_adjust(hspace=0.35, wspace=0.25)


plt.savefig("rabitq_ip_err.png", bbox_inches='tight', dpi=300)