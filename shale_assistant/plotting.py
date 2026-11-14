"""One-page log plot: inputs, QC flags, computed curves, facies and zone boundaries.

Looking at this plot is the first check before trusting any number.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def log_plot(df: pd.DataFrame, summary: dict, path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    depth = df["DEPTH"].to_numpy()
    unit = summary["well"]["depth_unit"].lower()
    has_facies = "FACIES" in df.columns and df["FACIES"].notna().any()
    n_tracks = 9 if has_facies else 8
    height = float(np.clip((depth[-1] - depth[0]) / 55.0, 8.0, 26.0))
    fig, axes = plt.subplots(1, n_tracks, figsize=(2.05 * n_tracks, height), sharey=True)

    def col(name):
        return df[name].to_numpy(dtype=float) if name in df.columns else np.full(len(df), np.nan)

    # 1 gamma ray + caliper
    ax = axes[0]
    ax.plot(col("GR"), depth, color="green", lw=0.6)
    ax.set_xlim(0, max(150.0, float(np.nanpercentile(col("GR"), 99.5))))
    ax.set_xlabel("GR [API]", color="green")
    ax.set_ylabel(f"Measured depth [{unit}]")
    if np.isfinite(col("CALI")).any():
        twin = ax.twiny()
        twin.spines['top'].set_position(('outward', 30))
        twin.plot(col("CALI"), depth, color="black", lw=0.5, ls="--")
        twin.set_xlim(6, 22)
        twin.set_xlabel("CALI [in] (dashed)")
    bad = df["QC_BADHOLE"].to_numpy(dtype=bool)
    ax.fill_betweenx(depth, 0, 1, where=bad, color="red", alpha=0.25, transform=ax.get_yaxis_transform(), lw=0)

    # 2 resistivity
    ax = axes[1]
    ax.semilogx(col("RT"), depth, color="red", lw=0.6)
    ax.set_xlim(0.2, 2000)
    ax.set_xlabel("RT [ohm.m]", color="red")

    # 3 density / neutron
    ax = axes[2]
    ax.plot(col("RHOB"), depth, color="firebrick", lw=0.6)
    ax.set_xlim(1.95, 2.95)
    ax.set_xlabel("RHOB [g/cc]", color="firebrick")
    twin = ax.twiny()
    twin.spines['top'].set_position(('outward', 30))
    twin.plot(col("NPHI"), depth, color="blue", lw=0.6)
    twin.set_xlim(0.45, -0.15)
    twin.set_xlabel("NPHI [v/v]", color="blue")

    # 4 sonic + delta log R
    ax = axes[3]
    ax.plot(col("DT"), depth, color="purple", lw=0.6)
    ax.set_xlim(140, 40)
    ax.set_xlabel("DT [us/ft]", color="purple")
    twin = ax.twiny()
    twin.spines['top'].set_position(('outward', 30))
    dlogr = col("DLOGR")
    twin.plot(dlogr, depth, color="black", lw=0.5)
    twin.fill_betweenx(depth, 0, dlogr, where=dlogr > 0, color="gold", alpha=0.6, lw=0)
    twin.set_xlim(-2, 2)
    twin.set_xlabel("delta log R")

    # 5 Vsh
    ax = axes[4]
    ax.plot(col("VSH_LINEAR"), depth, color="grey", lw=0.5, label="linear")
    ax.plot(col("VSH_LARIONOV"), depth, color="saddlebrown", lw=0.7, label="Larionov")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Vsh [v/v]")
    ax.legend(loc="lower right", fontsize=6)

    # 6 porosity
    ax = axes[5]
    ax.plot(col("PHIT"), depth, color="black", lw=0.6, label="PHIT")
    ax.plot(col("PHIE"), depth, color="deepskyblue", lw=0.6, label="PHIE")
    ax.set_xlim(0.4, 0)
    ax.set_xlabel("Porosity [v/v]")
    ax.legend(loc="lower left", fontsize=6)

    # 7 Sw
    ax = axes[6]
    ax.plot(col("SW_ARCHIE"), depth, color="navy", lw=0.6, label="Archie")
    ax.plot(col("SW_SIMANDOUX"), depth, color="darkorange", lw=0.6, label="Simandoux")
    ax.set_xlim(1, 0)
    ax.set_xlabel("Sw [v/v]")
    ax.legend(loc="lower left", fontsize=6)

    # 8 TOC
    ax = axes[7]
    toc = col("TOC")
    ax.plot(toc, depth, color="black", lw=0.6)
    ax.fill_betweenx(depth, 0, toc, where=toc > 0, color="dimgray", alpha=0.7, lw=0)
    ax.set_xlim(0, max(10.0, float(np.nanmax(toc)) if np.isfinite(toc).any() else 10.0))
    ax.set_xlabel("TOC [wt%]")

    # 9 facies
    if has_facies:
        ax = axes[8]
        classes = sorted(df["FACIES"].dropna().unique())
        cmap = plt.get_cmap("tab20")
        colors = {c: cmap(i % 20) for i, c in enumerate(classes)}
        fac = df["FACIES"].to_numpy(dtype=object)
        for c in classes:
            ax.fill_betweenx(depth, 0, 1, where=(fac == c), color=colors[c], lw=0, step="mid")
        ax.plot(col("FACIES_CONF"), depth, color="black", lw=0.4)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Facies + confidence")
        fig.legend(handles=[Patch(color=colors[c], label=c) for c in classes], loc="lower center",
                   ncol=min(len(classes), 6), fontsize=7, frameon=False)

    for ax in axes:
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.grid(True, lw=0.2, alpha=0.6)
        ax.tick_params(labelsize=7)
        for z in summary["zones"]:
            ax.axhline(z["top"], color="black", lw=0.5, alpha=0.7)
    for z in summary["zones"]:
        label = z["name"] + (" *" if z["low_confidence"] else "")
        axes[0].text(0.02, (z["top"] + z["base"]) / 2, label, transform=axes[0].get_yaxis_transform(),
                     fontsize=7, va="center", ha="left", fontweight="bold")
    axes[0].set_ylim(depth[-1], depth[0])
    fig.suptitle(f"{summary['well']['name']}  -  red shading = bad hole, * = low-confidence zone", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0.03 if has_facies else 0, 1, 0.985))
    fig.savefig(path, dpi=130)
    plt.close(fig)
