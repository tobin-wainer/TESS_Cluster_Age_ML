import matplotlib.pyplot as plt
import numpy as np


def _apply_x_lim(x, pg, x_lim):
    """Slice x and pg to the specified range (index or frequency)."""
    if x_lim is None:
        return x, pg
    mask = (x >= x_lim[0]) & (x <= x_lim[1])
    return x[mask], pg[mask]


def plot_cluster_periodograms(master_table, name, frequency_bins=None, max_num_pgs=10,
                               yscale='log', xscale='linear', x_lim=None, y_lim=None):
    """
    Plot pre-computed periodograms for a cluster, grouped by n_sectors.
    One subplot per unique n_sectors value; each subplot shows up to max_num_pgs lines.

    Parameters
    ----------
    master_table : pd.DataFrame
    name : str, bytes, or None
        Cluster name matching the 'name' column. If None, a random cluster is chosen.
    frequency_bins : array-like or None
        X-axis values. If None, integer indices are used.
    max_num_pgs : int
        Maximum number of periodogram lines per subplot. Excess rows are randomly sampled.
    yscale : str
        Y-axis scale: 'log' (default) or 'linear'.
    xscale : str
        X-axis scale: 'linear' (default) or 'log'.
    x_lim : tuple of (min, max) or None
        X-axis range. Interpreted as bin indices if frequency_bins is None,
        or as frequency values if frequency_bins is provided.
    y_lim : tuple of (min, max) or None
        Y-axis range applied to all subplots.
    """
    if name is None:
        name = np.random.choice(master_table['name'].unique())
        print(f"Randomly selected cluster: '{name}'")

    name_str = name.decode() if isinstance(name, bytes) else name
    cluster_data = master_table[master_table['name'] == name]

    if cluster_data.empty:
        print(f"No data found for cluster '{name_str}'.")
        return

    ns_vals = sorted(cluster_data['n_sectors'].unique())
    n_subplots = len(ns_vals)

    fig, axes = plt.subplots(n_subplots, 1, figsize=(16, 6 * n_subplots))
    if n_subplots == 1:
        axes = [axes]

    for ax, ns in zip(axes, ns_vals):
        rows = cluster_data[cluster_data['n_sectors'] == ns]
        n_total = len(rows)

        sampled = rows.sample(min(max_num_pgs, n_total), random_state=42)
        colors = plt.cm.tab10(np.linspace(0, 1, len(sampled)))

        for (_, row), color in zip(sampled.iterrows(), colors):
            pg = row['FullPeriodogram']
            x = np.array(frequency_bins) if frequency_bins is not None else np.arange(len(pg))
            x, pg = _apply_x_lim(x, pg, x_lim)
            sectors_label = list(row['sectors'])
            label = f'{sectors_label} | {row["cadence"]:.0f} min'
            ax.plot(x, pg, lw=1.5, alpha=0.7, color=color, label=label)

        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        ax.set_ylabel('power', fontsize=14)
        ax.tick_params(labelsize=12)
        shown = len(sampled)
        ax.set_title(f'n_sectors={ns}  ({n_total} total, showing {shown})', fontsize=14)
        ax.legend(fontsize=11, ncol=2, loc='upper right')

    xlabel = 'frequency' if frequency_bins is not None else 'frequency (bin index)'
    axes[-1].set_xlabel(xlabel, fontsize=14)
    fig.suptitle(name_str, fontsize=18)
    fig.tight_layout()
    plt.show()


def plot_cluster_periodograms_specific(master_table, name, sectors_list,
                                       frequency_bins=None, yscale='log', xscale='linear',
                                       x_lim=None, y_lim=None):
    """
    Plot a specific set of periodograms for a cluster on a single axes.

    Parameters
    ----------
    master_table : pd.DataFrame
    name : str or bytes
        Cluster name matching the 'name' column.
    sectors_list : list of lists
        Each entry is a sector combination to look up and plot,
        e.g. [[1], [1, 2], [1, 2, 3]].
    frequency_bins : array-like or None
        X-axis values. If None, integer indices are used.
    yscale : str
        Y-axis scale: 'log' (default) or 'linear'.
    xscale : str
        X-axis scale: 'linear' (default) or 'log'.
    x_lim : tuple of (min, max) or None
        X-axis range. Interpreted as bin indices if frequency_bins is None,
        or as frequency values if frequency_bins is provided.
    y_lim : tuple of (min, max) or None
        Y-axis range.
    """
    name_str = name.decode() if isinstance(name, bytes) else name
    cluster_data = master_table[master_table['name'] == name]

    if cluster_data.empty:
        print(f"No data found for cluster '{name_str}'.")
        return

    fig, ax = plt.subplots(figsize=(16, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(sectors_list)))

    for sectors, color in zip(sectors_list, colors):
        match = cluster_data[cluster_data['sectors'].apply(
            lambda s: list(s) == list(sectors)
        )]
        if match.empty:
            print(f"No row found for sectors={sectors} in cluster '{name_str}'. Skipping.")
            continue
        row = match.iloc[0]
        pg = row['FullPeriodogram']
        x = np.array(frequency_bins) if frequency_bins is not None else np.arange(len(pg))
        x, pg = _apply_x_lim(x, pg, x_lim)
        label = f'{list(sectors)} | {row["cadence"]:.0f} min'
        ax.plot(x, pg, lw=1.5, alpha=0.8, color=color, label=label)
        ax.axhline(row['FAP'], color=color, lw=1.0, linestyle='--', alpha=0.6,
                   label=f'FAP={row["FAP"]:.2e}')

    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.set_ylabel('power', fontsize=14)
    xlabel = 'frequency' if frequency_bins is not None else 'frequency (bin index)'
    ax.set_xlabel(xlabel, fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title(name_str, fontsize=16)
    ax.legend(fontsize=12, loc='upper right')
    fig.tight_layout()
    plt.show()
