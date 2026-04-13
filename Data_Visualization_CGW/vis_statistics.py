import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import gaussian_kde


def draw_half_violin(ax, pos, data, side='left', color='steelblue', alpha=0.7, width=0.35):
    """Draw one half of a violin using KDE, untruncated (tails extend naturally)."""
    if len(data) == 0:
        return
    if len(data) == 1:
        ax.hlines(data[0], pos - width * 0.8, pos + width * 0.8, colors=color,
                  linewidth=1.5, alpha=alpha)
        ax.text(pos, data[0], f'n=1: {data[0]:.3g}', ha='center', va='bottom',
                fontsize=7, color=color, alpha=min(alpha + 0.15, 1.0))
        return
    if np.all(data == data[0]):
        ax.hlines(data[0], pos - width * 0.8, pos + width * 0.8, colors=color,
                  linewidth=1.5, alpha=alpha)
        ax.text(pos, data[0], f'all={data[0]:.3g}', ha='center', va='bottom',
                fontsize=7, color=color, alpha=min(alpha + 0.15, 1.0))
        return
    kde = gaussian_kde(data, bw_method=0.5)
    data_range = data.max() - data.min() if data.max() != data.min() else 1.0
    y_grid = np.linspace(data.min() - 0.5 * data_range, data.max() + 0.5 * data_range, 300)
    density = kde(y_grid)
    density = density / density.max() * width

    if side == 'left':
        ax.fill_betweenx(y_grid, pos - density, pos, alpha=alpha, color=color)
        ax.hlines(np.median(data), pos - width * 0.8, pos, colors='white', linewidth=1.5, zorder=5)
    else:
        ax.fill_betweenx(y_grid, pos, pos + density, alpha=alpha, color=color)
        ax.hlines(np.median(data), pos, pos + width * 0.8, colors='white', linewidth=1.5, zorder=5)


def draw_half_violin_h(ax, pos, data, side='bottom', color='steelblue', alpha=0.7, width=0.35):
    """Draw one half of a horizontal violin using KDE, untruncated.
    'bottom' half goes below pos, 'top' half goes above pos.
    """
    if len(data) == 0:
        return
    if len(data) == 1:
        ax.vlines(data[0], pos - width * 0.8, pos + width * 0.8, colors=color,
                  linewidth=1.5, alpha=alpha)
        ax.text(data[0], pos, f'n=1: {data[0]:.3g}', ha='left', va='center',
                fontsize=7, color=color, alpha=min(alpha + 0.15, 1.0))
        return
    if np.all(data == data[0]):
        ax.vlines(data[0], pos - width * 0.8, pos + width * 0.8, colors=color,
                  linewidth=1.5, alpha=alpha)
        ax.text(data[0], pos, f'all={data[0]:.3g}', ha='left', va='center',
                fontsize=7, color=color, alpha=min(alpha + 0.15, 1.0))
        return
    kde = gaussian_kde(data, bw_method=0.5)
    data_range = data.max() - data.min() if data.max() != data.min() else 1.0
    x_grid = np.linspace(data.min() - 0.5 * data_range, data.max() + 0.5 * data_range, 300)
    density = kde(x_grid)
    density = density / density.max() * width

    if side == 'bottom':
        ax.fill_between(x_grid, pos - density, pos, alpha=alpha, color=color)
        ax.vlines(np.median(data), pos - width * 0.8, pos, colors='white', linewidth=1.5, zorder=5)
    else:
        ax.fill_between(x_grid, pos, pos + density, alpha=alpha, color=color)
        ax.vlines(np.median(data), pos, pos + width * 0.8, colors='white', linewidth=1.5, zorder=5)


def plot_cluster_statistics(master_table, name, statistics_list, max_n_sectors=6):
    """
    For a given cluster, plot violin plots of each statistic grouped by n_sectors.
    Each violin is split: left half = full distribution, right half = overlapping
    per-cadence distributions.

    Parameters
    ----------
    master_table : pd.DataFrame
    name : str or None
        Cluster name (matches the 'name' column). If None, a random cluster is selected.
    statistics_list : list of str
        Column names to plot; one subplot per statistic.
    max_n_sectors : int
        Maximum n_sectors to plot. Capped at actual max in the data.
    """
    if name is None:
        name = np.random.choice(master_table['name'].unique())
        print(f"Randomly selected cluster: '{name}'")

    cluster_data = master_table[master_table['name'] == name]

    if cluster_data.empty:
        print(f"No data found for cluster '{name}'.")
        return

    actual_max = int(cluster_data['n_sectors'].max())
    n_sectors_max = min(max_n_sectors, actual_max)
    sector_values = list(range(1, n_sectors_max + 1))

    cadences = sorted(cluster_data['cadence'].unique())
    cadence_colors = plt.cm.Set2(np.linspace(0, 1, max(len(cadences), 3)))
    cadence_color_map = {c: cadence_colors[i] for i, c in enumerate(cadences)}

    n_stats = len(statistics_list)
    fig, axes = plt.subplots(n_stats, 1, figsize=(8, 4 * n_stats))
    if n_stats == 1:
        axes = [axes]

    for ax, stat in zip(axes, statistics_list):
        for s in sector_values:
            sector_mask = cluster_data['n_sectors'] == s
            all_vals = cluster_data.loc[sector_mask, stat].dropna().values

            draw_half_violin(ax, pos=s, data=all_vals, side='left',
                             color='steelblue', alpha=0.85)

            for cadence in cadences:
                cad_vals = cluster_data.loc[
                    sector_mask & (cluster_data['cadence'] == cadence), stat
                ].dropna().values
                if len(cad_vals) < 2:
                    continue
                draw_half_violin(ax, pos=s, data=cad_vals, side='right',
                                 color=cadence_color_map[cadence], alpha=0.5)

            ax.axvline(x=s, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)

        ax.set_xticks(sector_values)
        ax.set_xticklabels([str(s) for s in sector_values])
        ax.set_xlabel('n_sectors')
        ax.set_ylabel(stat)
        ax.set_title(f'{name} — {stat}')

    legend_handles = [mpatches.Patch(color='steelblue', alpha=0.85, label='All (left)')]
    for c in cadences:
        legend_handles.append(mpatches.Patch(color=cadence_color_map[c], alpha=0.7, label=f'cadence={c}'))
    axes[0].legend(handles=legend_handles, title='cadence (right)', loc='upper right')

    fig.tight_layout()
    plt.show()


def _draw_per_cadence_h(ax, pos, cluster_mask, subset, column,
                        cadences, cadence_color_map, side, alpha=0.7):
    """Draw per-cadence half-violins for one column on one side of a row."""
    for cadence in cadences:
        vals = subset.loc[
            cluster_mask & (subset['cadence'] == cadence), column
        ].dropna().values
        if len(vals) == 0:
            continue
        draw_half_violin_h(ax, pos=pos, data=vals, side=side,
                           color=cadence_color_map[cadence], alpha=alpha)


def plot_clusters_comparison(master_table, n_sectors, statistic, max_clusters=None,
                             x_lim=None, sort_by='n_sectors', comparison_column=None,
                             debug=False):
    """
    Compare all clusters for a given n_sectors and statistic.

    Default mode: each cluster gets a horizontal split violin where
    bottom = full distribution of ``statistic``, top = per-cadence distributions.

    With ``comparison_column``: bottom = per-cadence distributions of ``statistic``,
    top = per-cadence distributions of ``comparison_column``. Useful for directly
    comparing e.g. 'std' vs 'intrinsic_std' across clusters.

    Parameters
    ----------
    master_table : pd.DataFrame
    n_sectors : int
        The n_sectors value to filter on.
    statistic : str
        The column name to plot on the x-axis (bottom half).
    max_clusters : int or None
        If set, randomly sample this many clusters (useful for readability).
    x_lim : tuple or None
    sort_by : str
    comparison_column : str or None
        If given, plot per-cadence violins of this column on the top half
        instead of the default all-data violin.
    """
    cols = ['name', 'cadence', statistic]
    if comparison_column is not None:
        cols.append(comparison_column)
    subset = master_table[master_table['n_sectors'] == n_sectors][cols].dropna(subset=[statistic])

    if subset.empty:
        print(f"No data found for n_sectors={n_sectors}.")
        return

    max_nsectors_per_cluster = master_table.groupby('name')['n_sectors'].max()

    if sort_by == 'n_sectors':
        sort_vals = max_nsectors_per_cluster
    else:
        sort_vals = master_table.groupby('name')[sort_by].first()

    cluster_names = sorted(subset['name'].unique(),
                           key=lambda c: sort_vals.get(c, 0))

    if max_clusters is not None and len(cluster_names) > max_clusters:
        cluster_names = list(np.random.choice(cluster_names, max_clusters, replace=False))
        cluster_names = sorted(cluster_names, key=lambda c: max_nsectors_per_cluster.get(c, 0))

    cadences = sorted(subset['cadence'].unique())
    cadence_colors = plt.cm.Set2(np.linspace(0, 1, max(len(cadences), 3)))
    cadence_color_map = {c: cadence_colors[i] for i, c in enumerate(cadences)}

    n_clusters = len(cluster_names)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.8 * n_clusters)))

    for i, cluster in enumerate(cluster_names):
        cluster_mask = subset['name'] == cluster

        if debug:
            cluster_str = cluster.decode() if isinstance(cluster, bytes) else cluster
            for cadence in cadences:
                cad_mask = cluster_mask & (subset['cadence'] == cadence)
                vals = subset.loc[cad_mask, statistic].dropna().values
                print(f"  [{cluster_str}] cadence={cadence}  n={len(vals)}  "
                      f"{statistic}={vals}")
                if comparison_column is not None:
                    cmp_vals = subset.loc[cad_mask, comparison_column].dropna().values
                    print(f"  [{cluster_str}] cadence={cadence}  n={len(cmp_vals)}  "
                          f"{comparison_column}={cmp_vals}")

        if comparison_column is None:
            # Default: bottom = all data, top = per-cadence
            all_vals = subset.loc[cluster_mask, statistic].dropna().values
            draw_half_violin_h(ax, pos=i, data=all_vals, side='bottom',
                               color='steelblue', alpha=0.85)
            _draw_per_cadence_h(ax, i, cluster_mask, subset, statistic,
                                cadences, cadence_color_map, side='top', alpha=0.5)
        else:
            # Comparison: bottom = statistic per-cadence, top = comparison_column per-cadence
            _draw_per_cadence_h(ax, i, cluster_mask, subset, statistic,
                                cadences, cadence_color_map, side='bottom', alpha=0.85)
            _draw_per_cadence_h(ax, i, cluster_mask, subset, comparison_column,
                                cadences, cadence_color_map, side='top', alpha=0.5)

        ax.axhline(y=i, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)

    yticklabels = [f'{c}  (max={max_nsectors_per_cluster.get(c, "?")})'
                   for c in cluster_names]
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel(statistic if comparison_column is None else f'{statistic}  /  {comparison_column}')

    if comparison_column is None:
        ax.set_title(f'{statistic} by cluster  (n_sectors={n_sectors})')
        legend_handles = [mpatches.Patch(color='steelblue', alpha=0.85, label='All (bottom)')]
        for c in cadences:
            legend_handles.append(mpatches.Patch(color=cadence_color_map[c], alpha=0.7, label=f'cadence={c}'))
        ax.legend(handles=legend_handles, title='cadence (top)', loc='upper right')
    else:
        ax.set_title(f'{statistic} (bottom)  vs  {comparison_column} (top)  —  n_sectors={n_sectors}')
        legend_handles = []
        for c in cadences:
            legend_handles.append(mpatches.Patch(color=cadence_color_map[c], alpha=0.7, label=f'cadence={c}'))
        ax.legend(handles=legend_handles, loc='upper right')

    if x_lim is not None:
        ax.set_xlim(x_lim)

    fig.tight_layout()
    plt.show()
