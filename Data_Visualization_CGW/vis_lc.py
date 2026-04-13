import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as astropy_fits

from invariant_summary_stats import LC_BASE, _slug, get_lc_path, resample_lc


def plot_cluster_lcs(master_table, name, x_zoom=None, plot_errors=False,
                     resample=False, cadence_bin_min=30.0):
    """
    Plot all LC sectors for a given cluster, one subplot per sector.
    Each subplot is labelled with the sector index and cadence.

    Parameters
    ----------
    master_table : pd.DataFrame
    name : str or bytes
        Cluster name matching the 'name' column in master_table.
    x_zoom : float or None
        If given, zoom each subplot to the first ``x_zoom`` fraction of its
        time range (e.g. 0.5 shows the first half of each sector).
    plot_errors : bool
        If True, plot flux uncertainties as a shaded band around the LC.
    resample : bool
        If True, apply resample_lc to each sector and overlay the resampled
        LC in orange on top of the original.
    cadence_bin_min : float
        Target cadence in minutes used when ``resample=True``. Default 30.
    """
    name_str = name.decode() if isinstance(name, bytes) else name
    path = get_lc_path(master_table, name)

    single_sector_rows = master_table[
        (master_table['name'] == name) & (master_table['n_sectors'] == 1)
    ]
    cadence_by_hdu = {}
    for _, row in single_sector_rows.iterrows():
        sectors = row['sectors']
        if hasattr(sectors, '__len__') and len(sectors) == 1:
            cadence_by_hdu[int(sectors[0])] = row['cadence']

    with astropy_fits.open(path) as hdul:
        n_sectors = len(hdul) - 1
        print(f"{name_str}  —  {n_sectors} HDU(s)  —  {path}")
        for j, hdu in enumerate(hdul):
            h = hdu.header
            print(f"  HDU {j}: {hdu.name:<12}  SECTOR={h.get('SECTOR','—')!s:<4}  "
                  f"TIMEDEL={h.get('TIMEDEL','—')!s:<10}  "
                  f"nrows={len(hdu.data) if hdu.data is not None else 0}")

        # First pass: compute global y-range across all sectors
        y_mins, y_maxs = [], []
        for i in range(n_sectors):
            data = hdul[i + 1].data
            f_all = np.asarray(data['flux'], dtype=float)
            err_all = np.asarray(data['flux_err'], dtype=float)
            med_all = np.nanmedian(f_all)
            if med_all == 0 or not np.isfinite(med_all):
                continue
            f_norm = f_all / med_all
            if plot_errors:
                y_mins.append(np.nanmin((f_all - err_all) / med_all))
                y_maxs.append(np.nanmax((f_all + err_all) / med_all))
            else:
                y_mins.append(np.nanmin(f_norm))
                y_maxs.append(np.nanmax(f_norm))
        if y_mins and y_maxs:
            y_span = max(y_maxs) - min(y_mins)
            pad = 0.05 * y_span
            global_ylim = (min(y_mins) - pad, max(y_maxs) + pad)
        else:
            global_ylim = None

        fig, axes = plt.subplots(n_sectors, 1, figsize=(12, 3 * n_sectors))
        if n_sectors == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            hdu_idx = i + 1
            hdu = hdul[hdu_idx]
            tess_sector = hdu.header.get('SECTOR', '?')
            cadence = cadence_by_hdu.get(i, '?')
            data = hdu.data
            if cadence == '?':
                diffs = np.diff(data['time'])
                cadence = round(float(np.median(diffs[np.isfinite(diffs)])) * 24 * 60, 1)
            t = np.asarray(data['time'], dtype=float)
            f = np.asarray(data['flux'], dtype=float)
            err = np.asarray(data['flux_err'], dtype=float)
            med = np.nanmedian(f)
            ax.plot(t, f / med, lw=0.5, color='steelblue', label='original')
            if plot_errors:
                ax.fill_between(t, (f - err) / med, (f + err) / med,
                                color='steelblue', alpha=0.3, linewidth=0)
            if resample:
                mask = np.isfinite(t) & np.isfinite(f) & np.isfinite(err)
                t_r, f_r, err_r = resample_lc(t[mask], f[mask], err[mask], cadence_bin_min)
                med_r = np.nanmedian(f_r)
                ax.plot(t_r, f_r / med_r, lw=1.2, color='orange',
                        alpha=0.85, label=f'resampled ({cadence_bin_min:.0f} min)')
                if plot_errors:
                    ax.fill_between(t_r, (f_r - err_r) / med_r, (f_r + err_r) / med_r,
                                    color='orange', alpha=0.3, linewidth=0)
                ax.legend(fontsize=7, loc='upper right')
            if global_ylim is not None:
                ax.set_ylim(global_ylim)
            if x_zoom is not None:
                t_finite = t[np.isfinite(t)]
                t_min, t_max = t_finite.min(), t_finite.max()
                ax.set_xlim(t_min, t_min + x_zoom * (t_max - t_min))
            ax.set_ylabel('flux / median')
            ax.set_title(f'Sector {hdu_idx}  |  TESS sector {tess_sector}  |  cadence = {cadence} min',
                         fontsize=9)

        axes[-1].set_xlabel('time (days)')
        fig.suptitle(name_str, fontsize=13, y=1.001)
        fig.tight_layout()
        plt.show()
