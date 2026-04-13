import os
import glob as glob_module
import numpy as np
from astropy.io import fits as astropy_fits

LC_BASE = '/astro/users/cgwill/TESS_Cluster_Age_ML/light_curves'


def _slug(name):
    """Canonical key for matching: lowercase, strip spaces, hyphens, and brackets."""
    s = name.decode() if isinstance(name, bytes) else name
    return s.lower().replace(' ', '').replace('-', '').replace('[', '').replace(']', '')


def get_lc_path(master_table, name):
    """
    Given a cluster name (str or bytes), return the path to its LC .fits file.
    Uses the LOC column from master_table to determine the subdirectory.
    Matches by stripping spaces/hyphens from both the name and filename slug,
    to handle complex multi-word catalog names (e.g. 'OGLE CL SMC 133' -> ogle-cl-smc-133).
    """
    name_str = name.decode() if isinstance(name, bytes) else name
    row = master_table[master_table['name'] == name].iloc[0]
    loc = row['LOC'].decode() if isinstance(row['LOC'], bytes) else row['LOC']

    target = _slug(name_str)
    pattern = f'{LC_BASE}/{loc}/hlsp_elk_tess_ffi_*_tess_v1_llc.fits'
    for fpath in glob_module.glob(pattern):
        basename = os.path.basename(fpath)
        slug = basename.replace('hlsp_elk_tess_ffi_', '').replace('_tess_v1_llc.fits', '')
        if _slug(slug) == target:
            return fpath

    raise FileNotFoundError(f"No LC file found for '{name_str}' in {LC_BASE}/{loc}/")


# ---------------------------------------------------------------------------
# Step 1: Cadence resampling (on raw flux, before normalization)
# ---------------------------------------------------------------------------

def resample_lc(t, f, err_f, cadence_bin_min=30.0):
    """
    Downsample a raw lightcurve to a common cadence by block-averaging
    groups of N consecutive points.

    Resampling is performed on raw flux before normalization. Groups that
    contain an internal time gap (diff > 1.5 × native cadence) are discarded
    entirely so that no averaging occurs across gaps.

    TESS cadences are 3.33, 10, and 30 min — integer multiples of 3 — so the
    decimation factor N is always 1, 3, or 9. An assertion enforces this.

    Parameters
    ----------
    t, f, err_f : np.ndarray
        Time (days), raw flux, and raw flux uncertainty. Must be finite and
        time-sorted before calling.
    cadence_bin_min : float
        Target cadence in minutes. Default 30.

    Returns
    -------
    t_out, f_out, err_f_out : np.ndarray
        Resampled arrays. Returned unchanged if N <= 1.
    """
    if len(t) < 2:
        return t.copy(), f.copy(), err_f.copy()

    native_cadence_min = np.nanmedian(np.diff(t)) * 1440.0
    N = round(cadence_bin_min / native_cadence_min)

    if N <= 1:
        return t.copy(), f.copy(), err_f.copy()

    assert N == 1 or N % 3 == 0, (
        f"Unexpected decimation factor N={N} for native cadence {native_cadence_min:.2f} min "
        f"and target {cadence_bin_min:.1f} min. Expected N in {{1, 3, 9}} (TESS cadences are "
        f"3.33, 10, 30 min)."
    )

    dt = np.diff(t)
    gap_threshold = 1.5 * (native_cadence_min / 1440.0)

    t_out, f_out, err_out = [], [], []
    n_discarded = 0

    for i in range(0, len(t) - N + 1, N):
        if np.any(dt[i:i + N - 1] > gap_threshold):
            n_discarded += 1
            continue
        t_out.append(np.mean(t[i:i + N]))
        f_out.append(np.mean(f[i:i + N]))
        err_out.append(np.sqrt(np.sum(err_f[i:i + N] ** 2)) / N)

    n_out = len(t_out)
    if n_out < 2:
        return t.copy(), f.copy(), err_f.copy()

    return np.array(t_out), np.array(f_out), np.array(err_out)


# ---------------------------------------------------------------------------
# Step 2: Flux normalization
# ---------------------------------------------------------------------------

def normalize_flux(t, f, err_f):
    """
    Normalize a lightcurve to relative flux centered on 1.

    Drops all points where t, f, or err_f is non-finite, then computes a
    single global median and transforms:

        x_i     = f_i / median(f)
        err_x_i = err_f_i / median(f)

    Parameters
    ----------
    t, f, err_f : array-like
        Time, flux, and flux uncertainty arrays of equal length.

    Returns
    -------
    t_clean, x, err_x : np.ndarray
        Filtered and normalized arrays.

    Raises
    ------
    ValueError
        If fewer than 2 finite points remain after filtering.
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    err_f = np.asarray(err_f, dtype=float)

    mask = np.isfinite(t) & np.isfinite(f) & np.isfinite(err_f)
    t, f, err_f = t[mask], f[mask], err_f[mask]

    if len(f) < 2:
        raise ValueError("Fewer than 2 finite points after masking NaNs/infs.")

    med = np.median(f)
    if med == 0 or not np.isfinite(med):
        raise ValueError(f"Median flux is {med}; cannot normalize.")

    x = f / med
    err_x = err_f / med
    return t, x, err_x


# ---------------------------------------------------------------------------
# Step 3: Noise-corrected statistics
# ---------------------------------------------------------------------------

def _compute_stats(x_binned, err_x_binned, sigma_clip_thresh=None):
    """
    Compute noise-corrected variability statistics on a resampled lightcurve.

    Parameters
    ----------
    x_binned, err_x_binned : np.ndarray
        Resampled normalized flux and uncertainty.
    sigma_clip_thresh : float or None
        If given, iteratively sigma-clip outliers at this threshold (up to
        5 iterations) before computing statistics.

    Returns
    -------
    dict with keys:
        intrinsic_std   : noise-corrected standard deviation
        intrinsic_rms   : noise-corrected RMS
        intrinsic_mad   : noise-corrected MAD (Gaussian-equivalent sigma)
        n_bins          : number of points used after clipping
        sigma_phot      : sqrt(mean photon noise variance)
        sigma_obs       : observed standard deviation (before noise correction)
        rms_obs         : observed RMS around 1 (before noise correction)
        sigma_mad_obs   : observed MAD scaled to Gaussian sigma (1.4826 * MAD)
    """
    x = x_binned.copy()
    sx = err_x_binned.copy()

    if sigma_clip_thresh is not None:
        n_before = len(x)
        for i in range(5):
            mu = x.mean()
            std = x.std()
            if std == 0:
                break
            keep = np.abs(x - mu) <= sigma_clip_thresh * std
            if keep.sum() == len(x):
                break
            x, sx = x[keep], sx[keep]

    N = len(x)
    sigma_phot2 = np.mean(sx ** 2)
    sigma_phot  = np.sqrt(sigma_phot2)

    sigma_obs2 = np.mean((x - x.mean()) ** 2)
    sigma_obs  = np.sqrt(sigma_obs2)
    intrinsic_std = np.sqrt(max(0.0, sigma_obs2 - sigma_phot2))

    rms_obs2 = np.mean((x - 1.0) ** 2)
    rms_obs  = np.sqrt(rms_obs2)
    intrinsic_rms = np.sqrt(max(0.0, rms_obs2 - sigma_phot2))

    mad_obs = np.median(np.abs(x - np.median(x)))
    sigma_mad_obs = 1.4826 * mad_obs
    intrinsic_mad = np.sqrt(max(0.0, sigma_mad_obs ** 2 - sigma_phot2))

    for stat_name, val in [('sigma_phot', sigma_phot), ('sigma_obs', sigma_obs),
                            ('rms_obs', rms_obs), ('sigma_mad_obs', sigma_mad_obs),
                            ('intrinsic_std', intrinsic_std), ('intrinsic_rms', intrinsic_rms),
                            ('intrinsic_mad', intrinsic_mad)]:
        if not np.isfinite(val):
            print(f"  WARNING [_compute_stats] {stat_name}=NaN/inf  |  "
                  f"N={N}  sigma_phot2={sigma_phot2:.4e}  sigma_obs2={sigma_obs2:.4e}  "
                  f"rms_obs2={rms_obs2:.4e}  sigma_mad_obs2={sigma_mad_obs**2:.4e}  "
                  f"x: min={x.min():.4e} max={x.max():.4e} mean={x.mean():.4e}  "
                  f"sx: min={sx.min():.4e} max={sx.max():.4e}")

    return {
        'intrinsic_std':  intrinsic_std,
        'intrinsic_rms':  intrinsic_rms,
        'intrinsic_mad':  intrinsic_mad,
        'n_bins':         N,
        'sigma_phot':     sigma_phot,
        'sigma_obs':      sigma_obs,
        'rms_obs':        rms_obs,
        'sigma_mad_obs':  sigma_mad_obs,
    }


# ---------------------------------------------------------------------------
# Step 4: Full pipeline wrapper
# ---------------------------------------------------------------------------

def compute_variability_metrics(t, f, err_f,
                                cadence_bin_min=30.0,
                                sigma_clip_thresh=None):
    """
    Compute full-baseline, cadence-standardized, noise-corrected variability
    metrics for a TESS lightcurve.

    Pipeline
    --------
    1. resample_lc     — block-average raw flux to common cadence (before normalization)
    2. normalize_flux  — relative flux via global median (x = f/median, centered on 1)
    3. _compute_stats  — noise-corrected STD, RMS, MAD

    Corrections applied
    -------------------
    - Cadence: raw flux resampled to ``cadence_bin_min`` minutes before
      normalization, so metrics are comparable across different native cadences.
    - Flux normalization: relative flux x_i = f_i/median(f)
    - Photometric noise: mean measurement variance subtracted in quadrature.

    NOT corrected
    -------------
    - Total baseline length: longer baselines (more stitched sectors) may
      naturally show higher variability if the source has long-timescale power.
      No duration normalization is applied.

    Parameters
    ----------
    t, f, err_f : array-like
        Time (days), raw flux, and raw flux uncertainty.
    cadence_bin_min : float
        Target resampling cadence in minutes. Default 30.
    sigma_clip_thresh : float or None
        Sigma-clipping threshold applied after resampling. None = no clipping.

    Returns
    -------
    dict
        All keys from ``_compute_stats`` plus ``cadence_bin_min``.
        Returns None if normalization fails (e.g. all-NaN input).
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    err_f = np.asarray(err_f, dtype=float)

    t, f, err_f = resample_lc(t, f, err_f, cadence_bin_min)

    try:
        t_norm, x, err_x = normalize_flux(t, f, err_f)
    except ValueError as e:
        print(f"  WARNING [compute_variability_metrics] normalize_flux failed: {e}")
        return None

    result = _compute_stats(x, err_x, sigma_clip_thresh)
    result['cadence_bin_min'] = cadence_bin_min
    return result


# ---------------------------------------------------------------------------
# Master-table wrapper
# ---------------------------------------------------------------------------

def add_variability_metrics(master_table, cadence_bin_min=30.0,
                             sigma_clip_thresh=None):
    """
    Add full-baseline variability metric columns to master_table in-place.

    For each row, loads all sectors listed in the 'sectors' column. Each sector
    is resampled to the common cadence on raw flux, then normalized individually
    to remove inter-sector flux offsets. Normalized sectors are concatenated and
    variability statistics are computed over the full baseline.

    Adds columns: ``intrinsic_std``, ``intrinsic_rms``, ``intrinsic_mad``,
    ``n_bins_used``. Rows whose lightcurve file cannot be found or that fail
    for any other reason are left as NaN.

    Parameters
    ----------
    master_table : pd.DataFrame
    cadence_bin_min : float
    sigma_clip_thresh : float or None

    Returns
    -------
    master_table : pd.DataFrame (modified in-place)
    """
    n_clusters = master_table['name'].nunique()
    n_rows = len(master_table)
    print(f"[add_variability_metrics] processing {n_rows} rows across "
          f"{n_clusters} clusters  |  cadence_bin_min={cadence_bin_min} min")

    n = len(master_table)
    intrinsic_std_vals = np.full(n, np.nan)
    intrinsic_rms_vals = np.full(n, np.nan)
    intrinsic_mad_vals = np.full(n, np.nan)
    n_bins_vals        = np.full(n, np.nan)

    n_ok, n_missing, n_failed = 0, 0, 0

    for name, group in master_table.groupby('name'):
        name_str = name.decode() if isinstance(name, bytes) else name
        try:
            path = get_lc_path(master_table, name)
        except FileNotFoundError:
            print(f"  WARNING [add_variability_metrics] {name_str}: LC file not found — skipping")
            n_missing += len(group)
            continue

        try:
            with astropy_fits.open(path) as hdul:
                for idx, row in group.iterrows():
                    try:
                        sectors = row['sectors']
                        all_t, all_x, all_err_x = [], [], []

                        for s in sectors:
                            hdu_idx = int(s) + 1
                            data = hdul[hdu_idx].data

                            time = np.asarray(data['time'],     dtype=float)
                            flux = np.asarray(data['flux'],     dtype=float)
                            ferr = np.asarray(data['flux_err'], dtype=float)

                            mask = (np.isfinite(time) &
                                    np.isfinite(flux) &
                                    np.isfinite(ferr))
                            time, flux, ferr = time[mask], flux[mask], ferr[mask]
                            if len(time) < 2:
                                print(f"  WARNING [{name_str}] row {idx} sector {s}: "
                                      f"too few finite points — skipping sector")
                                continue

                            time, flux, ferr = resample_lc(time, flux, ferr, cadence_bin_min)
                            try:
                                t_s, x_s, err_x_s = normalize_flux(time, flux, ferr)
                            except ValueError as e:
                                print(f"  WARNING [{name_str}] row {idx} sector {s}: "
                                      f"normalize_flux failed — {e}")
                                continue
                            all_t.append(t_s)
                            all_x.append(x_s)
                            all_err_x.append(err_x_s)

                        if not all_t:
                            print(f"  WARNING [{name_str}] row {idx}: "
                                  f"no valid data in any sector — skipping")
                            n_failed += 1
                            continue

                        t = np.concatenate(all_t)
                        x = np.concatenate(all_x)
                        err_x = np.concatenate(all_err_x)

                        sort_idx = np.argsort(t)
                        t, x, err_x = t[sort_idx], x[sort_idx], err_x[sort_idx]

                        result = _compute_stats(x, err_x, sigma_clip_thresh)
                        result['cadence_bin_min'] = cadence_bin_min

                        for key, arr in [('intrinsic_std', intrinsic_std_vals),
                                         ('intrinsic_rms', intrinsic_rms_vals),
                                         ('intrinsic_mad', intrinsic_mad_vals)]:
                            val = result[key]
                            if not np.isfinite(val):
                                print(f"  WARNING [{name_str}] row {idx} "
                                      f"sectors={list(sectors)}: {key}=NaN/inf")
                            arr[idx] = val
                        n_bins_vals[idx] = result['n_bins']
                        n_ok += 1

                    except Exception as e:
                        print(f"  WARNING [{name_str}] row {idx}: unexpected error — {e}")
                        n_failed += 1

        except Exception as e:
            print(f"  WARNING [{name_str}]: failed to open FITS — {e}")
            n_failed += len(group)

    print(f"[add_variability_metrics] done  |  "
          f"ok={n_ok}  missing={n_missing}  failed={n_failed}")

    master_table['intrinsic_std'] = intrinsic_std_vals
    master_table['intrinsic_rms'] = intrinsic_rms_vals
    master_table['intrinsic_mad'] = intrinsic_mad_vals
    master_table['n_bins_used']   = n_bins_vals
    return master_table


def add_std_norm(master_table, cadence_bin_min=30.0):
    """Deprecated alias for add_variability_metrics."""
    return add_variability_metrics(master_table, cadence_bin_min=cadence_bin_min)
