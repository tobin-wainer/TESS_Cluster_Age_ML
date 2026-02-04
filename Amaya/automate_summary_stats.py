from astropy import *
import astropy.units as u
import numpy as np
from astroquery.simbad import Simbad
import astropy.coordinates as coord

from astropy.io import ascii
from astropy.io import fits
import itertools
import os.path
# 
import elk
from elk.ensemble import EnsembleLC
from elk.lightcurve import BasicLightcurve

from astropy.table import Table, join, MaskedColumn, vstack, Column
from matplotlib import pyplot as plt
import glob
import time 
from itertools import combinations
import sys

from tqdm import tqdm
from scipy.special import entr

from astropy.table import Column
from scipy.stats import entropy as entr
#############################################################################################

def print_progress_bar(i, total, length=30, start_time=None):
    percent = i / total
    filled = int(length * percent)
    bar = '█' * filled + '-' * (length - filled)

    elapsed = time.time() - start_time
    if i > 0:
        eta = (elapsed / i) * (total - i)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
    else:
        eta_str = "--:--:--"

    sys.stdout.write(f'\rProgress: |{bar}| {round(percent * 100, 1)}% | ETA: {eta_str}')
    sys.stdout.flush()
    
def get_cluster_resampled_files(name):
    names = path + name + '*.fits'
    filenames = glob.glob(names)
    print('Name: {}, total number of files: {}'.format(name,len(filenames)))
    return filenames

def get_lightcurves(fnames):
    l_of_cs=[]
    sectors = []
    for file in fnames:
        hdul = fits.open(file)
        sectors.append(hdul[0].header['sectors'])
        hdul.close()
        data = elk.ensemble.from_fits(file)
        l_of_cs.append(elk.ensemble.from_fits(file))
    ###############
    l_of__all_lcs=[]
    for i in range(len(l_of_cs)):
        data = []
        med = 0
        for j in range(len(l_of_cs[i].lcs)):
            lc = l_of_cs[i].lcs[j].corrected_lc
            if j == 0:
                med = np.nanmedian(lc['flux'])
            else:
                delta = np.nanmedian(lc['flux']) - med
                lc['flux'] = lc['flux'] - delta
            data.append(lc)
        stitched_data = vstack(data)
        l_of__all_lcs.append(stitched_data)
    ##############
    data_augmented_lcs=[]
    for i in range(len(l_of__all_lcs)):
        if use_full_sectors:
            data_augmented_lcs.append(l_of__all_lcs[i])
    ############## 
    names=[]
    for i in range(len(l_of_cs)):
        names.append(l_of_cs[i].callable)

    return names, sectors, data_augmented_lcs


def make_stats_table(names, sectors, data_augmented_lcs):
    l_of_stat_tables = []

    # Start timer before the loop
    start_time = time.time()
    for i in range(len(data_augmented_lcs)):
        print_progress_bar(i + 1, len(data_augmented_lcs), start_time=start_time)

        lc = BasicLightcurve(data_augmented_lcs[i]['time'],
                         data_augmented_lcs[i]['flux'],
                         data_augmented_lcs[i]['flux_err'],
                         sector=99)

        lc.get_stats_using_defaults()

        table = lc.get_stats_table(names[i])[['name','rms','std','MAD','sigmaG','skewness','von_neumann_ratio','J_Stetson','max_power','freq_at_max_power',
        'n_peaks','ratio_of_power_at_high_v_low_freq','FAP','max_autocorrelation','time_of_max_autocorrelation']]

        # Periodogram and frequency grid
        frequency_list = 1 / np.arange(0.04, 11, 0.01)
        periodogram = lc.periodogram

        # Because the lightcurves were smoothed over timescales >10days, don't use those scales.
        periodogram = periodogram[(frequency_list < 10)]
        frequency_list = frequency_list[(frequency_list < 10)]
    
        # Sum power in specific period bands
        sum_LSP_power_10_7_days = np.sum(periodogram[(frequency_list < 10) & (frequency_list > 7)])
        sum_LSP_power_7_4_days  = np.sum(periodogram[(frequency_list < 7) & (frequency_list > 4)])
        sum_LSP_power_4_1_days  = np.sum(periodogram[(frequency_list < 4) & (frequency_list > 1)])
        sum_LSP_power_1_p5_days = np.sum(periodogram[(frequency_list < 1) & (frequency_list > 0.5)])

        # Shannon entropy of the flux
        entropy_val = entr(data_augmented_lcs[i]['flux'].value).sum()

        # Add new features
        table.add_column(Column(sum_LSP_power_10_7_days), name='SumLSP_10_7_Day_Power')
        table.add_column(Column(sum_LSP_power_7_4_days), name='SumLSP_7_4_Day_Power')
        table.add_column(Column(sum_LSP_power_4_1_days), name='SumLSP_4_1_Day_Power')
        table.add_column(Column(sum_LSP_power_1_p5_days), name='SumLSP_1_p5_Day_Power')
        table.add_column(Column(entropy_val), name='Entropy')

        # NEW: add full periodogram as a variable-length array column
        table.add_column(Column([periodogram], name='FullPeriodogram'))
        l_of_stat_tables.append(table)
        
    stat_table=vstack(l_of_stat_tables)
    stat_table['sectors'] = sectors
    return stat_table

def get_loc_age(stat_table):
    ages=[]
    loc=[]
    for name in list(stat_table['name']):
        for i in range(len(mw)):
            if mw[i]['NAME']== name:
                ages.append(mw[i]['LOG_AGE'])
                loc.append('MW')
    
        for j in range(len(smc)):
            if smc[j]['SimbadName']==name:
                ages.append(smc[j]['logAge'])
                loc.append('SMC')
            
        for k in range(len(lmc)):
            if lmc[k]['SimbadName']==name:
                ages.append(lmc[k]['Age'])
                loc.append('LMC')
    stat_table['LOC'] = loc
    stat_table['AGE'] = ages
    return stat_table

def save_cluster_data(name, table):
    fpath = '/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/tess_data/data/cluster_summary_files/'
    f_name = '{}{}_summary_data.fits'.format(fpath,name)
    table.write(f_name,overwrite=True)

def main(name):
    print('reading filenames....')
    filenames = get_cluster_resampled_files(name)
    print('making & augmenting lightcurves....')
    names, sectors, data_augmented_lcs = get_lightcurves(filenames)
    print('N ROWS: {}'.format(len(names)))
    print('making stats table....')
    stat_table = make_stats_table(names, sectors, data_augmented_lcs)
    print()
    print('adding age & location data....')
    stat_table = get_loc_age(stat_table)
    print('writing out summary file....')
    save_cluster_data(name, stat_table)
    print('Done with {}!'.format(name))
    print()
    
    
#############################################################################################

path= '/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/tess_data/data/resampled_summary_data/'
lc_path='/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/tess_data/light_curves/'
age_path = '/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/tess_data/data/'


# f = open('cluster_names.txt', 'r')
# names = f.readlines()
# cnames = []
# for i in names:
#     t = i.split('\n')[0]
#     cnames.append(t)


cluster_stats = Table.read('../cluster_summary_statistics.fits')
cluster_stats = cluster_stats[cluster_stats['n_rows'] > 2]
cluster_stats.sort('n_rows')
cnames = list(cluster_stats['cluster_name'])

# for i in cnames:
#     t = get_cluster_resampled_files(i)

lower = int(input('lower bound: '))
upper = int(input('upper bound: '))

use_full_sectors = True
use_12day_rolling_window = False
use_stiched_sectors = False

mw=Table.read(age_path + 'Use_MW.fits')
smc=Table.read(age_path + 'Bica_Cut_down.fits')
lmc=Table.read(age_path + 'Glatt_Cut_down.fits')

count = lower
for i in cnames[lower:upper]:
    print(i)
    main(i)
    count +=1
    print('Done {}/{}'.format(count,len(cnames)))
    
print('COMPLETED!!!')
    
