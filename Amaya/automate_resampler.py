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

path = '/uufs/astro.utah.edu/common/home/u1363702/notebooks/tess_clusters/TESS_Cluster_Age_ML'
lc_path='/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/tess_data/light_curves/'
resampled_fpath = '/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/tess_data/light_curves/resampled_fits_tables/'

##############

def get_combinations(num):

    arr = np.arange(1,num)

    all_combinations = []

    for i in range(1, 5):
        combos = list(combinations(arr,i))
        for j in combos:
            all_combinations.append(list(j))
    return all_combinations

def make_resampled_files(name):
    with fits.open(lc_path+ 'all/' + name + '.fits') as hdul:
        num_hdus = len(hdul)
        count = 0
        combos = get_combinations(num_hdus)
        print('N combinations: {}'.format(len(combos)))
        for combination in combos:
            keep_hdus = [hdul[0]]
            string = ''
            for idx,j in enumerate(combination):
                keep_hdus.append(hdul[j])
                string = string + str(j) + ','
            string = string[:-1]
            new_hdul = fits.HDUList(keep_hdus)
            new_hdul[0].header['sectors'] = string
            count +=1

            fname = '{}_{}.fits'.format(name,count)
            new_hdul.writeto(resampled_fpath + fname, overwrite=True)
        hdul.close()
        print('Done: ' + name)


def main():

    print('loading in summary data ....')
    cluster_summary_stats = Table.read('cluster_summary_statistics.fits')
    cluster_summary_stats = cluster_summary_stats[cluster_summary_stats['n_rows'] > 2]

    filenames =  glob.glob(lc_path + 'all/*.fits')
    l_of_cs=[]
    fnames = []
    nhdus = []

    print('getting filenames ....')

    t0 = time.time()
    print(len(filenames))
    for file in filenames:
        hdul = fits.open(file)
        num_hdus = len(hdul) - 1
        hdul.close()

        cluster_fname = file.split(lc_path + 'all/')[1].split('.fits')[0]
        if np.isin(cluster_fname, cluster_summary_stats['cluster_name']):
            fnames.append(file)
            nhdus.append(num_hdus)
    t1 = time.time()
    delta = t1 - t0
    print('for all clusters, took {} seconds'.format(delta))
    t1 = time.time()
    delta = t1 - t0

    cluster_files = []
    for i in filenames:
        t = i.split(lc_path + 'all/')[1]
        t = t.split('.fits')[0]
        if np.isin(t, cluster_summary_stats['cluster_name']):
            cluster_files.append(t)

    print('writing cluster_names.txt ....')
    
    with open('Amaya/cluster_names.txt', 'w') as f:
        for line in cluster_files:
            f.write(f"{line}\n")
    f.close()
    
    print('beginning resampling ....')
    
    count = 0
    t0 = time.time()
    for i in cluster_files:
        make_resampled_files(i)
        t1 = time.time()
        dt = np.round((t1-t0)/60,1)
        count +=1
        print('Time elapsed: {} minutes; Done {}/{} clusters'.format(dt, count, len(cluster_files)))
        print()
    print('finished!')


main()