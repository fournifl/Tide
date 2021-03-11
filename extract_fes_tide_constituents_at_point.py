import numpy as np
from os.path import join as join
from os.path import exists as exists
from netCDF4 import Dataset
import pdb


def get_fes_constituent_fname(fes_2014_constituent, path_fes_2014_data):
    if fes_2014_constituent == 'LAMBDA2':
        name_constituent = 'la2'
    else:
        name_constituent = fes_2014_constituent.lower()
    try:
        fname_constituent = join(path_fes_2014_data, name_constituent + '.nc')
    except IOError:
        print('There was an error opening the file!')
        return
    return fname_constituent


def get_fes_2014_data_at_extraction_point(fname_constituent, lon_extraction, lat_extraction):
    with Dataset(fname_constituent, 'r') as fes_data:
        grid_lat = fes_data.variables['lat'][:]
        grid_lon = fes_data.variables['lon'][:]
        ind_lon, ind_lat = get_closest_grid_point(grid_lon, grid_lat, lon_extraction, lat_extraction)
        print('fname constituent: %s' % fname_constituent)
        print('ind_lon: %s' %ind_lon)
        print('ind_lat: %s' %ind_lat)
        print('lon_extract: %s' % grid_lon[ind_lon])
        print('lat_extract: %s' % grid_lat[ind_lat])
        phase = fes_data.variables['phase'][:]
        amplitude = fes_data.variables['amplitude'][:]
        phase_extract = phase[ind_lat, ind_lon]
        amplitude_extract = amplitude[ind_lat, ind_lon]
        print('phase_extract: %s' %phase_extract)
        print('amplitude_extract: %s' %amplitude_extract)
        return amplitude_extract, phase_extract


def get_closest_grid_point(grid_lon, grid_lat, lon, lat):
    diffs_lon = np.abs(grid_lon - lon)
    diffs_lat = np.abs(grid_lat - lat)
    ind_lon = np.where(diffs_lon == np.min(diffs_lon))[0][0]
    ind_lat = np.where(diffs_lat == np.min(diffs_lat))[0][0]
    print('lon lat grid extract at: lon=%s, lat=%s' % (grid_lon[ind_lon], grid_lat[ind_lat]))
    print('lon lat grid extract at: %s, %s' % (ind_lon, ind_lat))

    return ind_lon, ind_lat


# Figueirette
location = 'La Figueirette'
lon_extraction = 6.93494
lat_extraction = 43.4835
f_constituents_extracted = '/home/florent/Projects/Cannes/Water_levels/tide_from_harmonic_constituents/\
tide_constituents_extracted_from_fes_2014.txt'

fes_2014_constituents = ['M3', 'M4', 'M6', 'M8', 'K1', 'O1', 'Q1', 'P1', 'S1', 'J1', 'M2', 'S2', 'N2', 'K2', 'L2',
                         'LAMBDA2', 'EPS2', 'R2', '2N2', 'MU2', 'NU2', 'T2', 'MKS2', 'MS4', 'MN4', 'N4', 'S4', 'MF',
                         'MSF', 'MM', 'MTM', 'MSQM', 'SSA', 'SA']

path_fes_2014_data = r'/home/florent/ownCloud/R&D/DATA/TIDE/FES_2014/FES2014_b_elevations_extrapolated/ocean_tide_extrapolated/'

with open(f_constituents_extracted, 'w') as f_out:
    f_out.write('Tide harmonic constituents extracted from FES 2014 at {location}: (name, amplitude, phase)\n'.format(
        location=location))
    for i, constituent in enumerate(fes_2014_constituents):
        fname_constituent = get_fes_constituent_fname(constituent, path_fes_2014_data)
        amplitude, phase = get_fes_2014_data_at_extraction_point(fname_constituent, lon_extraction, lat_extraction)
        f_out.write('%s %s %s\n' % (constituent, amplitude, phase))
