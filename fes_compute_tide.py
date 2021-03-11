#!/usr/bin/env python3
# This file is part of FES library.
#
#  FES is free software: you can redistribute it and/or modify
#  it under the terms of the GNU LESSER GENERAL PUBLIC LICENSE as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  FES is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU LESSER GENERAL PUBLIC LICENSE for more details.
#
#  You should have received a copy of the GNU LESSER GENERAL PUBLIC LICENSE
#  along with FES.  If not, see <http://www.gnu.org/licenses/>.
"""
Example of using the FES Python interface
"""
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pyfes
import pdb


def usage():
    """
    Command syntax
    """
    parser = argparse.ArgumentParser(
        description='Program example using the Python API for FES.')
    parser.add_argument('ocean',
                        help='Path to the configuration file that contains '
                             'the defintion of grids to use to compute the '
                             'ocean tide',
                        type=argparse.FileType('r'))
    parser.add_argument('--load',
                        help='Path to the configuration file that contains '
                             'the defintion of grids to use to compute the '
                             'load tide',
                        type=argparse.FileType('r'))
    return parser.parse_args()


def get_closest_grid_point(lons, lats, lon, lat):
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    diffs_lon = np.abs(lons - lon)
    diffs_lat = np.abs(lats - lat)
    ind_lon = np.where(diffs_lon == np.min(diffs_lon))[0][0]
    ind_lat = np.where(diffs_lat == np.min(diffs_lat))[0][0]
    lon_extract = grid_lon[ind_lat, ind_lon]
    lat_extract = grid_lat[ind_lat, ind_lon]
    return ind_lon, ind_lat, lon_extract, lat_extract


def get_fes_wl_at_extraction_point(lons, lats, lon_extraction, lat_extraction, geo_tide):
    ind_lon, ind_lat, lon_extract, lat_extract = get_closest_grid_point(lons, lats, lon_extraction,
                                                                        lat_extraction)
    tide_extract = geo_tide[ind_lat, ind_lon]
    return tide_extract, lon_extract, lat_extract


def compute_dates(start_date, end_date, step):
    vec_dates = [start_date]
    date_i = start_date
    while date_i < end_date:
        date_i += np.timedelta64(step, 'm')
        vec_dates.append(date_i)
    return vec_dates


def main():
    """
    Main program
    """
    "export FES_DATA='/home/florent/ownCloud/R&D/DATA/TIDE/FES_2014/FES2014_b_elevations_extrapolated/ocean_tide_extrapolated/'"
    f_tide_out = '/home/florent/Projects/Cannes/Water_levels/tide_from_harmonic_constituents/tide_from_fes_constituents.pk'
    tide_results = {}
    tide_results['tide_from_fes'] = []
    args = usage()
    visu = False

    # study location
    location = 'La Figueirette'
    lon_study = 6.93494
    lat_study = 43.4835

    # Create handler
    short_tide = pyfes.Handler("ocean", "memory", args.ocean.name)
    if args.load is not None:
        radial_tide = pyfes.Handler("radial", "memory", args.load.name)
    else:
        radial_tide = None

    # Creating a grid that will be used to interpolate the tide
    lats = np.arange(43.4, 43.6, 0.01)
    lons = np.arange(6.8, 7.0, 0.01)
    grid_lons, grid_lats = np.meshgrid(lons, lats)
    shape = grid_lons.shape

    # dates
    start_date = np.datetime64('2020-03-11 12:00')
    end_date  = np.datetime64('2021-03-01 12:00')
    step = 10
    vec_dates = compute_dates(start_date, end_date, step)
    tide_results['dates'] = vec_dates
    dates = np.empty(shape, dtype='datetime64[us]')
    for date in vec_dates:
        dates.fill(date)

        # Calculate tide
        tide, lp, _ = short_tide.calculate(grid_lons.ravel(), grid_lats.ravel(),
                                           dates.ravel())
        tide, lp = tide.reshape(shape), lp.reshape(shape)
        if radial_tide is not None:
            load, load_lp, _ = radial_tide.calculate(grid_lons.ravel(), grid_lats.ravel(),
                                                     dates.ravel())
            load, load_lp = load.reshape(shape), load_lp.reshape(shape)
        else:
            load = np.zeros(grid_lons.shape)
            load_lp = load

        # Convert tide to cm and to a 2d numpy masked array
        geo_tide = (tide + lp + load) * 0.01
        geo_tide = geo_tide.reshape(grid_lons.shape)
        geo_tide = np.ma.masked_where(np.isnan(geo_tide), geo_tide)

        # Extract tide at study location
        tide_extract, lon_extract, lat_extract = get_fes_wl_at_extraction_point(lons, lats, lon_study,
                                                                                lat_study, geo_tide)

        # save tide results in dictionnary
        tide_results['tide_from_fes'].append(tide_extract)

        # Affichage
        if visu:
            plt.pcolormesh(grid_lons, grid_lats, geo_tide)
            plt.text(lon_extract, lat_extract, '%.3f' %tide_extract)
            plt.colorbar()
            plt.show()

    # save tide results into pickle
    with open(f_tide_out, 'wb') as file_tide_out:
        pickle.dump(tide_results, file_tide_out, protocol=2)

if __name__ == '__main__':
    main()
