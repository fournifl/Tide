from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate
from copy import copy

import json
from plots import regression_plots_interp_all_sources_on_ref, regression_plots_interp_all_sources_on_biggest_timestep, \
    plot_timely_differences, plot_taylor_diagrams, plot_water_levels
from read_water_levels import read_tide_tg_zh_to_other_ref, read_tide_from_fes, read_csv_wl_shom, read_ncdf_wl_cmems, \
    read_ncdf_wl_lops_at_harbour
from simple_utils import convert_datetime64_array_to_datetime_array, select_data_inside_study_period, str_to_dtm, \
    datetime_range
from data_processing import interp_all_water_levels_on_regular_time_array
from spotter import read_spotter_pressure, convert_p_spotter_and_p_atmo_to_water_height, get_synop_station_infos, \
    read_pressure_station_synop


def find_and_apply_phase_difference_between_water_levels(water_levels, key_ref=None, t_delta_in_s=None):
    phase_difference = {}
    t_array = datetime_range(min(water_levels[key_ref]['dates']), max(water_levels[key_ref]['dates']),
                             timedelta(seconds=t_delta_in_s))
    water_levels_interp = interp_all_water_levels_on_regular_time_array(water_levels, t_array)
    # water_levels_interp = interp_all_water_levels_on_specified_one(water_levels, key_ref)
    water_level_interp_ref = water_levels_interp[key_ref]
    for i, key in enumerate(water_levels.keys()):
        if key != key_ref:
            print(key)
            water_level_interp = water_levels_interp[key]
            # indices without nan
            inds_no_nan = (~ np.isnan(water_level_interp_ref)) * (~ np.isnan(water_level_interp))
            inds_no_nan = np.where(inds_no_nan)[0]
            A = water_levels_interp[key_ref][inds_no_nan]
            B = water_level_interp[inds_no_nan]
            nsamples = A.size
            dates_no_nan = np.array(t_array)[inds_no_nan]
            # regularize datasets by subtracting mean and dividing by s.d.
            A -= A.mean()
            A /= A.std()
            B -= B.mean()
            B /= B.std()
            # Find cross-correlation
            xcorr = correlate(A, B)
            # delta time array to match xcorr
            dt = np.arange(1 - nsamples, nsamples)
            recovered_time_shift = dt[xcorr.argmax()]
            # apply phase difference
            phase_difference = recovered_time_shift * t_delta_in_s
            print('time shift correction: %s s' %phase_difference)
            water_levels[key]['dates'] = np.array(water_levels[key]['dates']) + timedelta(seconds=int(phase_difference))
    return phase_difference, water_levels


# OPTIONS D'EXECUTION, AFFICHAGE
options = dict(
    read_despiked_tide=0,
)

# json parameters
with open('compare_tg_fes_models/json/compare_tg_fes_models_etretat.json', 'r') as myfile:
    data = myfile.read()
settings = json.loads(data)
location = settings['location']
t_start = str_to_dtm(settings['start'])
t_end = str_to_dtm(settings['end'])
f_tide_from_fes = settings['f_fes']
lon_extract = settings['lon_extract_models']
lat_extract = settings['lat_extract_models']
ZH_to_IGN69 = settings['ZH_to_IGN69']
NM_to_ZH = settings['NM_to_ZH']
NM_to_IGN69 = NM_to_ZH + ZH_to_IGN69
dir_wl_cmems = settings['dir_model_cmems']
dir_wl_lops = settings['dir_model_lops']
dir_plots = settings['dir_plots']
png_out = settings['png_out'].format(dir_plots=dir_plots, location=location)
rho = settings['rho']
g = settings['g']

# dictinnary containing all water levels types
water_levels = {}

# tg
if settings['tg_available']:
    water_levels['tg'] = {}
    f_tg = settings['tg_file']
    dates_tg, water_level_tg = read_tide_tg_zh_to_other_ref(f_tg, ZH_to_IGN69)
    dates_tg, water_level_tg = select_data_inside_study_period(t_start, t_end, np.array(dates_tg),
                                                               np.array(water_level_tg))
    water_levels['tg']['color'] = 'navy'
    water_levels['tg']['dates'] = dates_tg
    water_levels['tg']['water_level'] = water_level_tg
    water_levels['tg']['label'] = 'Tide gauge water level'

# fes
if settings['fes_available']:
    water_levels['fes'] = {}
    dates_fes, tide_height_fes = read_tide_from_fes(f_tide_from_fes, NM_to_IGN69)
    dates_fes = convert_datetime64_array_to_datetime_array(dates_fes)
    dates_fes, tide_height_fes = select_data_inside_study_period(t_start, t_end, np.array(dates_fes),
                                                                 np.array(tide_height_fes))
    water_levels['fes']['color'] = 'dodgerblue'
    water_levels['fes']['dates'] = dates_fes
    water_levels['fes']['water_level'] = tide_height_fes
    water_levels['fes']['label'] = 'FES2014 tide'

# model shom
if settings['model_shom_available']:
    f_shom = settings['f_shom']
    dates_hycom_shom, water_level_hycom_shom = read_csv_wl_shom(f_shom, NM_to_IGN69)
    water_levels['model_shom'] = {}
    water_levels['model_shom']['color'] = 'darkblue'
    water_levels['model_shom']['dates'] = dates_hycom_shom
    water_levels['model_shom']['water_level'] = water_level_hycom_shom
    water_levels['model_shom']['label'] = 'model shom'

# model cmems
if settings['model_cmems_available']:
    water_levels['model_cmems'] = {}
    water_levels['model_cmems']['color'] = 'm'
    dates_cmems, water_level_cmems = read_ncdf_wl_cmems(dir_wl_cmems, NM_to_IGN69, lon_extract, lat_extract)
    water_levels['model_cmems']['dates'] = dates_cmems
    water_levels['model_cmems']['water_level'] = water_level_cmems
    water_levels['model_cmems']['label'] = 'cmems'

# model lops
if settings['model_lops_available']:
    water_levels['model_lops'] = {}
    water_levels['model_lops']['color'] = 'y'
    dates_lops, water_level_lops = read_ncdf_wl_lops_at_harbour(dir_wl_lops, NM_to_IGN69, t_start, t_end)
    water_levels['model_lops']['dates'] = dates_lops
    water_levels['model_lops']['water_level'] = water_level_lops
    water_levels['model_lops']['label'] = 'lops MARS2D'

# spotter pressure
if settings['spotter_pressure_available']:
    # get atmospheric pressure to correct spotter pressure sensor
    id_synop, station_name, station_lat, station_lon = get_synop_station_infos(settings['location_synop'],
                                                                               settings['f_ids_stations_synop'])
    print('id synop for %s: %s' % (location, id_synop))
    date_synop, pressure_station_synop, pressure_mer_synop = read_pressure_station_synop(id_synop, settings[
        'dir_pressure_atmo_synop'])
    # read spotter pressure sensor, convert it to water level
    water_levels['spotter'] = {}
    f_spotter = settings['f_spotter']
    water_levels['spotter']['color'] = 'g'
    shift_spotter_z = 8.64
    # shift_spotter_z = 0.0
    dates_spotter, p_spotter, dates_std_spotter, std_wl_spotter = read_spotter_pressure(f_spotter)
    # apply atmospheric pressure correction on spotter's pressure sensor
    water_height_spotter = convert_p_spotter_and_p_atmo_to_water_height(dates_spotter, p_spotter, date_synop,
                                                                        pressure_mer_synop, shift_spotter_z, dir_plots,
                                                                        rho, g)
    water_levels['spotter']['dates'] = dates_spotter
    # water_levels['spotter']['water_level_raw'] = wl_spotter
    water_levels['spotter']['water_level'] = water_height_spotter
    water_levels['spotter']['label'] = 'spotter'

# timesteps
# spotter: 30mn
# fes: 10mn
# lops: 15mn
# shom: 60mn

# define reference water level
key_ref = 'spotter'
key_biggest_timestep = 'model_shom'

# time plot water levels
plot_water_levels(water_levels, png_out, 'IGN69', location, 7, key_ref)

# regression plots
regression_plots_interp_all_sources_on_ref(water_levels, dir_plots, key_ref='spotter')
regression_plots_interp_all_sources_on_biggest_timestep(water_levels, dir_plots, key_ref=key_ref,
                                                        key_biggest_timestep=key_biggest_timestep)
# timely differences
plot_timely_differences(water_levels, dir_plots, key_ref=key_ref, key_biggest_timestep=key_biggest_timestep)

# taylor diagrams
plot_taylor_diagrams(water_levels, dir_plots, key_ref=key_ref, key_biggest_timestep=key_biggest_timestep)


## study with phase correction

print('\n results with phase correction:')
# find phase difference between water levels and ref water level
phase_differences_in_s, water_levels = find_and_apply_phase_difference_between_water_levels(water_levels,
                                                                                            key_ref=key_ref,
                                                                                            t_delta_in_s=60)

# regression plots
regression_plots_interp_all_sources_on_biggest_timestep(water_levels, dir_plots, key_ref=key_ref,
                                                        key_biggest_timestep=key_biggest_timestep, phase_corrected=True)

# timely differences
plot_timely_differences(water_levels, dir_plots, key_ref=key_ref, key_biggest_timestep=key_biggest_timestep,
                        phase_corrected=True)

# taylor diagrams
plot_taylor_diagrams(water_levels, dir_plots, key_ref=key_ref, key_biggest_timestep=key_biggest_timestep,
                     phase_corrected=True)

# plot water levels
# vertical_ref = 'IGN69'
# t_interval = 7
# plot_water_levels(water_levels, png_out, vertical_ref, location, t_interval)
