import datetime
import glob
import pickle
from datetime import timedelta

import netCDF4
from netCDF4 import Dataset
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pdb


def compute_period(start_date, end_date, t_step):
    period = []
    start = start_date
    n = 0
    end = start_date + n * timedelta(t_step)
    while end < end_date:
        end = start_date + n * timedelta(t_step)
        period.append(end)
        n += 1
    return period


def read_tide_tg_zh_to_other_ref(tide_file, other_z_ref):
    dates = []
    water_level = []
    shift_z = other_z_ref
    with open(tide_file, 'r') as file:
        lines = file.readlines()[1:]
        for row in lines:
            date = datetime.datetime.strptime(row.split(',')[0], '%Y-%m-%d %H:%M:%S')
            # remove eventual seconds (donnees brutes HF)
            date = date - timedelta(seconds=date.second)
            dates.append(date)
            water_level.append(float(row.split(',')[1]) - shift_z)
    return dates, water_level


def read_tide_from_fes(f_tide_from_fes, NM_to_other_z_ref):
    with open(f_tide_from_fes, 'rb') as file_tide_from_fes:
        tide = pickle.load(file_tide_from_fes)
        dates_fes = tide['dates']
        tide_height_fes = np.array(tide['tide_from_fes'])
        tide_height_fes -= NM_to_other_z_ref
        return dates_fes, tide_height_fes


def get_lon_lat_and_indices_to_extract_ncdf(f_cmems, lon_extract, lat_extract):
    h = netCDF4.Dataset(f_cmems)
    longitude = h.variables['longitude'][:]
    latitude = h.variables['latitude'][:]
    ind_lon = np.where(abs(longitude - lon_extract) == min(abs(longitude - lon_extract)))[0][0]
    ind_lat = np.where(abs(latitude - lat_extract) == min(abs(latitude - lat_extract)))[0][0]
    return longitude, latitude, ind_lon, ind_lat


def read_ncdf_wl_cmems(dir_cmems, geoid_to_other_z_ref, lon_extract, lat_extract):
    ls_cmems = glob.glob(dir_cmems + '*.nc')
    ls_cmems = np.sort(ls_cmems)
    time_dtm = []
    ssh = []
    t_ref = datetime.datetime(1950, 1, 1)
    counter = 0
    for f_cmems in ls_cmems:
        if counter == 0:
            longitude, latitude, ind_lon, ind_lat = get_lon_lat_and_indices_to_extract_ncdf(f_cmems,
                                                                                            lon_extract, lat_extract)
        h = netCDF4.Dataset(f_cmems)
        time = h.variables['time'][:]
        for t in time:
            time_dtm.append(t_ref + timedelta(seconds=int(t * 3600)))
        ssh_tmp = h.variables['zos'][:, ind_lat, ind_lon]
        ssh_tmp -= geoid_to_other_z_ref
        for s in ssh_tmp.data:
            ssh.append(s)
        counter += 1
    return time_dtm, ssh


def convert_datetime64_array_to_datetime_array(datetime64_array):
    date_dtm = []
    for date in datetime64_array:
        date_dtm.append(date.astype(datetime.datetime))
    return date_dtm


def select_data_inside_study_period(t_start, t_end, dates, data):
    inds = np.where((dates > t_start) * (dates < t_end))[0]
    dates_out = dates[inds]
    data_out = data[inds]
    return dates_out, data_out


def oversample_array_with_constant_tstep(arr, dates, dates_oversampling):
    arr_tstep_constant = []
    for i in range(len(dates_oversampling)):
        ind_dates = [j for j, value in enumerate(dates) if value == dates_oversampling[i]]
        if len(ind_dates) > 0:
            arr_tstep_constant.append(arr[ind_dates[0]])
        else:
            arr_tstep_constant.append(np.nan)
    print('ouhba')
    return np.array(arr_tstep_constant)


def plot_water_levels(dates_tg, water_level_tg, dates_fes, tide_height_fes, dates_cmems,
                      water_level_cmems, png_out, vertical_ref, location, t_interval):
    f, ax = plt.subplots(1, figsize=(28, 6), sharex=True)
    ax.plot(dates_tg, water_level_tg, color='navy', markersize=2, label='Tide gauge water level')
    ax.plot(dates_fes, tide_height_fes, color='dodgerblue', markersize=2, label='FES2014 tide')

    ax.set_ylabel('Water level (m)', fontsize=16)
    ax.set_title(
        'WATER LEVEL RELATIVE TO {vertical_ref} AT {loc}'.format(vertical_ref=vertical_ref, loc=location.upper()),
        fontsize=16)
    ax.grid(True)
    ax.plot(dates_cmems, water_level_cmems, color='m', markersize=2, label='cmems')
    ax.axhline(y=0, linewidth=2, color='gray', dashes=(4, 4))
    ax.legend(loc='upper right', fontsize=10, framealpha=0.6)
    ax.set_xlim([min(dates_fes), max(dates_fes)])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d %H'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=t_interval))
    f.autofmt_xdate()
    plt.show()
    f.savefig(png_out)
    plt.close()


def str_to_dtm(date_str):
    if '_' in date_str:
        ymd_hm = date_str
        date = datetime.datetime(np.int(ymd_hm[0:4]), np.int(ymd_hm[4:6]), np.int(ymd_hm[6:8]), np.int(ymd_hm[9:11]),
                        np.int(ymd_hm[11:13]))
    else:
        ymd = date_str
        date = datetime.datetime(np.int(ymd[0:4]), np.int(ymd[4:6]), np.int(ymd[6:8]))

    return date


# OPTIONS D'EXECUTION, AFFICHAGE
options = dict(
    read_despiked_tide=0,
)

# json parameters
with open('compare_tg_fes_models.json', 'r') as myfile:
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
dir_wl_cmems = settings['dir_model_cmems']
png_out = settings['png_out']

if settings['tg_available']:
    f_tg = settings['tg_file']
else:
    f_tg = None
if settings['model_shom_available']:
    f_shom = settings['f_shom']
else:
    f_shom = None
if settings['model_cmems_available']:
    dir_cmems = settings['dir_model_cmems']
else:
    dir_cmems = None

# tg
dates_tg, water_level_tg = read_tide_tg_zh_to_other_ref(f_tg, ZH_to_IGN69)
dates_tg, water_level_tg = select_data_inside_study_period(t_start, t_end, np.array(dates_tg),
                                                           np.array(water_level_tg))

# fes
NM_to_IGN69 = NM_to_ZH + ZH_to_IGN69
dates_fes, tide_height_fes = read_tide_from_fes(f_tide_from_fes, NM_to_IGN69)
dates_fes = convert_datetime64_array_to_datetime_array(dates_fes)
dates_fes, tide_height_fes = select_data_inside_study_period(t_start, t_end, np.array(dates_fes),
                                                             np.array(tide_height_fes))

# cmems
geoide_to_IGN69 = NM_to_ZH + ZH_to_IGN69
# geoide_to_IGN69 = -1.0
dates_cmems, water_level_cmems = read_ncdf_wl_cmems(dir_wl_cmems, geoide_to_IGN69, lon_extract, lat_extract)

# plot
vertical_ref = 'IGN69'
t_interval = 7
plot_water_levels(dates_tg, water_level_tg, dates_fes, tide_height_fes, dates_cmems, water_level_cmems, png_out, vertical_ref, location, t_interval)
