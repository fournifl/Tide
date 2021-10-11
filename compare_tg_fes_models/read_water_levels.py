import datetime
import glob
import pickle
from datetime import timedelta
from glob import glob

import netCDF4
import numpy as np

from simple_utils import get_lon_lat_and_indices_to_extract_ncdf


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


def read_csv_wl_shom(f_shom, NM_to_other_z_ref):
    dates = []
    tide = []
    surge = []
    water_level = []
    with open(f_shom, 'r') as f_in:
        lines = f_in.readlines()[1:]
        for row in lines:
            date = datetime.datetime.strptime((row.split(',')[0]).split('.')[0], '%Y-%m-%d %H:%M:%S')
            dates.append(date)
            surge_tmp = float(row.split(',')[1])
            surge.append(surge_tmp)
            tide_tmp = float(row.split(',')[2]) - NM_to_other_z_ref
            tide.append(tide_tmp)
            water_level.append(tide_tmp + surge_tmp)
    return dates, water_level


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


def read_ncdf_wl_lops_at_harbour(dir_lops, NM_to_other_z_ref, t_start, t_end):
    ls_lops = glob(dir_lops + '*.nc')
    ls_lops = np.sort(ls_lops)
    time_dtm = []
    ssh = []
    t_ref = datetime.datetime(1900, 1, 1)
    for f_lops in ls_lops:
        h = netCDF4.Dataset(f_lops)
        time = h.variables['time'][:]
        t0_f_lops = t_ref + timedelta(seconds=int(time[0]))
        if (t0_f_lops > t_start) * (t0_f_lops < t_end):
            for t in time:
                time_dtm.append(t_ref + timedelta(seconds=int(t)))
            ssh_tmp = h.variables['XE'][:]
            ssh_tmp = np.reshape(ssh_tmp, ssh_tmp.shape[0])
            surge_tmp = h.variables['DELTA_XE'][:]
            surge_tmp = np.reshape(surge_tmp, surge_tmp.shape[0])
            ssh_tmp += surge_tmp
            ssh_tmp -= NM_to_other_z_ref
            for s in ssh_tmp.data:
                ssh.append(s)
    return time_dtm, ssh
