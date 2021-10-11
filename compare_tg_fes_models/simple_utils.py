import datetime

import netCDF4
import numpy as np


def get_lon_lat_and_indices_to_extract_ncdf(f_cmems, lon_extract, lat_extract):
    h = netCDF4.Dataset(f_cmems)
    longitude = h.variables['longitude'][:]
    latitude = h.variables['latitude'][:]
    ind_lon = np.where(abs(longitude - lon_extract) == min(abs(longitude - lon_extract)))[0][0]
    ind_lat = np.where(abs(latitude - lat_extract) == min(abs(latitude - lat_extract)))[0][0]
    return longitude, latitude, ind_lon, ind_lat


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


def str_to_dtm(date_str):
    if '_' in date_str:
        ymd_hm = date_str
        date = datetime.datetime(np.int(ymd_hm[0:4]), np.int(ymd_hm[4:6]), np.int(ymd_hm[6:8]), np.int(ymd_hm[9:11]),
                                 np.int(ymd_hm[11:13]))
    else:
        ymd = date_str
        date = datetime.datetime(np.int(ymd[0:4]), np.int(ymd[4:6]), np.int(ymd[6:8]))

    return date


def datetime_range(start, end, step):
    """like range() for datetime"""
    return [start + i * step for i in range((end - start) // step)]
