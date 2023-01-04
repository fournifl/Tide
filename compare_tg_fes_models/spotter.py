import datetime
import glob
from glob import glob
from os.path import join as join

import matplotlib.pyplot as plt
import numpy as np


def read_spotter_pressure(f_spotter):
    dates_spotter = []
    # water_depth_spotter = []
    std_water_depth_spotter = []
    dates_spotter_std = []
    pressure_spotter = []
    with open(f_spotter, 'r') as f_in:
        lines = f_in.readlines()[1:]
        for row in lines:
            data_type = row.split(',')[-3]
            if 'meanpressure' in data_type:
                date_str = ((row.split(',')[0]).split('Z')[0]).split('.')[0]
                date = datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
                dates_spotter.append(date)
                pressure_tmp = int(row.split(',')[-1].split('\n')[0]) * 1e-1
                pressure_spotter.append(pressure_tmp)
                # water_depth_spotter.append(pressure_tmp * 10.0 - shift_spotter_z)
            if 'stdevpressure' in data_type:
                date_str = ((row.split(',')[0]).split('Z')[0]).split('.')[0]
                date = datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
                dates_spotter_std.append(date)
                std = int(row.split(',')[-1].split('\n')[0]) * 1e-6
                std_water_depth_spotter.append(std)
    # return dates_spotter[::-1], pressure_spotterwater_depth_spotter[::-1], dates_spotter_std[::-1], std_water_depth_spotter[::-1]
    return dates_spotter[::-1], pressure_spotter[::-1], dates_spotter_std[::-1], std_water_depth_spotter[::-1]


def convert_p_spotter_and_p_atmo_to_water_height(dates_spotter, p_spotter, date_synop, pressure_mer_synop,
                                                 shift_spotter_z, dir_plots, rho, g):
    indate = [(t - datetime.datetime(1970, 1, 1)).total_seconds() for t in date_synop]
    outdate = [(t - datetime.datetime(1970, 1, 1)).total_seconds() for t in dates_spotter]
    pressure_mer_synop_interp = np.interp(np.array(outdate), np.array(indate), pressure_mer_synop)
    water_depth_spotter = []
    # water_level_spotter_corrected = []
    # p_mer_moyenne = 1013
    # for i, wl in enumerate(water_level_spotter):
    #     wl_corrected = wl - (pressure_mer_synop_interp[i] / 100 - p_mer_moyenne) * 0.01
    #     water_level_spotter_corrected.append(wl_corrected)
    # return water_level_spotter_corrected
    # 1 bar = 1000hPa, donc 1 bar = 1.e5 Pa, soit 1µbar = 0.1 Pa
    for i, p in enumerate(p_spotter):
        h = (p - pressure_mer_synop_interp[i]) / (rho * g) - shift_spotter_z
        water_depth_spotter.append(h)
    fig, ax = plt.subplots(1, 1, figsize=(22, 10))
    ax.plot(dates_spotter, np.array(water_depth_spotter) + shift_spotter_z, label='water height spotter')
    ax.grid(True)
    ax.set_xlim([min(dates_spotter), max(dates_spotter)])
    ax.legend()
    fig.savefig(join(dir_plots, 'water_height_spotter.jpg'))

    return(water_depth_spotter)


def get_synop_station_infos(location, f_ids_stations_synop):
    f_ids = open(f_ids_stations_synop, 'r')
    lines = f_ids.readlines()
    lines = lines[1:]
    for line in lines:
        if location.upper() in line:
            id = line.split(';')[0]
            station_name = line.split(';')[1]
            lat = float(line.split(';')[2])
            lon = float(line.split(';')[3])
    return id, station_name, lat, lon


def convert_date_synop_to_datetime(date_synop):
    date = datetime.datetime.strptime(date_synop, '%Y%m%d%H%M%S')
    return date


def read_pressure_station_synop(id_station, dir_synop):
    ls_synop = np.sort(glob(dir_synop + '*.csv'))
    date = []
    pressure_mer = []
    pressure_station = []
    for f_synop in ls_synop:
        with open(f_synop, 'r') as f_in:
            lines = f_in.readlines()
            for line in lines:
                if line.split(';')[0] == id_station:
                    date_tmp = line.split(';')[1]
                    date_tmp = convert_date_synop_to_datetime(date_tmp)
                    pressure_mer_tmp = line.split(';')[2]
                    pressure_station_tmp = line.split(';')[20]
                    date.append(date_tmp)
                    pressure_mer.append(int(pressure_mer_tmp))
                    pressure_station.append(int(pressure_station_tmp))
    # fig, ax = plt.subplots(2, 1, figsize=(22, 10))
    # ax[0].axhline(y=1013, linewidth=1, color='k', dashes=(4, 4))
    # ax[0].plot(np.array(date), np.array(pressure_mer) / 100, label='P mer')
    # ax[0].grid(True)
    # ax[0].set_ylabel('P (hPa)', fontsize=12)
    # ax[0].set_title('PRESSURE AT SEA LEVEL')
    # ax[0].set_xlim([min(date), max(date)])
    # ax[0].legend()
    # ax[1].axhline(y=0, linewidth=1, color='k', dashes=(4, 4))
    # ax[1].plot(np.array(date), np.array(pressure_mer) / 100 - 1013, label='surge', color='g')
    # ax[1].set_ylabel('Surge (cm)', fontsize=12)
    # ax[1].set_title('ATMOSPHERIC SURGE')
    # ax[1].set_xlim([min(date), max(date)])
    # ax[1].grid(True)
    # ax[1].legend()
    # plt.show()
    return date, pressure_station, pressure_mer
