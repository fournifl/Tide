
import numpy as np
import pdb
import matplotlib.pyplot as plt
import pickle
import datetime
from datetime import timedelta
from netCDF4 import Dataset
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def convert_list_dt64_to_datetime(my_list):
    list_dtm = []
    for dt64 in my_list:
        ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        list_dtm.append(datetime.datetime.utcfromtimestamp(ts))
    return list_dtm


def select_data_inside_study_period(t_start, t_end, dates, data):
    inds = np.where((dates > t_start) * (dates < t_end))[0]
    dates_out = dates[inds]
    data_out = data[inds]
    return dates_out, data_out


def read_tide_from_fes(f_tide_from_fes, NM_to_IGN69):
    with open(f_tide_from_fes, 'rb') as file_tide_from_fes:
        tide = pickle.load(file_tide_from_fes)
        dates_fes = tide['dates']
        dates_fes = convert_list_dt64_to_datetime(dates_fes)
        tide_height_fes = np.array(tide['tide_from_fes'])
        tide_height_fes = tide_height_fes - NM_to_IGN69
        return dates_fes, tide_height_fes
        
def read_tide_shom_ncdf_NM_to_IGN69(tide_file, NM_to_ZH, ZH_to_IGN69):
    shift_IGN69 = NM_to_ZH + ZH_to_IGN69
    t_ref = datetime.datetime(1950, 1, 1)
    h = Dataset(tide_file)
    surge = h.variables['surge'][:]
    ssh = h.variables['ssh'][:]
    time = h.variables['time'][:]
    water_level = ssh + surge - shift_IGN69
    time_dtm = []
    for t in time:
        time_dtm.append(t_ref + timedelta(t))
    return time_dtm, ssh, surge, water_level


def plot_diff_wl(dates_shom, water_level_shom, tide_height_fes, surge_shom, png_out):
    diff_wl = water_level_shom - tide_height_fes
    tide_shom = water_level_shom - surge_shom
    diff_tide = tide_shom - tide_height_fes
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[1., 1.],
        specs=[[{"type": "scatter"}],
               [{"type": "scatter"}]],
        subplot_titles=("WATER LEVELS", "DIFF WATER LEVEL SHOM vs FES"))
    
    fig.add_trace(
            go.Scatter(x=dates_shom, y=water_level_shom,
            mode='lines', line=dict(color='lightskyblue', width=1),
            name='Water level model Shom', showlegend=True), row=1, col=1
        )
    fig.add_trace(
            go.Scatter(x=dates_shom, y=tide_height_fes,
            mode='lines', line=dict(color='yellow', width=1),
            name='Water level FES', showlegend=True), row=1, col=1
        )
    fig.add_trace(
        go.Scatter(x=dates_shom, y=diff_tide * 0.0,
                   mode='lines', line=dict(color='white', width=1, dash='dash'),
                   showlegend=False), row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates_shom, y=diff_tide,
                   mode='lines', line=dict(color='orange', width=2),
                   name='Diff Tide model_Shom vs FES', showlegend=True), row=2, col=1
    )
    fig.add_trace(
            go.Scatter(x=dates_shom, y=diff_wl,
            mode='lines', line=dict(color='crimson', width=2),
            name='Diff Water level model_Shom vs FES', showlegend=True), row=2, col=1
    )
    fig.update_layout(
        template="plotly_dark",
        width=1800,
        height=600,
        legend=dict(orientation="v",
                    yanchor="top",
                    xanchor="right",
                    y=0.57,
                    x=0.95),
    )
    fig.update_annotations(font=dict(color='white'))
    # ~ fig.show()
    fig.write_image(png_out)

NM_to_ZH = -4.80
ZH_to_IGN69 = 4.367
NM_to_IGN69 = NM_to_ZH + ZH_to_IGN69

file_wl_shom = '/home/florent/Projects/Etretat_1er_contrat/Water_levels/NIVEAUX_HYCOM2D_R1000_ATL-CG-ETRETAT.nc'
file_wl_fes = '/home/florent/Projects/Etretat/Water_levels/tide_from_harmonic_constituents/tide_from_fes_constituents_Etretat_2018_01_01_2023_01_01.pk'

dates_shom, tide_shom, surge_shom, water_level_shom = read_tide_shom_ncdf_NM_to_IGN69(file_wl_shom, NM_to_ZH, ZH_to_IGN69)
dates_fes, tide_height_fes = read_tide_from_fes(file_wl_fes, NM_to_IGN69)

# time ranges comparison Lidar
x_range = [datetime.datetime(2019, 8, 24), datetime.datetime(2019, 9, 2, 20)]
# x_range = [datetime.datetime(2020, 1, 5), datetime.datetime(2020, 1, 14, 16)]

# select data inside study period
dates_shom_tmp, water_level_shom = select_data_inside_study_period(x_range[0], x_range[1], np.array(dates_shom), np.array(water_level_shom))
dates_fes, tide_height_fes = select_data_inside_study_period(x_range[0], x_range[1], np.array(dates_fes), np.array(tide_height_fes))
dates_shom, surge_shom = select_data_inside_study_period(x_range[0], x_range[1], np.array(dates_shom), np.array(surge_shom))

# interpolate fes on shom dates
indate = [(t - datetime.datetime(1970, 1, 1)).total_seconds() for t in dates_fes]
outdate = [(t - datetime.datetime(1970, 1, 1)).total_seconds() for t in dates_shom]
tide_height_fes  = np.interp(np.array(outdate), np.array(indate), tide_height_fes)

png_out = '/home/florent/tmp/tmp/slope_wl_comparison_shom_fes_{}_{}.png'.format(x_range[0].strftime('%Y%m%d'), x_range[1].strftime('%Y%m%d'))
plot_diff_wl(dates_shom, water_level_shom, tide_height_fes, surge_shom, png_out)
