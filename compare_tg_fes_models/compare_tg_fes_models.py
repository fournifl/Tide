import datetime
import glob
import pickle
from datetime import timedelta

import netCDF4
from glob import glob
import scipy
from scipy.signal import correlate
from scipy.stats import gaussian_kde

import skill_metrics as sm
from sklearn.metrics import mean_squared_error
import numpy as np
from os.path import join as join
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


def plotly_2d_scatter_plot(x, y, z, dates):
    import plotly
    import plotly.graph_objects as go
    fig = go.Figure(data=go.Scatter(
        x=[x[0]],
        y=[y[0]],
        mode='markers',
        showlegend=False,
        marker=dict(
            size=1.5,
            color=z[0],
            colorscale='Viridis',
            showscale=True
        )
    ))
    n = len(x)
    for i in range(len(x)):
        fig.add_trace(go.Scatter(
            x=[x[i]],
            y=[y[i]],
            name=dates[i].strftime('%Y%m%d %H:M'),
            showlegend=False,
            mode='markers',
            marker=dict(size=3, opacity=1.0, color=z[i]),
        ))
    fig.show()


def regression_plots_interp_all_sources_on_ref(water_levels, dir_plots, key_ref=None, phase_corrected=False):
    water_levels_interp = interp_all_water_levels_on_specified_one(water_levels, key_ref)
    water_level_ref_interp = water_levels_interp[key_ref]

    water_level_ref = water_levels[key_ref]
    fig, ax = plt.subplots(1, len(water_levels.keys()) - 1, figsize=(22, 10))
    for i, key in enumerate( water_levels.keys()):
        if key != key_ref:
            water_level_interp = water_levels_interp[key]
            # Calculate the point density
            x = np.array(water_level_ref['water_level'])
            y = water_level_interp
            xy = np.vstack([water_level_ref['water_level'], water_level_interp])
            z = gaussian_kde(xy)(xy)

            # Sort the points by density, so that the densest points are plotted last
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]

            # calculate r2
            r2 = scipy.stats.pearsonr(x, y)
            # regression
            from statsmodels.formula.api import ols
            regression = ols("data ~ x", data=dict(data=y, x=x)).fit()
            ax[i].scatter(x, y, c=z, s=50, label='water levels: %s vs %s' %(key, key_ref))
            ax[i].text(min(x) + 0.15 * (max(x) - min(x)), min(y) + 0.8 * (max(y) - min(y)),
                       'y = %.3f x + %.3f \n R²=%.3f' %(regression.params[1], regression.params[0], r2[0]),  color='b',
                       backgroundcolor='white')
            ax[i].set_xlabel(key_ref)
            ax[i].set_ylabel(key)
            ax[i].legend()
    if ~ phase_corrected:
        info_jpg = ""
    else:
        info_jpg = "_phase_corrected"
    fig.savefig(join(dir_plots, 'regression_plots_interp_all_sources_on_spotter_time_serie{info_jpg}.jpg'.format(
        info_jpg=info_jpg)))


def put_gaps_to_nan(water_level_ref_interp, indate, outdate):
    t_step = np.median(np.diff(indate))
    for i, t in enumerate(outdate):
        s_delta = abs((indate - t)).min()
        if s_delta > t_step:
            water_level_ref_interp[i] = np.nan
    return water_level_ref_interp


def interp_all_water_levels_on_specified_one(water_levels, key_to_interp_to):
    # interp all sources of water level to the dataset to interp to
    water_levels_interp = {}
    for i, key in enumerate( water_levels.keys()):
        if key != key_to_interp_to:
            indate = [(t - datetime.datetime(1970, 1, 1)).total_seconds() for t in water_levels[key]['dates']]
            outdate = [(t - datetime.datetime(1970, 1, 1)).total_seconds() for t in water_levels[key_to_interp_to]['dates']]
            water_level_interp = np.interp(np.array(outdate), np.array(indate), np.array(water_levels[key]['water_level']))
            water_level_interp = put_gaps_to_nan(water_level_interp, np.array(indate), np.array(outdate))
            water_levels_interp[key] = water_level_interp
        else:
            water_levels[key]['water_level'] = np.array(water_levels[key]['water_level'])
            water_level_interp = water_levels[key]['water_level']
            water_levels_interp[key] = water_level_interp
    return water_levels_interp


def interp_all_water_levels_on_regular_time_array(water_levels, t_array):
    # interp all sources of water level to regular time array
    outdate = [(t - datetime.datetime(1970, 1, 1)).total_seconds() for t in t_array]
    water_levels_interp = {}
    for i, key in enumerate( water_levels.keys()):
        indate = [(t - datetime.datetime(1970, 1, 1)).total_seconds() for t in water_levels[key]['dates']]
        water_level_interp = np.interp(np.array(outdate), np.array(indate), np.array(water_levels[key]['water_level']))
        water_level_interp = put_gaps_to_nan(water_level_interp, np.array(indate), np.array(outdate))
        water_levels_interp[key] = water_level_interp
        # plt.close('all')
        # plt.figure()
        # plt.plot(np.array(indate), water_levels[key]['water_level'], '+-', label='%s before interp' % key)
        # plt.plot(outdate, water_level_interp, '+-', label='%s after interp' % key)
        # plt.legend()
        # plt.show()
    return water_levels_interp


def regression_plots_interp_all_sources_on_biggest_timestep(water_levels, dir_plots, key_ref=None,
                                                            key_biggest_timestep=None, phase_corrected=False):
    water_levels_interp = interp_all_water_levels_on_specified_one(water_levels, key_biggest_timestep)
    water_level_ref_interp = water_levels_interp[key_ref]
    fig, ax = plt.subplots(1, len(water_levels.keys()) - 1, figsize=(22, 10))
    for i, key in enumerate( water_levels.keys()):
        if key != key_ref:
            water_level_interp = water_levels_interp[key]
            # indices without nan
            inds_no_nan = (~ np.isnan(water_level_ref_interp)) * (~ np.isnan(water_level_interp))
            inds_no_nan = np.where(inds_no_nan)[0]
            # plot timely differences
            x = water_level_ref_interp[inds_no_nan]
            y = water_level_interp[inds_no_nan]
            # Calculate the point density
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            # Sort the points by density, so that the densest points are plotted last
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            # calculate r2
            r2 = scipy.stats.pearsonr(x, y)
            # regression
            from statsmodels.formula.api import ols
            regression = ols("data ~ x", data=dict(data=y, x=x)).fit()
            ax[i].scatter(x, y, c=z, s=50, label='water levels: %s vs %s' %(key, key_ref))
            ax[i].text(min(x) + 0.15 * (max(x) - min(x)), min(y) + 0.8 * (max(y) - min(y)),
                       'y = %.3f x + %.3f \n R²=%.3f' %(regression.params[1], regression.params[0], r2[0]),  color='b',
                       backgroundcolor='white')
            ax[i].set_xlabel(key_ref)
            ax[i].set_ylabel(key)
            ax[i].legend()
    if not phase_corrected:
        info_jpg = ""
    else:
        info_jpg = "_phase_corrected"
    fig.savefig(join(dir_plots, 'regression_plots_interp_all_sources_on_shom_time_serie{info_jpg}.jpg'.format(
        info_jpg=info_jpg)))


def plot_timely_differences(water_levels, key_ref=None, key_biggest_timestep=None, phase_corrected=False):
    water_levels_interp = interp_all_water_levels_on_specified_one(water_levels, key_biggest_timestep)
    water_level_ref_interp = water_levels_interp[key_ref]
    for i, key in enumerate( water_levels.keys()):
        if key != key_ref:
            water_level_interp = water_levels_interp[key]
            # indices without nan
            inds_no_nan = (~ np.isnan(water_level_ref_interp)) * (~ np.isnan(water_level_interp))
            inds_no_nan = np.where(inds_no_nan)[0]
            # plot timely differences
            x = water_level_ref_interp[inds_no_nan]
            y = water_level_interp[inds_no_nan]
            dates_no_nan = np.array(water_levels[key_biggest_timestep]['dates'])[inds_no_nan]
            print("Mean Diff %s - %s = %s"%(key_ref, key, (x-y).mean()))
            print("Median Diff %s - %s = %s"%(key_ref, key, np.median(x-y)))
            fig2, ax2 = plt.subplots(1, 1, figsize=(22, 10))
            ax2.axhline(y=0, linewidth=1, color='k', dashes=(4, 4))
            ax2.plot(dates_no_nan, x - y, label='diff water level %s - %s' %(key, key_ref))
            ax2.grid(True)
            ax2.set_ylim([-1, 1])
            ax2.set_xlim([min(dates_no_nan), max(dates_no_nan)])
            ax2.legend()
            if not phase_corrected:
                info_jpg = ""
            else:
                info_jpg = "_phase_corrected"
            fig2.savefig(join(
                dir_plots, 'timely_diff_wl_{key_ref}_vs_{key}{info_jpg}.jpg'.format(key_ref=key_ref, key=key,
                                                                                    info_jpg=info_jpg)))


def plot_taylor_diagrams(water_levels, dir_plots, key_ref=None, key_biggest_timestep=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    water_levels_interp = interp_all_water_levels_on_specified_one(water_levels, key_biggest_timestep)
    # water_level_ref_taylor = water_levels_interp[key_ref]
    sdevs = []
    crmsds = []
    ccoefs = []
    labels = ['spotter']
    for key in water_levels_interp.keys():
        labels.append(key)
        # indices without nan
        inds_no_nan = (~ np.isnan(water_levels_interp[key_ref])) * (~ np.isnan(water_levels_interp[key]))
        inds_no_nan = np.where(inds_no_nan)[0]
        water_level_ref_taylor = water_levels_interp[key_ref][inds_no_nan]
        water_level_taylor = water_levels_interp[key][inds_no_nan]
        taylor_stats = sm.taylor_statistics(water_level_taylor, water_level_ref_taylor, 'data')
        sdev = np.array([taylor_stats['sdev'][0], taylor_stats['sdev'][1]])
        if len(sdevs) == 0:
            sdevs.append(sdev[0])
            sdevs.append(sdev[1])
        else:
            sdevs.append(sdev[1])
        crmsd = np.array([taylor_stats['crmsd'][0], taylor_stats['crmsd'][1]])
        if len(crmsds) == 0:
            crmsds.append(crmsd[0])
            crmsds.append(crmsd[1])
        else:
            crmsds.append(crmsd[1])
        ccoef = np.array([taylor_stats['ccoef'][0], taylor_stats['ccoef'][1]])
        if len(ccoefs) == 0:
            ccoefs.append(ccoef[0])
            ccoefs.append(ccoef[1])
        else:
            ccoefs.append(ccoef[1])
        # Mean Squared Error from numpy and sklearn.metrics -> different from crmsd !
        # RMSE = np.sqrt(np.square(np.subtract(Topo_lidar_taylor, Topo_wc_taylor)).mean())
        RMSE = np.sqrt(mean_squared_error(water_level_taylor, water_level_ref_taylor))
        crmsd[-1] = RMSE
    sm.taylor_diagram(np.array(sdevs), np.array(crmsds), np.array(ccoefs), markerLabel=labels, markerLabelColor='k',
                      styleOBS='-', colOBS='b', markerobs='o', colCOR='firebrick', widthCOR=1.0, titleOBS=key_ref)
    jpg = join(dir_plots, 'taylor_diagram_water_levels_ref_%s.jpg'%key_ref)
    plt.savefig(jpg)


def plot_water_levels(water_levels, png_out, vertical_ref, location, t_interval):
    f, ax = plt.subplots(1, figsize=(28, 6), sharex=True)
    for type_water_level in water_levels.keys():
        water_level = water_levels[type_water_level]
        ax.plot(water_level['dates'], water_level['water_level'], color=water_level['color'], markersize=2,
                label=water_level['label'])
    ax.set_ylabel('Water level (m)', fontsize=16)
    ax.set_title(
        'WATER LEVEL COMPARISON, RELATIVE TO {vertical_ref} AT {loc}'.format(vertical_ref=vertical_ref, loc=location.upper()),
        fontsize=16)
    ax.grid(True)
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
                                                 shift_spotter_z, dir_plots):
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
    return date, pressure_station, pressure_mer


def datetime_range(start, end, step):
    """like range() for datetime"""
    return [start + i * step for i in range((end - start) // step)]


def find_and_apply_phase_difference_between_water_levels(water_levels, key_ref=None, t_delta_in_s=None):
    phase_difference = {}
    t_array = datetime_range(min(water_levels[key_ref]['dates']), max(water_levels[key_ref]['dates']),
                             timedelta(seconds=t_delta_in_s))
    water_levels_interp = interp_all_water_levels_on_regular_time_array(water_levels, t_array)
    # water_levels_interp = interp_all_water_levels_on_specified_one(water_levels, key_ref)
    water_level_interp_ref = water_levels_interp[key_ref]
    for i, key in enumerate( water_levels.keys()):
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
            # plt.close('all')
            # plt.plot(dates_no_nan, A)
            # plt.plot(dates_no_nan, B)
            # plt.show()
            # Find cross-correlation
            xcorr = correlate(A, B)
            # delta time array to match xcorr
            dt = np.arange(1 - nsamples, nsamples)
            recovered_time_shift = dt[xcorr.argmax()]
            print('recovered_time_shift:', recovered_time_shift)
            # apply phase difference
            phase_difference = recovered_time_shift * t_delta_in_s
            water_levels[key]['dates'] = np.array(water_levels[key]['dates']) + timedelta(seconds=int(phase_difference))
    return phase_difference, water_levels


# OPTIONS D'EXECUTION, AFFICHAGE
options = dict(
    read_despiked_tide=0,
)

# json parameters
with open('json/compare_tg_fes_models_etretat.json', 'r') as myfile:
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
    id_synop, station_name, station_lat, station_lon = get_synop_station_infos(settings['location_synop'], settings['f_ids_stations_synop'])
    print('id synop for %s: %s' % (location, id_synop))
    date_synop, pressure_station_synop, pressure_mer_synop = read_pressure_station_synop(id_synop, settings['dir_pressure_atmo_synop'])
    # read spotter pressure sensor, convert it to water level
    water_levels['spotter'] = {}
    f_spotter = settings['f_spotter']
    water_levels['spotter']['color'] = 'g'
    shift_spotter_z = 8.64
    # shift_spotter_z = 0.0
    dates_spotter, p_spotter, dates_std_spotter, std_wl_spotter = read_spotter_pressure(f_spotter)
    # apply atmospheric pressure correction on spotter's pressure sensor
    water_height_spotter = convert_p_spotter_and_p_atmo_to_water_height(dates_spotter, p_spotter, date_synop,
                                                                        pressure_mer_synop, shift_spotter_z, dir_plots)
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

# regression plots
regression_plots_interp_all_sources_on_ref(water_levels, dir_plots, key_ref='spotter')
regression_plots_interp_all_sources_on_biggest_timestep(water_levels, dir_plots, key_ref=key_ref,
                                                        key_biggest_timestep=key_biggest_timestep)
# timely differences
plot_timely_differences(water_levels, key_ref=key_ref, key_biggest_timestep=key_biggest_timestep)

# taylor diagrams
plot_taylor_diagrams(water_levels, dir_plots, key_ref=key_ref, key_biggest_timestep=key_biggest_timestep)


## study with phase correction

# find phase difference between water levels and ref water level
phase_differences_in_s, water_levels = find_and_apply_phase_difference_between_water_levels(water_levels,
                                                                                            key_ref=key_ref,
                                                                                            t_delta_in_s=60)
# regression plots
regression_plots_interp_all_sources_on_biggest_timestep(water_levels, dir_plots, key_ref=key_ref,
                                                        key_biggest_timestep=key_biggest_timestep, phase_corrected=True)

# timely differences
plot_timely_differences(water_levels, key_ref=key_ref, key_biggest_timestep=key_biggest_timestep, phase_corrected=True)

# plot water levels
# vertical_ref = 'IGN69'
# t_interval = 7
# plot_water_levels(water_levels, png_out, vertical_ref, location, t_interval)