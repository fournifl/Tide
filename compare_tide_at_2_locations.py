import datetime
import pickle
from datetime import timedelta
import numpy as np
import os
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
        lines = file.readlines()[14:]
        for row in lines:
            date = datetime.datetime.strptime(row.split(';')[0], '%d/%m/%Y %H:%M:%S')
            # remove eventual seconds (donnees brutes HF)
            date = date - timedelta(seconds=date.second)
            dates.append(date)
            water_level.append(float(row.split(';')[1]) - shift_z)
    return dates, water_level


def read_tide_from_fes(f_tide_from_fes, NM_to_other_z_ref):
    with open(f_tide_from_fes, 'rb') as file_tide_from_fes:
        tide = pickle.load(file_tide_from_fes)
        dates_fes = tide['dates']
        tide_height_fes = np.array(tide['tide_from_fes'])
        tide_height_fes -= NM_to_other_z_ref
        return dates_fes, tide_height_fes


def read_wl_cmems(f_water_level_cmems, geoid_to_other_z_ref):
    water_level = pickle.load(open(f_water_level_cmems, 'rb'))
    time_cmems = water_level['time']
    ssh_cmems = water_level['ssh'][:]
    ssh_cmems -= geoid_to_other_z_ref
    return time_cmems, ssh_cmems


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


def compute_surge(t_start, t_end, dates_tg, water_level_tg, dates_fes, tide_height_fes, step):
    # interpolate FES and tg data on the same time array
    dates_tstep_constant = compute_period(t_start, t_end, step / (3600.0 * 24))
    water_level_tg_tstep_constant = oversample_array_with_constant_tstep(water_level_tg, dates_tg, dates_tstep_constant)
    tide_height_fes_tstep_constant = oversample_array_with_constant_tstep(tide_height_fes, dates_fes,
                                                                          dates_tstep_constant)
    # surge calculation
    surge = water_level_tg_tstep_constant - tide_height_fes_tstep_constant
    return dates_tstep_constant, surge


def save_surge(dates_tstep_constant, surge, file_pk):
    results_surge = {}
    results_surge['dates'] = dates_tstep_constant
    results_surge['surge'] = surge
    with open(file_pk, 'wb') as f_out:
        pickle.dump(results_surge, f_out)


def plot_water_levels(dates_fes_1, tide_height_fes_1, dates_fes_2, tide_height_fes_2, location_1, location_2, png_out,
                  vertical_ref, t_interval):
    diff_tide_height_cm = (tide_height_fes_2 - tide_height_fes_1) * 100
    f, ax = plt.subplots(2, figsize=(25, 10), sharex=True)
    ax[0].plot(dates_fes_1, tide_height_fes_1, color='navy', markersize=2, label='{location}'.format(location=location_1))
    ax[0].plot(dates_fes_2, tide_height_fes_2, color='dodgerblue', markersize=2, label='{location}'.format(location=location_2))
    ax[0].set_ylabel('Water level (m)', fontsize=16)
    ax[0].set_title(
        'WATER LEVEL RELATIVE TO {vertical_ref} AT {loc1} AND {loc2}'.format(vertical_ref=vertical_ref, loc1=location_1.upper(),
                                                                             loc2=location_2.upper()), fontsize=16)
    ax[0].legend(loc='upper right', fontsize=10, framealpha=0.6)
    ax[0].grid(True)
    ax[1].set_ylabel('Difference (cm)', fontsize=16)
    ax[1].set_title('DIFFERENCE', fontsize=16)
    ax[1].plot(dates_fes_1, diff_tide_height_cm, color='black', markersize=2,
               label='tide difference, %s vs %s, mean=%.4f, std=%.4f' %(location_2, location_1, np.mean(diff_tide_height_cm),
                                                                    np.std(diff_tide_height_cm)))
    ax[1].axhline(y=0, linewidth=2, color='gray', dashes=(4, 4))
    ax[1].legend(loc='upper right', fontsize=10, framealpha=0.6)
    ax[1].grid(True)
    ax[1].set_xlim([min(dates_fes_1), max(dates_fes_1)])
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d %H'))
    ax[1].xaxis.set_major_locator(mdates.DayLocator(interval=t_interval))
    f.autofmt_xdate()
    plt.tight_layout()
    f.savefig(png_out)
    plt.close()

# OPTIONS D'EXECUTION, AFFICHAGE
options = dict(
    read_despiked_tide=1,
)

path_water_levels = '/home/florent/Projects/Palavas-les-flots/Water_levels/'
path_fes = os.path.join(path_water_levels, 'tide_from_harmonic_constituents/')
location_1 = 'Palavas'
# location_2 = 'Port_Camargue'
location_2 = 'Sete'
f_tide_from_fes_1 = path_fes + 'tide_from_fes_constituents_{location_1}.pk'.format(location_1=location_1)
f_tide_from_fes_2 = path_fes + 'tide_from_fes_constituents_{location_2}.pk'.format(location_2=location_2)

t_start = datetime.datetime(2021, 4, 9)
t_end = datetime.datetime(2021, 5, 2)

# read FES tide
NM_to_NM = 0.0
dates_fes_1, tide_height_fes_1 = read_tide_from_fes(f_tide_from_fes_1, NM_to_NM)
dates_fes_1 = convert_datetime64_array_to_datetime_array(dates_fes_1)
dates_fes_1, tide_height_fes_1 = select_data_inside_study_period(t_start, t_end, np.array(dates_fes_1),
                                                             np.array(tide_height_fes_1))
dates_fes_2, tide_height_fes_2 = read_tide_from_fes(f_tide_from_fes_2, NM_to_NM)
dates_fes_2 = convert_datetime64_array_to_datetime_array(dates_fes_2)
dates_fes_2, tide_height_fes_2 = select_data_inside_study_period(t_start, t_end, np.array(dates_fes_2),
                                                             np.array(tide_height_fes_2))

# plot
png_out = path_fes + 'fes_tide_difference_{loc1}_vs_{loc2}.png'.format(loc1=location_1, loc2=location_2)
vertical_ref = 'NM'
t_interval = 2
plot_water_levels(dates_fes_1, tide_height_fes_1, dates_fes_2, tide_height_fes_2, location_1, location_2, png_out,
                  vertical_ref, t_interval)
