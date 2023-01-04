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
    try:
        water_level = pickle.load(open(f_water_level_cmems, 'rb'))
    except UnicodeDecodeError:
        water_level = pickle.load(open(f_water_level_cmems, 'rb'), encoding='latin1')
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
            print(dates[ind_dates[0]])
        else:
            arr_tstep_constant.append(np.nan)
    print('ouhba')
    return np.array(arr_tstep_constant)


def compute_surge(t_start, t_end, dates_tg, water_level_tg, dates_fes, tide_height_fes, step):
    # interpolate FES and tg data on the same time array
    dates_tstep_constant = compute_period(t_start, t_end, step / (3600.0 * 24))
    water_level_tg_tstep_constant = oversample_array_with_constant_tstep(water_level_tg, dates_tg, dates_tstep_constant)
    tide_height_fes_tstep_constant = oversample_array_with_constant_tstep(tide_height_fes, dates_fes, dates_tstep_constant)
    # surge calculation
    surge = water_level_tg_tstep_constant - tide_height_fes_tstep_constant
    return dates_tstep_constant, surge


def save_surge(dates_tstep_constant, surge, file_pk):
    results_surge = {}
    results_surge['dates'] = dates_tstep_constant
    results_surge['surge'] = surge
    with open(file_pk, 'wb') as f_out:
        pickle.dump(results_surge, f_out)


def load_surge(file_pk):
    with open(file_pk, 'rb') as f_pk:
        surge_results = pickle.load(f_pk)
    return surge_results['dates'], surge_results['surge']


def plot_water_levels(dates_tg, water_level_tg, dates_fes, tide_height_fes, dates_tstep_constant, surge, dates_cmems,
                      water_level_cmems, png_out, vertical_ref, location, t_interval):
    f, ax = plt.subplots(2, figsize=(28, 6), sharex=True)
    ax[0].plot(dates_tg, water_level_tg, color='navy', markersize=2, label='Tide gauge water level')
    ax[0].plot(dates_fes, tide_height_fes, color='dodgerblue', markersize=2, label='FES2014 tide')
    ax[0].set_ylabel('Water level (m)', fontsize=16)
    ax[0].set_title(
        'WATER LEVEL RELATIVE TO {vertical_ref} AT {loc}'.format(vertical_ref=vertical_ref, loc=location.upper()),
        fontsize=16)
    ax[0].legend(loc='upper right', fontsize=10, framealpha=0.6)
    ax[0].grid(True)
    ax[1].set_ylabel('Surge (m)', fontsize=16)
    ax[1].set_title('SURGE', fontsize=16)
    ax[1].plot(dates_tstep_constant, surge, color='darkseagreen', markersize=2, label='surge')
    # ax[1].plot(dates_cmems, water_level_cmems, '.g', markersize=2)
    ax[1].axhline(y=0, linewidth=2, color='gray', dashes=(4, 4))
    ax[1].legend(loc='upper right', fontsize=10, framealpha=0.6)
    ax[1].grid(True)
    ax[1].set_xlim([min(dates_fes), max(dates_fes)])
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d %H'))
    ax[1].xaxis.set_major_locator(mdates.DayLocator(interval=t_interval))
    f.autofmt_xdate()
    plt.show()
    f.savefig(png_out, bbox_inches='tight')
    plt.close()

# OPTIONS D'EXECUTION, AFFICHAGE
options = dict(
    read_despiked_tide=1,
    compute_surge=True
)

location = 'La Figueirette'
path_water_levels = '/home/florent/Projects/Cannes/Water_levels/'
path_fes = os.path.join(path_water_levels, 'tide_from_harmonic_constituents/')

f_wl_cmems = path_water_levels + '/MEDSEA_ANALYSIS_FORECAST_PHY_006_013/extract_pt_data/wl_extracted_at_La_Figueirette.pk'

t_start = datetime.datetime(2020, 3, 1)
t_end = datetime.datetime(2022, 3, 31)
# t_start = datetime.datetime(2020, 10, 1)
# t_end = datetime.datetime(2020, 10, 30)

# read tg water level
print('read tg')
ZH_to_IGN69 = 0.329
if options['read_despiked_tide']:
    print('ouhbiiiiiiiiiii')
    f_tg = path_water_levels + 'Cannes_water_level_tg_despiked.pk'
    tide_despiked = pickle.load(open(f_tg, 'rb'))
    dates_tg = tide_despiked['dates']
    water_level_tg = tide_despiked['water_level']
else:
    f_tg = path_water_levels + 'Cannes_water_level_tg.txt'
    dates_tg, water_level_tg = read_tide_tg_zh_to_other_ref(f_tg, ZH_to_IGN69)


dates_tg, water_level_tg = select_data_inside_study_period(t_start, t_end, np.array(dates_tg),
                                                           np.array(water_level_tg))

# read FES tide
print('read fes')
f_tide_from_fes = path_fes + 'tide_from_fes_constituents_la_figueirette_2020_03_01_2022_03_31.pk'
NM_to_ZH = - 0.51
NM_to_IGN69 = NM_to_ZH + ZH_to_IGN69
dates_fes, tide_height_fes = read_tide_from_fes(f_tide_from_fes, NM_to_IGN69)
dates_fes = convert_datetime64_array_to_datetime_array(dates_fes)
dates_fes, tide_height_fes = select_data_inside_study_period(t_start, t_end, np.array(dates_fes),
                                                             np.array(tide_height_fes))

# surge computation
if options['compute_surge']:
    print('surge computation')
    step = 600  # seconds
    dates_tstep_constant, surge = compute_surge(t_start, t_end, dates_tg, water_level_tg, dates_fes, tide_height_fes, step)

    # surge save
    print('save surge')
    save_surge(dates_tstep_constant, surge, path_fes + 'surge_la_figueirette.pk')

else:
    # load surge
    dates_tstep_constant, surge = load_surge(path_fes + 'surge_la_figueirette.pk')

# read CMEMS model data
geoide_to_ZH = -0.51
geoide_to_IGN69 = geoide_to_ZH + ZH_to_IGN69
dates_cmems, water_level_cmems = read_wl_cmems(f_wl_cmems, geoide_to_IGN69)


# plot
png_out = '/home/florent/Projects/Cannes/Water_levels/tide_from_harmonic_constituents/surge_by_comparing_fes_tide_with_tg.png'
vertical_ref = 'IGN69'
t_interval = 15
plot_water_levels(dates_tg, water_level_tg, dates_fes, tide_height_fes, dates_tstep_constant, surge, dates_cmems,
                  water_level_cmems, png_out, vertical_ref, location, t_interval)
