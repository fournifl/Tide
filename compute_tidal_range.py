import datetime
import pandas as pd
import pdb

import numpy as np
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt


def read_tide_shom_csv(tide_file):
    tide_dates = []
    tide_level = []
    tide_surge = []
    water_level = []
    with open(tide_file, 'r') as file:
        lines = file.readlines()[1:]
        for row in lines:
            date = datetime.datetime.strptime((row.split(',')[0]).split('.')[0], '%Y-%m-%d %H:%M:%S')
            tide_dates.append(date)
            # For tide and water_level, conversion from ref=mean_level to IGN69
            ssh = float(row.split(',')[2])
            surge = float(row.split(',')[1])
            tide_level.append(ssh)
            tide_surge.append(surge)
            water_level.append(ssh + surge)
    return np.array(tide_dates), np.array(tide_level), tide_surge, water_level


water_level_file = "/home/florent/Projects/Etretat/Water_levels/shom/NIVEAUX_HYCOM2D_R1000_ATL-CG.csv"

tide_dates, tide_level, _, _ = read_tide_shom_csv(water_level_file)
ind_local_maxs = argrelextrema(tide_level, np.greater)
ind_local_mins = argrelextrema(tide_level, np.less)

derivate = np.diff(tide_level)
time_derivate = tide_dates[0: tide_dates.size - 1]
ind_local_maxs_derivate = argrelextrema(derivate, np.greater)
ind_local_mins_derivate = argrelextrema(derivate, np.less)

ind_local_extremums_derivate = np.sort(np.append(ind_local_maxs_derivate[0], ind_local_mins_derivate[0]))

# plt.figure()
# plt.plot(tide_dates, tide_level, '+b')
# plt.plot(tide_dates[ind_local_maxs], tide_level[ind_local_maxs], 'dr')
# plt.plot(tide_dates[ind_local_mins], tide_level[ind_local_mins], 'dg')
# plt.plot(time_derivate, derivate, '.k')
# plt.plot(time_derivate[ind_local_extremums_derivate], derivate[ind_local_extremums_derivate], 'dc')
# plt.show()

t_tidal_range = []
tidal_range = []

for i, date in enumerate(time_derivate[ind_local_extremums_derivate]):
    print(date)

    # check that the date considered is surrounded by local min an local max
    if (any(tide_dates[ind_local_maxs] < date) or any(tide_dates[ind_local_mins] < date)) &\
        (any(tide_dates[ind_local_maxs] > date) or any(tide_dates[ind_local_mins] > date)):

        t_deltas = np.array([np.abs((date - tide_dates[ind_local_maxs][j]).total_seconds()) for j in range(tide_dates[ind_local_maxs].size)])
        ind_local_max = np.where(t_deltas == np.min(t_deltas))
        t_deltas = np.array([np.abs((date - tide_dates[ind_local_mins][j]).total_seconds()) for j in range(tide_dates[ind_local_mins].size)])
        ind_local_min = np.where(t_deltas == np.min(t_deltas))

        extrema_local_max = tide_level[ind_local_maxs][ind_local_max[0][0]]
        t_local_max = tide_dates[ind_local_maxs][ind_local_max[0][0]]
        extrema_local_min = tide_level[ind_local_mins][ind_local_min[0][0]]
        t_local_min = tide_dates[ind_local_mins][ind_local_min[0][0]]
        marnage = abs(extrema_local_max - extrema_local_min)

        # plt.plot(t_local_max, extrema_local_max, '.y')
        # plt.plot(t_local_min, extrema_local_min, '.m')

        t_tidal_range.append(date)
        tidal_range.append(np.around(marnage, decimals=2))

results = {}
results['dates'] = t_tidal_range
results['tidal_range'] = tidal_range
df = pd.DataFrame(data=results)
df.to_csv('test.csv', index=False)
# plt.plot(t_tidal_range, tidal_range, color='gold')
# plt.show()






