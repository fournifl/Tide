import datetime

import numpy as np


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
    for i, key in enumerate(water_levels.keys()):
        if key != key_to_interp_to:
            indate = [(t - datetime.datetime(1970, 1, 1)).total_seconds() for t in water_levels[key]['dates']]
            outdate = [(t - datetime.datetime(1970, 1, 1)).total_seconds() for t in
                       water_levels[key_to_interp_to]['dates']]
            water_level_interp = np.interp(np.array(outdate), np.array(indate),
                                           np.array(water_levels[key]['water_level']))
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
    for i, key in enumerate(water_levels.keys()):
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
