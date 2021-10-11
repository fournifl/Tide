import pdb
from os.path import join as join

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skill_metrics as sm
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error

from data_processing import interp_all_water_levels_on_specified_one


def regression_plots_interp_all_sources_on_ref(water_levels, dir_plots, key_ref=None, phase_corrected=False):
    water_levels_interp = interp_all_water_levels_on_specified_one(water_levels, key_ref)
    water_level_ref_interp = water_levels_interp[key_ref]

    water_level_ref = water_levels[key_ref]
    fig, ax = plt.subplots(1, len(water_levels.keys()) - 1, figsize=(22, 10))
    for i, key in enumerate(water_levels.keys()):
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
            ax[i].scatter(x, y, c=z, s=50, label='water levels: %s vs %s' % (key, key_ref))
            ax[i].text(min(x) + 0.15 * (max(x) - min(x)), min(y) + 0.8 * (max(y) - min(y)),
                       'y = %.3f x + %.3f \n R²=%.3f' % (regression.params[1], regression.params[0], r2[0]), color='b',
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


def regression_plots_interp_all_sources_on_biggest_timestep(water_levels, dir_plots, key_ref=None,
                                                            key_biggest_timestep=None, phase_corrected=False):
    water_levels_interp = interp_all_water_levels_on_specified_one(water_levels, key_biggest_timestep)
    water_level_ref_interp = water_levels_interp[key_ref]
    fig, ax = plt.subplots(1, len(water_levels.keys()) - 1, figsize=(22, 10))
    for i, key in enumerate(water_levels.keys()):
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
            ax[i].scatter(x, y, c=z, s=50, label='water levels: %s vs %s' % (key, key_ref))
            ax[i].text(min(x) + 0.15 * (max(x) - min(x)), min(y) + 0.8 * (max(y) - min(y)),
                       'y = %.3f x + %.3f \n R²=%.3f' % (regression.params[1], regression.params[0], r2[0]), color='b',
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


def plot_timely_differences(water_levels, dir_plots, key_ref=None, key_biggest_timestep=None, phase_corrected=False):
    water_levels_interp = interp_all_water_levels_on_specified_one(water_levels, key_biggest_timestep)
    water_level_ref_interp = water_levels_interp[key_ref]
    for i, key in enumerate(water_levels.keys()):
        if key != key_ref:
            water_level_interp = water_levels_interp[key]
            # indices without nan
            inds_no_nan = (~ np.isnan(water_level_ref_interp)) * (~ np.isnan(water_level_interp))
            inds_no_nan = np.where(inds_no_nan)[0]
            # plot timely differences
            x = water_level_ref_interp[inds_no_nan]
            y = water_level_interp[inds_no_nan]
            dates_no_nan = np.array(water_levels[key_biggest_timestep]['dates'])[inds_no_nan]
            print("Mean Diff %s - %s = %s" % (key_ref, key, (x - y).mean()))
            print("Median Diff %s - %s = %s" % (key_ref, key, np.median(x - y)))
            fig2, ax2 = plt.subplots(1, 1, figsize=(22, 10))
            ax2.axhline(y=0, linewidth=1, color='k', dashes=(4, 4))
            ax2.plot(dates_no_nan, x - y, label='diff water level %s - %s' % (key, key_ref))
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


def plot_taylor_diagrams(water_levels, dir_plots, key_ref=None, key_biggest_timestep=None, phase_corrected=False):
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
    if not phase_corrected:
        info_jpg = ""
    else:
        info_jpg = "_phase_corrected"
    jpg = join(dir_plots,
               'taylor_diagram_water_levels_ref_{key_ref}{info_jpg}.jpg'.format(key_ref=key_ref, info_jpg=info_jpg))
    plt.savefig(jpg)


def plot_water_levels(water_levels, png_out, vertical_ref, location, t_interval):
    f, ax = plt.subplots(1, figsize=(28, 6), sharex=True)
    for type_water_level in water_levels.keys():
        water_level = water_levels[type_water_level]
        ax.plot(water_level['dates'], water_level['water_level'], color=water_level['color'], markersize=2,
                label=water_level['label'])
    ax.set_ylabel('Water level (m)', fontsize=16)
    ax.set_title(
        'WATER LEVEL COMPARISON, RELATIVE TO {vertical_ref} AT {loc}'.format(vertical_ref=vertical_ref,
                                                                             loc=location.upper()),
        fontsize=16)
    ax.grid(True)
    ax.axhline(y=0, linewidth=2, color='gray', dashes=(4, 4))
    ax.legend(loc='upper right', fontsize=10, framealpha=0.6)
    # ax.set_xlim([min(dates_fes), max(dates_fes)])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d %H'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=t_interval))
    f.autofmt_xdate()
    # plt.show()
    # f.savefig(png_out)
    # plt.close()
