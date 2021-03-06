from __future__ import division
import os
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from itertools import combinations
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn import metrics, linear_model
import copy
import scipy.stats
from statsmodels.robust import mad
from cell_fitting.data.plot_IV import check_v_at_i_inj_0_is_at_right_sweep_idx
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj_from_function
from cell_fitting.util import init_nan
from cell_characteristics.analyze_APs import get_AP_onset_idxs, get_AP_max_idx, \
    get_fAHP_min_idx_using_splines, get_DAP_max_idx_using_splines, get_DAP_amp, get_DAP_deflection
from cell_characteristics import to_idx
pl.style.use('paper')


def get_sta(v_APs):
    sta = np.mean(v_APs, 0)
    sta_std = np.std(v_APs, 0)
    return sta, sta_std


def get_sta_median(v_APs):
    sta = np.median(v_APs, 0)
    sta_mad = mad(v_APs, axis=0)
    return sta, sta_mad


def get_stc(v_APs):
    cov = np.cov(v_APs.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # keeps eigvals, eigvecs real (instead of np.lingalg.eig)
    assert np.all([np.round(np.dot(v1, v2), 10) == 0
                   for v1, v2 in combinations(eigvecs.T, 2)])  # check orthogonality of eigvecs
    eigvals, eigvecs = sort_eigvals_descending(eigvals, eigvecs)
    expl_var = eigvals / np.sum(eigvals) * 100
    return eigvals, eigvecs, expl_var


def sort_eigvals_descending(eigvals, eigvecs):
    idx_sort = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx_sort]
    eigvecs = eigvecs[:, idx_sort]
    return eigvals, eigvecs


def choose_eigvecs(eigvecs, eigvals, n_eigvecs=3, least_expl_var=None):
    if least_expl_var is not None:
        n_eigvecs = np.where(np.cumsum(eigvals) / np.sum(eigvals) >= least_expl_var)[0][0] + 1
    chosen_eigvecs = eigvecs[:, :n_eigvecs]
    return chosen_eigvecs


def project_back(v_APs, chosen_eigvecs):
    v_APs_centered = v_APs - np.mean(v_APs, 0)
    back_projection = np.dot(v_APs_centered, np.dot(chosen_eigvecs, chosen_eigvecs.T)) + np.mean(v_APs, 0)
    return back_projection


def find_APs_in_v_trace(v, AP_threshold, before_AP_idx, after_AP_idx):
    v_APs = []
    onset_idxs = get_AP_onset_idxs(v, AP_threshold)
    if len(onset_idxs) > 0:
        onset_idxs = np.insert(onset_idxs, len(onset_idxs), len(v))  # add end point to delimit for all APs start and end
        AP_max_idxs = [get_AP_max_idx(v, onset_idx, next_onset_idx) for (onset_idx, next_onset_idx)
                       in zip(onset_idxs[:-1], onset_idxs[1:])]

        for i, AP_max_idx in enumerate(AP_max_idxs):
            if (AP_max_idx is not None  # None if no AP max found (e.g. at end of v trace)
                    and AP_max_idx - before_AP_idx >= 0 and AP_max_idx + after_AP_idx < len(v)):  # able to draw window
                v_AP = v[AP_max_idx - before_AP_idx:AP_max_idx + after_AP_idx + 1]

                if before_AP_idx > AP_max_idx - onset_idxs[i]:
                    n_APs_desired = 1  # if we start before the onset, the AP belonging to the onset should be detected
                else:
                    n_APs_desired = 0  # else no AP should be detected

                if len(get_AP_onset_idxs(v_AP, AP_threshold)) == n_APs_desired:  # only take windows where there is no other AP
                    v_APs.append(v_AP)
    return v_APs


def plot_sta(t_AP, sta, sta_std, save_dir_img):
    pl.figure()
    #pl.title('STA', fontsize=18)
    pl.plot(t_AP, sta, 'k')
    pl.fill_between(t_AP, sta + sta_std, sta - sta_std,
                    facecolor='k', alpha=0.5)
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane Potential (mV)')
    pl.tight_layout()
    pl.savefig(save_dir_img)
    
    
def plot_APs(v_APs, t_AP, save_dir_img):    
    v_APs_plots = v_APs[np.random.randint(0, len(v_APs), 50)]  # reduce to lower number
    
    pl.figure()
    pl.title('AP Traces (# %i)' % len(v_APs), fontsize=18)
    for v_AP in v_APs_plots:
        pl.plot(t_AP, v_AP)
    pl.ylabel('Membrane potential (mV)')
    pl.xlabel('Time (ms)')
    pl.tight_layout()
    if save_dir_img is not None:
        pl.savefig(save_dir_img)


def plots_stc(v_APs, t_AP, back_projection, chosen_eigvecs, expl_var, save_dir_img):

    APs_for_plots_idxs = np.random.randint(0, len(v_APs), 50)
    v_APs_plots = v_APs[APs_for_plots_idxs]  # reduce to lower number

    pl.figure()
    pl.title('AP Traces (# %i)' % len(v_APs), fontsize=18)
    for vec in v_APs_plots:
        pl.plot(t_AP, vec)
    pl.ylabel('Membrane Potential (mV)')
    pl.xlabel('Time (ms)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'STC_APs.png'))

    pl.figure()
    pl.title('AP Traces - Mean', fontsize=18)
    for vec in v_APs_plots:
        pl.plot(t_AP, vec - np.mean(v_APs, 0))
    pl.ylabel('Membrane Potential (mV)')
    pl.xlabel('Time (ms)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'STC_APs_minus_mean.png'))

    pl.figure()
    pl.title('Backprojected AP Traces', fontsize=18)
    for vec in back_projection[APs_for_plots_idxs]:
        pl.plot(t_AP, vec)
    pl.ylabel('Membrane Potential (mV)')
    pl.xlabel('Time (ms)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'STC_backprojected_APs.png'))

    fig, ax = pl.subplots()
    ax.set_title('Eigenvectors', fontsize=18)
    ax2 = ax.twinx()
    ax2.plot(t_AP, np.mean(v_APs, 0), 'k')
    ax2.set_ylabel('Membrane Potential (mV)')
    ax2.spines['right'].set_visible(True)
    for i, vec in enumerate(chosen_eigvecs.T):
        if vec[0] < 0:
            vec *= -1
        ax.plot(t_AP, vec, label='expl. var.: %i %%' % int(round(expl_var[i])))
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Eigenvector')
    ax.legend(loc='lower right', fontsize=10)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'STC_largest_eigenvecs.png'))

    pl.figure()
    pl.title('Cumulative Explained Variance', fontsize=18)
    pl.plot(np.arange(len(expl_var)), np.cumsum(expl_var), 'ok', markersize=6)
    pl.ylabel('Percent')
    pl.xlabel('#')
    pl.ylim(0, 105)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'STC_eigenvals.png'))


def plot_all_in_one(v_APs, t_AP, back_projection, mean_high, std_high, mean_low, std_low, chosen_eigvecs, expl_var,
                    ica_components, save_dir_img):
    ica_components_copy = copy.copy(ica_components)

    APs_for_plots_idxs = np.random.randint(0, len(v_APs), max(len(v_APs), 100))
    v_APs_plots = v_APs[APs_for_plots_idxs]  # reduce to lower number
    fig, axes = pl.subplots(2, 3, figsize=(14, 7))

    axes[0, 0].set_title('AP Traces (# %i)' % len(v_APs), fontsize=16)
    for vec in v_APs_plots:
        axes[0, 0].plot(t_AP, vec)
    axes[0, 0].set_ylabel('Membrane Potential (mV)', fontsize=16)
    axes[0, 0].set_xlabel('Time (ms)', fontsize=16)

    axes[0, 1].set_title('Backprojected AP Traces', fontsize=16)
    for vec in back_projection[APs_for_plots_idxs]:
        axes[0, 1].plot(t_AP, vec)
    axes[0, 1].set_ylabel('Membrane Potential (mV)', fontsize=16)
    axes[0, 1].set_xlabel('Time (ms)', fontsize=16)

    axes[0, 2].set_title('AP Traces - Mean', fontsize=16)
    for vec in v_APs_plots:
        axes[0, 2].plot(t_AP, vec - np.mean(v_APs, 0))
    axes[0, 2].set_ylabel('Membrane Potential (mV)', fontsize=16)
    axes[0, 2].set_xlabel('Time (ms)', fontsize=16)

    # axes[1, 0].plot(np.arange(len(expl_var)), np.cumsum(expl_var), 'ok', markersize=5)
    # axes[1, 0].set_title('Explained Variance', fontsize=16)
    # axes[1, 0].set_ylabel('CDF', fontsize=16)
    # axes[1, 0].set_xlabel('#', fontsize=16)
    # axes[1, 0].set_ylim(0, 105)

    axes[1, 0].set_title('Group by $AP_{max}$', fontsize=16)
    axes[1, 0].fill_between(t_AP, mean_high + std_high, mean_high - std_high,
                    facecolor='r', alpha=0.3)
    axes[1, 0].fill_between(t_AP, mean_low + std_low, mean_low - std_low,
                    facecolor='b', alpha=0.3)
    axes[1, 0].plot(t_AP, mean_high, 'r', label='High $AP_{max}$')
    axes[1, 0].plot(t_AP, mean_low, 'b', label='Low $AP_{max}$')
    axes[1, 0].plot(t_AP, mean_high - mean_low, 'k', label='High - Low')
    axes[1, 0].set_ylabel('Membrane Potential (mV)', fontsize=16)
    axes[1, 0].set_xlabel('Time (ms)', fontsize=16)
    axes[1, 0].legend(loc='lower right', fontsize=10)

    axes[1, 1].set_title('Eigenvectors', fontsize=16)
    ax2 = axes[1, 1].twinx()
    ax2.plot(t_AP, np.mean(v_APs, 0), 'k')
    ax2.set_ylabel('Membrane Potential (mV)', fontsize=16)
    ax2.spines['right'].set_visible(True)
    for i, vec in enumerate(chosen_eigvecs.T):
        if vec[0] < 0:
            vec *= -1
        axes[1, 1].plot(t_AP, vec, label='expl. var.: %i %%' % int(round(expl_var[i])))
    axes[1, 1].set_xlabel('Time (ms)', fontsize=16)
    axes[1, 1].set_ylabel('Eigenvector', fontsize=16)
    axes[1, 1].legend(loc='lower right', fontsize=10)

    axes[1, 2].set_title('ICA Components', fontsize=16)
    axes[1, 2].set_xlabel('Time (ms)', fontsize=16)
    axes[1, 2].set_ylabel('ICA Component', fontsize=16)
    ax2 = axes[1, 2].twinx()
    ax2.plot(t_AP, np.mean(v_APs, 0), 'k')
    ax2.set_ylabel('Membrane Potential (mV)', fontsize=16)
    ax2.spines['right'].set_visible(True)
    for i, vec in enumerate(ica_components_copy.T):
        if vec[0] < 0:
            vec *= -1
        axes[1, 2].plot(t_AP, vec, label='%i' % i)
    axes[1, 2].legend(loc='lower right', fontsize=10)

    pl.subplots_adjust(left=0.06, right=0.94, bottom=0.09, top=0.95, wspace=0.54, hspace=0.34)
    pl.savefig(os.path.join(save_dir_img, 'STC_all_in_one.png'))


def plot_group_by_AP_max(mean_high, std_high, mean_low, std_low, t_AP, save_dir_img):
    pl.figure()
    pl.title('Group by $AP_{max}$', fontsize=18)
    pl.fill_between(t_AP, mean_high + std_high, mean_high - std_high,
                    facecolor='r', alpha=0.5)
    pl.fill_between(t_AP, mean_low + std_low, mean_low - std_low,
                    facecolor='b', alpha=0.5)
    pl.plot(t_AP, mean_high, 'r', label='High $AP_{max}$')
    pl.plot(t_AP, mean_low, 'b', label='Low $AP_{max}$')
    pl.plot(t_AP, mean_high - mean_low, 'k', label='High - Low')
    pl.ylabel('Membrane Potential (mV)')
    pl.xlabel('Time (ms)')
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'Group_by_AP_max.png'))


def group_by_AP_max(v_APs):
    AP_max = np.array([vec[0] for vec in v_APs])
    AP_max_sort_idxs = np.argsort(AP_max)
    AP_max_sort = AP_max[AP_max_sort_idxs]
    AP_max_high_labels = AP_max >= AP_max_sort[int(round(len(AP_max_sort) * (1. / 2.)))]
    mean_high = np.mean(v_APs[AP_max_high_labels], 0)
    std_high = np.std(v_APs[AP_max_high_labels], 0)
    mean_low = np.mean(v_APs[~AP_max_high_labels], 0)
    std_low = np.std(v_APs[~AP_max_high_labels], 0)
    return mean_high, std_high, mean_low, std_low, AP_max_high_labels, AP_max


def plot_ICA(v_APs, t_AP, ica_components, save_dir_img):
    ica_components_copy = copy.copy(ica_components)
    fig, ax = pl.subplots()
    ax.set_title('ICA Components', fontsize=18)
    ax2 = ax.twinx()
    ax2.plot(t_AP, np.mean(v_APs, 0), 'k')
    ax2.set_ylabel('Membrane Potential (mV)')
    ax2.spines['right'].set_visible(True)
    for vec in ica_components_copy.T:
        if vec[0] < 0:
            vec *= -1
        ax.plot(t_AP, vec)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('ICA Component')
    ax.legend(loc='lower right', fontsize=10)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'ICA_components.png'))


def plot_PCA_3D(v_APs_centered, chosen_eigvecs, AP_max_high_labels=None, AP_max=None, DAP_maxs=None,
                DAP_deflections=None, save_dir_img=None):
    if np.shape(chosen_eigvecs)[1] > 3:
        raise Warning('More than 3 eigenvectors chosen!')
        chosen_eigvecs = chosen_eigvecs[:4]

    v_APs_projected = np.dot(v_APs_centered, chosen_eigvecs)

    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    for (x, y, z) in v_APs_projected:
        ax.scatter(x, y, z, color='k')
    ax.set_xlabel('Eigenvector 1', fontsize=16)
    ax.set_ylabel('Eigenvector 2', fontsize=16)
    ax.set_zlabel('Eigenvector 3', fontsize=16)
    ax.view_init(45, -45)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'PC_projection.png'))

    if AP_max_high_labels is not None:
        silhouette_score = metrics.silhouette_score(v_APs_projected, AP_max_high_labels, metric='euclidean')

        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('Silhouette Score: %.2f' % silhouette_score)
        for i, (x, y, z) in enumerate(v_APs_projected[AP_max_high_labels]):
            ax.scatter(x, y, z, color='r', label='$high\ AP_{max}$' if i == 0 else '')
        for i, (x, y, z) in enumerate(v_APs_projected[~AP_max_high_labels]):
            ax.scatter(x, y, z, color='b', label='$low\ AP_{max}$' if i == 0 else '')
        ax.set_xlabel('Eigenvector 1', fontsize=16)
        ax.set_ylabel('Eigenvector 2', fontsize=16)
        ax.set_zlabel('Eigenvector 3', fontsize=16)
        ax.view_init(45, -45)
        pl.legend(fontsize=14)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'PC_projection_with_AP_max_groups.png'))

    if AP_max is not None:
        cmap = pl.get_cmap('viridis')
        AP_max_normed = (AP_max - np.min(AP_max)) / np.max(AP_max - np.min(AP_max))
        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, (x, y, z) in enumerate(v_APs_projected):
            scat = ax.scatter(x, y, z, color=cmap(AP_max_normed[i]))
        ax.set_xlabel('Eigenvector 1', fontsize=16)
        ax.set_ylabel('Eigenvector 2', fontsize=16)
        ax.set_zlabel('Eigenvector 3', fontsize=16)
        ax.view_init(45, -45)
        scat.set_array(AP_max_normed)
        scat.set_clim(0, 1)
        cbar = fig.colorbar(mappable=scat, ax=ax)
        cbar.set_ticks(np.linspace(0, 1, 5))
        cbar.set_ticklabels(['%.1f' % tick for tick in np.linspace(np.min(AP_max), np.max(AP_max), 5)], True)
        cbar.set_label('$AP_{max}$')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'PC_projection_with_AP_max.png'))

    if DAP_deflections is not None:
        cmap = pl.get_cmap('viridis')
        DAP_deflections_normed = (DAP_deflections - np.nanmin(DAP_deflections)) \
                                 / np.nanmax(DAP_deflections - np.nanmin(DAP_deflections))
        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, (x, y, z) in enumerate(v_APs_projected):
            if np.isnan(DAP_deflections_normed[i]):
                color = '0.5'
            else:
                color = cmap(DAP_deflections_normed[i])
            scat = ax.scatter(x, y, z, color=color)
        ax.set_xlabel('Eigenvector 1', fontsize=16)
        ax.set_ylabel('Eigenvector 2', fontsize=16)
        ax.set_zlabel('Eigenvector 3', fontsize=16)
        ax.view_init(45, -45)
        scat.set_array(DAP_deflections_normed)
        scat.set_clim(0, 1)
        cbar = fig.colorbar(mappable=scat, ax=ax)
        cbar.set_ticks(np.linspace(0, 1, 5))
        cbar.set_ticklabels(['%.1f' % tick
                             for tick in np.linspace(np.nanmin(DAP_deflections), np.nanmax(DAP_deflections), 5)], True)
        cbar.set_label('$DAP_{deflection}$')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'PC_projection_with_DAP_deflections.png'))

    if DAP_maxs is not None:
        cmap = pl.get_cmap('viridis_r')
        DAP_maxs_normed = ((DAP_maxs - np.nanmin(DAP_maxs)) / np.nanmax(DAP_maxs - np.nanmin(DAP_maxs)))
        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, (x, y, z) in enumerate(v_APs_projected):
            if np.isnan(DAP_maxs_normed[i]):
                color = '0.5'
            else:
                color = cmap(DAP_maxs_normed[i])
            scat = ax.scatter(x, y, z, color=color)
        ax.set_xlabel('Eigenvector 1', fontsize=16)
        ax.set_ylabel('Eigenvector 2', fontsize=16)
        ax.set_zlabel('Eigenvector 3', fontsize=16)
        ax.view_init(45, -45)
        scat.set_array(DAP_maxs_normed)
        scat.set_clim(0, 1)
        cbar = fig.colorbar(mappable=scat, ax=ax)
        cbar.set_ticks(np.linspace(0, 1, 5))
        cbar.set_ticklabels(['%.1f' % tick for tick in np.linspace(np.nanmin(DAP_maxs), np.nanmax(DAP_maxs), 5)][::-1], True)
        cbar.set_label('$DAP_{max}$')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'PC_projection_with_DAP_max.png'))


def detect_outliers(ys, threshold=3):
    mean_y = np.nanmean(ys)
    stdev_y = np.nanstd(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return np.abs(z_scores) > threshold


def plot_corr_PCA_characteristics(v_APs_centered, chosen_eigvecs, AP_max=None, DAP_maxs=None,
                DAP_deflections=None, outlier_threshold=3.0, save_dir_img=None):
    if np.shape(chosen_eigvecs)[1] > 3:
        raise Warning('More than 3 eigenvectors chosen!')
        chosen_eigvecs = chosen_eigvecs[:4]

    v_APs_projected = np.dot(v_APs_centered, chosen_eigvecs)

    for eigvec_idx in range(3):
        if AP_max is not None:
            regr = linear_model.LinearRegression()
            regr.fit(np.array([v_APs_projected[:, eigvec_idx]]).T, AP_max)
            corr, p_val = scipy.stats.pearsonr(v_APs_projected[:, eigvec_idx], AP_max)
            pl.figure()
            pl.title('Corr.: %.2f, p-val.: %.3f' % (corr, p_val))
            pl.plot(v_APs_projected[:, eigvec_idx], AP_max, 'ok')
            pl.plot(v_APs_projected[:, eigvec_idx], regr.coef_[0] * v_APs_projected[:, eigvec_idx] + regr.intercept_, '0.5')
            pl.xlabel('Eigenvektor '+str(eigvec_idx+1))
            pl.ylabel('$AP_{max}$')
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_img, 'Corr_eigvec'+str(eigvec_idx+1)+'_AP_max.png'))

        if DAP_deflections is not None:
            #pl.close('all')
            not_nan = ~np.isnan(DAP_deflections)
            outliers = detect_outliers(DAP_deflections, outlier_threshold)
            regr = linear_model.LinearRegression()
            regr.fit(np.array([v_APs_projected[:, eigvec_idx][np.logical_and(not_nan, ~outliers)]]).T,
                     DAP_deflections[np.logical_and(not_nan, ~outliers)])
            corr, p_val = scipy.stats.pearsonr(v_APs_projected[:, eigvec_idx][np.logical_and(not_nan, ~outliers)],
                                               DAP_deflections[np.logical_and(not_nan, ~outliers)])
            pl.figure()
            pl.title('Corr.: %.2f, p-val.: %.3f' % (corr, p_val))
            pl.plot(v_APs_projected[:, eigvec_idx][np.logical_and(not_nan, ~outliers)], DAP_deflections[np.logical_and(not_nan, ~outliers)], 'ok')
            pl.plot(v_APs_projected[:, eigvec_idx][np.logical_and(not_nan, ~outliers)],
                    regr.coef_[0] * v_APs_projected[:, eigvec_idx][np.logical_and(not_nan, ~outliers)] + regr.intercept_, '0.5')
            #pl.plot(v_APs_projected[:, eigvec_idx][outliers], DAP_deflections[outliers], 'or')
            pl.xlabel('Eigenvektor '+str(eigvec_idx+1))
            pl.ylabel('$DAP_{deflection}$')
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_img, 'Corr_eigvec'+str(eigvec_idx+1)+'_DAP_deflection_without_outliers.png'))
            #pl.show()

        if DAP_maxs is not None:
            not_nan = ~np.isnan(DAP_deflections)
            regr = linear_model.LinearRegression()
            regr.fit(np.array([v_APs_projected[:, eigvec_idx][not_nan]]).T, DAP_maxs[not_nan])
            corr, p_val = scipy.stats.pearsonr(v_APs_projected[:, eigvec_idx][not_nan], DAP_maxs[not_nan])
            pl.figure()
            pl.title('Corr.: %.2f, p-val.: %.3f' % (corr, p_val))
            pl.plot(v_APs_projected[:, eigvec_idx], DAP_maxs, 'ok')
            pl.plot(v_APs_projected[:, eigvec_idx], regr.coef_[0] * v_APs_projected[:, eigvec_idx] + regr.intercept_, '0.5')
            pl.xlabel('Eigenvektor '+str(eigvec_idx+1))
            pl.ylabel('$DAP_{max}$')
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_img, 'Corr_eigvec'+str(eigvec_idx+1)+'_DAP_max.png'))

            # AP max vs DAP max
            not_nan = ~np.isnan(DAP_deflections)
            regr = linear_model.LinearRegression()
            regr.fit(np.array([AP_max[not_nan]]).T, DAP_maxs[not_nan])
            corr, p_val = scipy.stats.pearsonr(AP_max[not_nan], DAP_maxs[not_nan])
            pl.figure()
            pl.title('Corr.: %.2f, p-val.: %.3f' % (corr, p_val))
            pl.plot(AP_max, DAP_maxs, 'ok')
            pl.plot(AP_max, regr.coef_[0] * AP_max + regr.intercept_, '0.5')
            pl.xlabel('$AP_{max}$')
            pl.ylabel('$DAP_{max}$')
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_img, 'Corr_AP_max_DAP_max.png'))


def plot_ICA_3D(v_APs_centered, ica_source, AP_max_high_labels=None, save_dir_img=None):
    if np.shape(ica_source)[1] > 3:
        raise Warning('More than 3 components chosen!')
        ica_source = ica_source[:4]

    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    for (x, y, z) in ica_source:
        ax.scatter(x, y, z, color='k')
    ax.set_xlabel('Component 1', fontsize=16)
    ax.set_ylabel('Component 2', fontsize=16)
    ax.set_zlabel('Component 3', fontsize=16)
    ax.view_init(45, -45)
    pl.savefig(os.path.join(save_dir_img, 'ICA_projection.png'))

    if AP_max_high_labels is not None:
        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, (x, y, z) in enumerate(ica_source[AP_max_high_labels]):
            ax.scatter(x, y, z, color='r', label='$high\ AP_{max}$' if i == 0 else '')
        for i, (x, y, z) in enumerate(ica_source[~AP_max_high_labels]):
            ax.scatter(x, y, z, color='b', label='$low\ AP_{max}$' if i == 0 else '')
        ax.set_xlabel('Component 1', fontsize=16)
        ax.set_ylabel('Component 2', fontsize=16)
        ax.set_zlabel('Component 3', fontsize=16)
        ax.view_init(45, -45)
        pl.legend(fontsize=14)
        pl.savefig(os.path.join(save_dir_img, 'ICA_projection_with_AP_max.png'))


def plot_clustering_kmeans(v_APs, v_APs_centered, t_AP, chosen_eigvecs, n_clusters=2, save_dir_img=None):
    kmeans = KMeans(n_clusters=n_clusters)
    v_APs_projected = np.dot(v_APs_centered, chosen_eigvecs)
    kmeans.fit(v_APs_projected)
    labels = kmeans.labels_
    silhouette_score_val = metrics.silhouette_score(v_APs_projected, labels, metric='euclidean')

    colors = pl.get_cmap('plasma')([float(i) / n_clusters for i in range(n_clusters)])
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Silhouette Score: %.2f' % silhouette_score_val)
    for i, (x, y, z) in enumerate(v_APs_projected):
        ax.scatter(x, y, z, color=colors[labels[i]])
    ax.set_xlabel('Eigenvector 1', fontsize=16)
    ax.set_ylabel('Eigenvector 2', fontsize=16)
    ax.set_zlabel('Eigenvector 3', fontsize=16)
    ax.view_init(45, -45)
    pl.savefig(os.path.join(save_dir_img, 'PC_projection_clustered.png'))

    pl.figure()
    pl.title('APs clustered in 2 groups')
    for label in np.unique(labels):
        for vec in v_APs[labels==label]:
            pl.plot(t_AP, vec, color=colors[label])
    pl.ylabel('Membrane Potential (mV)')
    pl.xlabel('Time (ms)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'v_APs_clustered.png'))


def plot_backtransform(v_APs_centered, t_AP, mean_high_centered, mean_low_centered, std_high, std_low, chosen_eigvecs,
                       expl_var, ica_source, ica_mixing, save_dir_img):
    fig, axes = pl.subplots(3, 3, figsize=(14, 7), sharey='all', sharex='all')
    axes[0, 0].set_title('Centered APs')
    for vec in v_APs_centered:
        axes[0, 0].plot(t_AP, vec)
    ylim_min, ylim_max = axes[0, 0].get_ylim()
    ylim_max -= 1
    ylim_min += 1

    axes[1, 0].set_title('PCA: back-transform')
    for vec in np.dot(v_APs_centered, np.dot(chosen_eigvecs, chosen_eigvecs.T)):
        axes[1, 0].plot(t_AP, vec)

    axes[2, 0].set_title('ICA: back-transform')
    for vec in np.dot(ica_source, ica_mixing.T):
        axes[2, 0].plot(t_AP, vec)

    axes[0, 1].set_title('Error Centered APs')
    for vec in v_APs_centered - v_APs_centered:
        axes[0, 1].plot(t_AP, vec)

    axes[1, 1].set_title('Error PCA: back-transform')
    for vec in v_APs_centered - np.dot(v_APs_centered, np.dot(chosen_eigvecs, chosen_eigvecs.T)):
        axes[1, 1].plot(t_AP, vec)

    axes[2, 1].set_title('Error ICA: back-transform')
    for vec in v_APs_centered - np.dot(ica_source, ica_mixing.T):
        axes[2, 1].plot(t_AP, vec)

    # TODO: group by AP max
    axes[0, 2].set_title('Group by $AP_{max}$')
    axes[0, 2].axhline(0, 0, 1, color='0.5', linestyle='--')
    axes[0, 2].fill_between(t_AP, mean_high_centered + std_high, mean_high_centered - std_high,
                    facecolor='r', alpha=0.3)
    axes[0, 2].fill_between(t_AP, mean_low_centered + std_low, mean_low_centered - std_low,
                    facecolor='b', alpha=0.3)
    axes[0, 2].plot(t_AP, mean_high_centered, 'r', label='High $AP_{max}$')
    axes[0, 2].plot(t_AP, mean_low_centered, 'b', label='Low $AP_{max}$')
    diff = mean_high_centered - mean_low_centered
    diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    diff = diff * (ylim_max - ylim_min) + ylim_min
    axes[0, 2].plot(t_AP, diff, 'k', label='High - Low')
    axes[0, 2].legend(loc='lower right', fontsize=10)

    axes[1, 2].set_title('PCA: eigenvectors')
    axes[1, 2].axhline(0, 0, 1, color='0.5', linestyle='--')
    for i, vec in enumerate(chosen_eigvecs.T):
        if vec[0] < 0:
            vec = vec*-1
        vec = (vec - np.min(vec)) / (np.max(vec) - np.min(vec))
        vec = vec * (ylim_max - ylim_min) + ylim_min
        axes[1, 2].plot(t_AP, vec, label='expl. var.: %.2f' % expl_var[i])
    axes[1, 2].legend(fontsize=10, loc='lower right')

    axes[2, 2].set_title('ICA: mixing matrix')
    axes[2, 2].axhline(0, 0, 1, color='0.5', linestyle='--')
    for vec in ica_mixing.T:
        if vec[0] < 0:
            vec = vec * -1
        vec = (vec - np.min(vec)) / (np.max(vec) - np.min(vec))
        vec = vec * (ylim_max - ylim_min) + ylim_min
        axes[2, 2].plot(t_AP, vec)

    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'backtransform.png'))


def get_DAP_characteristics_from_v_APs(v_APs, t_AP, AP_threshold):
    DAP_maxs = init_nan(len(v_APs))
    DAP_deflections = init_nan(len(v_APs))
    dt = t_AP[1] - t_AP[0]
    for i, v in enumerate(v_APs):
        AP_max_idx = 0
        v_rest = np.min(v)
        std = np.std(v[-to_idx(2.0, dt):])
        w = np.ones(len(v)) / std
        fAHP_min_idx = get_fAHP_min_idx_using_splines(v, t_AP, AP_max_idx, len(t_AP), order=to_idx(1.0, dt),
                                                      interval=to_idx(4.0, dt),
                                                      w=w, k=3, s=None)
        if fAHP_min_idx is not None:
            DAP_max_idx = get_DAP_max_idx_using_splines(v, t_AP, int(fAHP_min_idx), len(t_AP), order=to_idx(1.0, dt),
                                                        interval=to_idx(5.0, dt), min_dist_to_max=to_idx(0.5, dt),
                                                        w=w, k=3, s=None)
            if DAP_max_idx is not None and v[DAP_max_idx] > v[fAHP_min_idx]:
                DAP_maxs[i] = v[DAP_max_idx]  # TODO: get_DAP_amp(v, int(DAP_max_idx), v_rest)
                DAP_deflections[i] = get_DAP_deflection(v, int(fAHP_min_idx), int(DAP_max_idx))

                # pl.figure()
                # pl.plot(t_AP, v, 'k')
                # pl.plot(t_AP[fAHP_min_idx], v[fAHP_min_idx], 'or')
                # pl.plot(t_AP[DAP_max_idx], v[DAP_max_idx], 'og')
                # pl.show()
        # else:
        #     pl.figure()
        #     pl.plot(t_AP, v, 'b')
        #     pl.show()

    return DAP_maxs, DAP_deflections


if __name__ == '__main__':
    # parameters
    cell_ids = ['2015_08_26b', '2015_08_26e']
    save_dir = './test'
    protocol = 'IV'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'

    AP_threshold = -20
    before_AP_STA = 25
    after_AP_STA = 25
    before_AP_STC = 0
    after_AP_STC = 25
    start_step = 250  # ms
    end_step = 750  # ms

    for cell_id in cell_ids:
        print cell_id

        # load data
        v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                         return_sweep_idxs=True)
        t = t_mat[0, :]
        dt = t[1] - t[0]
        i_inj_mat = get_i_inj_from_function(protocol, sweep_idxs, t[-1], t[1]-t[0])

        check_v_at_i_inj_0_is_at_right_sweep_idx(v_mat, i_inj_mat)

        start_step_idx = to_idx(start_step, dt)
        end_step_idx = to_idx(end_step, dt)
        before_AP_idx_sta = to_idx(before_AP_STA, dt)
        after_AP_idx_sta = to_idx(after_AP_STA, dt)
        before_AP_idx_STC = to_idx(before_AP_STC, dt)
        after_AP_idx_STC = to_idx(after_AP_STC, dt)

        v_mat = v_mat[:, start_step_idx:end_step_idx]  # only take v during step current
        t = t[start_step_idx:end_step_idx]

        # save and plot
        save_dir_img = os.path.join(save_dir, str(cell_id))
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        # STA
        v_APs = []
        for v in v_mat:
            v_APs.extend(find_APs_in_v_trace(v, AP_threshold, before_AP_idx_sta, after_AP_idx_sta))
        v_APs = np.vstack(v_APs)
        t_AP = np.arange(after_AP_idx_sta + before_AP_idx_sta + 1) * dt
        sta, sta_std = get_sta(v_APs)

        plots_sta(v_APs, t_AP, sta, sta_std, save_dir_img)

        # STC & Group by AP_max & ICA
        v_APs = []
        for v in v_mat:
            v_APs.extend(find_APs_in_v_trace(v, AP_threshold, before_AP_idx_STC, after_AP_idx_STC))
        v_APs = np.vstack(v_APs)
        v_APs_centered = v_APs - np.mean(v_APs, 0)
        t_AP = np.arange(after_AP_idx_STC + before_AP_idx_STC + 1) * dt
        v_APs = v_APs[np.random.randint(0, len(v_APs), 100)]  # TODO

        if len(v_APs) > 10:
            # STC
            eigvals, eigvecs, expl_var = get_stc(v_APs)
            chosen_eigvecs = choose_eigvecs(eigvecs, eigvals, n_eigvecs=3)
            back_projection = project_back(v_APs, chosen_eigvecs)
            plots_stc(v_APs, t_AP, back_projection, chosen_eigvecs, expl_var, save_dir_img)

            # Group by AP_max
            mean_high, std_high, mean_low, std_low, _, _ = group_by_AP_max(v_APs)
            plot_group_by_AP_max(mean_high, std_high, mean_low, std_low, t_AP, save_dir_img)

            # ICA
            ica = FastICA(n_components=3, whiten=True)
            ica_source = ica.fit_transform(v_APs_centered)
            plot_ICA(v_APs, t_AP, ica.mixing_, save_dir_img)

            plot_all_in_one(v_APs, t_AP, back_projection, mean_high, std_high, mean_low, std_low,
                            chosen_eigvecs, expl_var, ica.mixing_, save_dir_img)

        pl.close('all')