from __future__ import division
import os
import matplotlib.pyplot as pl
import numpy as np
from cell_fitting.data.plot_IV import check_v_at_i_inj_0_is_at_right_sweep_idx
from cell_characteristics.analyze_APs import get_AP_onset_idxs, get_AP_max_idx
from cell_characteristics import to_idx
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj_from_function
from itertools import combinations
from sklearn.decomposition import FastICA
pl.style.use('paper')


def get_sta(v_APs):
    sta = np.mean(v_APs, 0)
    sta_std = np.std(v_APs, 0)
    return sta, sta_std


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


def plots_sta(v_APs, t_AP, sta, sta_std, save_dir_img):
    v_APs_plots = v_APs[np.random.randint(0, len(v_APs), 50)]  # reduce to lower number

    pl.figure()
    pl.title('AP Traces (# %i)' % len(v_APs), fontsize=18)
    for v_AP in v_APs_plots:
        pl.plot(t_AP, v_AP)
    pl.ylabel('Membrane potential (mV)')
    pl.xlabel('Time (ms)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'STA_APs.png'))

    pl.figure()
    pl.title('STA', fontsize=18)
    pl.plot(t_AP, sta, 'k')
    pl.fill_between(t_AP, sta + sta_std, sta - sta_std,
                    facecolor='k', alpha=0.5)
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane Potential (mV)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'STA.png'))


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
    pl.plot(np.arange(len(expl_var)), np.cumsum(expl_var), 'ok', markersize=8)
    pl.ylabel('Percent')
    pl.xlabel('#')
    pl.ylim(0, 105)
    pl.tight_layout()                           
    pl.savefig(os.path.join(save_dir_img, 'STC_eigenvals.png'))


def plot_all_in_one(v_APs, t_AP, back_projection, mean_high, std_high, mean_low, std_low, chosen_eigvecs, expl_var,
                    ica_components, save_dir_img):

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
                    facecolor='r', alpha=0.7)
    axes[1, 0].fill_between(t_AP, mean_low + std_low, mean_low - std_low,
                    facecolor='b', alpha=0.7)
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
    for i, vec in enumerate(ica_components.T):
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
    AP_max_high = AP_max >= AP_max_sort[int(round(len(AP_max_sort) * (1. / 2.)))]
    AP_max_low = AP_max < AP_max_sort[int(round(len(AP_max_sort) * (1. / 2.)))]
    mean_high = np.mean(v_APs[AP_max_high], 0)
    std_high = np.std(v_APs[AP_max_high], 0)
    mean_low = np.mean(v_APs[AP_max_low], 0)
    std_low = np.std(v_APs[AP_max_low], 0)
    return mean_high, std_high, mean_low, std_low


def plot_ICA(v_APs, t_AP, ica_components, save_dir_img):
    fig, ax = pl.subplots()
    ax.set_title('ICA Components', fontsize=18)
    ax2 = ax.twinx()
    ax2.plot(t_AP, np.mean(v_APs, 0), 'k')
    ax2.set_ylabel('Membrane Potential (mV)')
    ax2.spines['right'].set_visible(True)
    for vec in ica_components.T:
        if vec[0] < 0:
            vec *= -1
        ax.plot(t_AP, vec)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('ICA Component')
    ax.legend(loc='lower right', fontsize=10)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'ICA_components.png'))


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
        t_AP = np.arange(after_AP_idx_STC + before_AP_idx_STC + 1) * dt
        v_APs = v_APs[np.random.randint(0, len(v_APs), 100)]  # TODO

        if len(v_APs) > 10:
            # STC
            eigvals, eigvecs, expl_var = get_stc(v_APs)
            chosen_eigvecs = choose_eigvecs(eigvecs, eigvals, n_eigvecs=3)
            back_projection = project_back(v_APs, chosen_eigvecs)
            plots_stc(v_APs, t_AP, back_projection, chosen_eigvecs, expl_var, save_dir_img)

            # Group by AP_max
            mean_high, std_high, mean_low, std_low = group_by_AP_max(v_APs)
            plot_group_by_AP_max(mean_high, std_high, mean_low, std_low, t_AP, save_dir_img)

            # ICA
            ica = FastICA(n_components=3)
            ica_components = ica.fit_transform(v_APs.T)
            plot_ICA(v_APs, t_AP, ica_components, save_dir_img)

            plot_all_in_one(v_APs, t_AP, back_projection, mean_high, std_high, mean_low, std_low,
                            chosen_eigvecs, expl_var, ica_components, save_dir_img)

        pl.close('all')