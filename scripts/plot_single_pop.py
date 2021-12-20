import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import seaborn as sns

from src.model import phi
from src.theory import lif_rate_homog, fixed_pt_iter_propagators_1pop, fixed_pt_iter_propagators_1pop_true
from src.sim import sim_lif_perturbation, sim_lif_pop, create_spike_train

fontsize = 10
labelsize = 9

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = ['k']+colors

root_dir = '/Users/gabeo/Documents/projects/path_lif'
results_dir = os.path.join(root_dir, 'results')

def plot_fig_single_pop_weakly_coupled(savefile='../results/fig_homog.png', Jmax=10, Emax=2):

    fig, ax = plt.subplots(2, 2, figsize=(3.4, 3.7))

    ### first, the phase diagram

    J = np.arange(2, Jmax, .01)
    Ebound = J*(4-J)/4

    ax[0, 0].plot(J, Ebound, 'k')

    J = np.arange(-4, Jmax, .01)
    ax[0, 0].plot(J, np.ones(len(J)), 'k')

    # ### shade two regions
    # ### v_- and v_+ both exist: bistability
    # J = np.arange(2, Jmax, .01)
    # Ebound = J*(4-J)/4
    # ax[0, 0].fill_between(J, Ebound, np.ones(len(J)), color=colors[1], alpha=0.5)

    # ### v_+ only
    # J = np.arange(-1, Jmax, .01)
    # plt.fill_between(J, np.ones(len(J)), Emax*np.ones(len(J)), color=colors[2], alpha=0.5)

    ax[0, 0].set_xlabel('J', fontsize=fontsize)
    ax[0, 0].set_ylabel('E', fontsize=fontsize)

    ax[0, 0].text(x=0, y=0, s='L', va='center', ha='center', fontsize=fontsize)
    ax[0, 0].text(x=3, y=1.5, s='H', va='center', ha='center', fontsize=fontsize)
    ax[0, 0].text(x=8, y=0, s='B', va='center', ha='center', fontsize=fontsize)

    ax[0, 0].set_ylim((-1, Emax))
    ax[0, 0].set_xlim((-4, Jmax))
    ax[0, 0].set_yticks(range(-1, Emax))

    ### simulation examples of bistability

    N = 100
    connection_prob = 0.5

    tstop = 20
    dt = .01
    tplot = np.arange(0, tstop, dt)
    E = 0
    perturb_amp = 2
    perturb_len = 2

    Nt = int(tstop/dt)
    E_plot = np.zeros(Nt,) + E
    t_start_perturb1 = Nt//4
    t_end_perturb1 = t_start_perturb1 + int(perturb_len / dt)

    t_start_perturb2 = 3*Nt//4
    t_end_perturb2 = t_start_perturb2 + int(perturb_len / dt)

    E_plot[t_start_perturb1:t_end_perturb1] += perturb_amp
    E_plot[t_start_perturb2:t_end_perturb2] -= perturb_amp

    Eshift = 15
    Escale = 5
    E_plot = Escale*E_plot + N + Eshift

    E = 1.5
    g = 5
    J = np.random.binomial(n=1, p=connection_prob, size=(N,N)) * g / connection_prob / N
    _, spktimes = sim_lif_perturbation(J=J, E=E, tstop=tstop, dt=dt, perturb_len=perturb_len, perturb_amp=perturb_amp)

    ax[0, 0].plot(g, E, 'ko')
    ax[0, 1].plot(spktimes[:, 0], spktimes[:, 1], 'k|', markersize=.5)
    # ax[0, 1].plot(tstop/2, N+Eshift+perturb_amp*Escale, 'ko')

    E = 0
    g = 2.5
    J = np.random.binomial(n=1, p=connection_prob, size=(N,N)) * g / connection_prob / N
    _, spktimes = sim_lif_perturbation(J=J, E=E, tstop=tstop, dt=dt, perturb_len=perturb_len, perturb_amp=perturb_amp)

    ax[0, 0].plot(g, E, 'ks')
    ax[1, 0].plot(spktimes[:, 0], spktimes[:, 1], 'k|', markersize=.5)
    # ax[1, 0].plot(tstop/2, N+Eshift+perturb_amp*Escale, 'ks')


    g = 5
    J = np.random.binomial(n=1, p=connection_prob, size=(N,N)) * g / connection_prob / N

    _, spktimes = sim_lif_perturbation(J=J, E=E, tstop=tstop, dt=dt, perturb_len=perturb_len, perturb_amp=perturb_amp)

    ax[0, 0].plot(g, E, 'kX')
    ax[1, 1].plot(spktimes[:, 0], spktimes[:, 1], 'k|', markersize=.5)
    # ax[1, 1].plot(tstop/2, N+Eshift+perturb_amp*Escale, 'kX')

    raster_yticks = (0, N)
    raster_yticklabels = (0, N)

    for axi in (ax[0, 1], ax[1, 0], ax[1, 1]):
        axi.plot(tplot, E_plot, 'k')
        axi.set_xlim((0, tstop))
        axi.set_ylim((0, N+Eshift+3.5*Escale))
        axi.set_yticks(raster_yticks)
        axi.set_yticklabels(raster_yticklabels)
        axi.set_xlabel('Time (ms/{})'.format(r'$\tau$'), fontsize=fontsize)
        # axi.set_ylabel('Neuron', fontsize=fontsize)

        # axi.set_xlabel('Time (ms/{})'.format(r'$\tau$'), fontsize=fontsize)
        # axi.set_ylabel('Neuron', fontsize=fontsize)

    # raster_yticks = (0, N, min(E_plot), E_plot[0], max(E_plot))
    # raster_yticklabels = (0, N, E-perturb_amp, E, E+perturb_amp)
    # print(perturb_amp)
    # ax[1, 1].set_yticks(raster_yticks)
    # ax[1, 1].set_yticklabels(raster_yticklabels)


    for axi in np.ravel(ax):

        axi.tick_params(axis='x', labelsize=labelsize)
        axi.tick_params(axis='y', labelsize=labelsize)

    fig.tight_layout()
    sns.despine(fig)

    fig.savefig(savefile, dpi=600)


def plot_fig_single_pop_bifurcation(savefile='../results/fig_homog_bif.pdf', Jbounds=(.5, 5.5), Emax=2, eps=1e-11):    

    fig, ax = plt.subplots(2, 1, figsize=(2, 3.7))

    E = .5
    connection_prob = 0.5
    N = 100
    tstop = 500
    tstim = 10
    Estim = 10

    Jmin, Jmax = Jbounds
    Jrange = np.arange(Jmin, Jmax, .5)

    r_sim = []
    r_sim_stim = []

    for g in Jrange:
        J = np.random.binomial(n=1, p=connection_prob, size=(N,N)) * g / connection_prob / N

        _, spktimes = sim_lif_pop(J=J, E=E, tstop=tstop, tstim=0, Estim=0)

        r_sim.append(len(spktimes) / N / tstop)
        
        _, spktimes = sim_lif_pop(J=J, E=E, tstop=tstop, tstim=tstim, Estim=Estim)
        
        spktimes = spktimes[spktimes[:, 0] > 2*tstim, :]
        if len(spktimes) > 0:
            r_sim_stim.append(len(spktimes) / N / (tstop - 2*tstim))
        else:
            r_sim_stim.append(np.nan)


    if E >= 1:
        Jrange_th = np.arange(Jmin, Jmax, .005)
    else:
        Jrange_th1 = np.linspace(Jmin, 2*(1+np.sqrt(1-E))-.01, 5)
        Jrange_th2 = np.arange(2*(1+np.sqrt(1-E))-.01, Jmax, .005)
        Jrange_th = np.concatenate((Jrange_th1, Jrange_th2))

    r_th = [lif_rate_homog(g, E) for g in Jrange_th]

    r_mft_low = 0 * Jrange_th
    v_mft_high = 0 * Jrange_th
    r_mft_high = v_mft_high.copy()

    Jbif = np.where(Jrange_th > 2*(1+np.sqrt(1-E)))[0][0]
    v_mft_high[Jbif:] = (Jrange_th[Jbif:] + np.sqrt(Jrange_th[Jbif:]**2 + 4*(E - Jrange_th[Jbif:]))) / 2
    r_mft_high = phi(v_mft_high)

    ax[0].plot(Jrange, r_sim, 'ko', alpha=0.5)
    ax[0].plot(Jrange, r_sim_stim, 'ko', alpha=0.5)

    ax[0].plot(Jrange_th, r_mft_low, 'k', linewidth=2)
    ax[0].plot(Jrange_th, r_mft_high, 'k--', linewidth=2, label='1st ord.')
    ax[0].plot(Jrange_th, r_th, 'k', linewidth=2, label='exact')

    ax[0].legend(loc=0, frameon=False, fontsize=fontsize)
    ax[0].set_xlabel('J', fontsize=fontsize)
    ax[0].set_ylabel('Rate ({} spk / ms)'.format(r'$\tau$'), fontsize=fontsize)

    g = 4
    Erange = np.linspace(-1, Emax, 6)

    r_sim = []
    r_sim_stim = []

    J = np.random.binomial(n=1, p=connection_prob, size=(N,N)) * g / connection_prob / N

    for E in Erange:

        _, spktimes = sim_lif_pop(J=J, E=E, tstop=tstop, tstim=0, Estim=0)
        r_sim.append(len(spktimes) / N / tstop)
        
        _, spktimes = sim_lif_pop(J=J, E=E, tstop=tstop, tstim=tstim, Estim=Estim)
        
        spktimes = spktimes[spktimes[:, 0] > 2*tstim, :]
        if len(spktimes) > 0:
            r_sim_stim.append(len(spktimes) / N / (tstop - 2*tstim))
        else:
            r_sim_stim.append(np.nan)


    Erange_th = np.arange(-1, Emax, .01)

    r_th = [lif_rate_homog(g, E) for E in Erange_th]

    r_mft_low = 0 * Erange_th
    r_mft_low[Erange_th > 1] = np.nan

    v_mft_high = 0 * Erange_th
    r_mft_high = v_mft_high.copy()

    Ebif = np.where(Erange_th > g*(4-g)/4)[0][0]

    v_mft_high[Ebif:] = (g + np.sqrt(g**2 + 4*(Erange_th[Ebif:] - g))) / 2
    r_mft_high = phi(v_mft_high)

    ax[1].plot(Erange, r_sim, 'ko', alpha=0.5)
    ax[1].plot(Erange, r_sim_stim, 'ko', alpha=0.5)
    ax[1].plot(Erange_th, r_mft_low, 'k', linewidth=2)
    ax[1].plot(Erange_th, r_mft_high, 'k--', linewidth=2)
    ax[1].plot(Erange_th, r_th, 'k', linewidth=2)

    ax[1].set_xlabel('E', fontsize=fontsize)
    ax[1].set_ylabel('Rate ({} spk / ms)'.format(r'$\tau$'), fontsize=fontsize)

    fig.tight_layout()
    sns.despine(fig)
    fig.savefig(savefile)


def plot_fig_two_pt(savefile='../results/fig5.pdf'):

    fig, ax = plt.subplots(2, 3, figsize=(5.4, 3.7))

    tstop = 1000
    dt = 0.01
    N = 100
    connection_prob = 0.5

    J = 1

    E_vec = (1.2, 10)
    # E_vec = np.arange(1, 3)

    for i, E in enumerate(E_vec):

        ### a: bare and dressed propagators

        w, Dnn_full, Dvn_full, Dnv_full, Dvv_full = fixed_pt_iter_propagators_1pop(J=J, E=E)

        ### define the mean-field theory
        if J == 0:
            vbar = np.sqrt(E)
        elif ((J * (4-J)/4) < E) and (J > 2):
            vbar = (J + np.sqrt(J**2 + 4*(E-J))) / 2
        elif (J <= 2) and (E > 1):
            vbar = (J + np.sqrt(J**2 + 4*(E-J))) / 2
        else:
            vbar = E
            
        phibar = phi(vbar)
        if vbar >= 1:
            phi_pr = 1
        else:
            phi_pr = 0

        ### bare propagators

        Dnn = (1 + phibar +1j * w) / (1 + phibar + phi_pr * vbar + 1j*w)
        Dvn = -phi_pr / (1 + phibar + phi_pr * vbar + 1j*w)
        Dnv = -vbar / (1 + phibar + phi_pr * vbar + 1j*w)
        Dvv = -1 / (1 + phibar + phi_pr * vbar + 1j*w)

        ax[0, 0].plot(w, np.abs(Dnn), '--', color=colors[i], linewidth=2)
        ax[0, 0].plot(w, np.abs(Dnn_full), color=colors[i], linewidth=2)

        ax[0, 1].plot(w, np.abs(Dnv), '--', color=colors[i], linewidth=2)
        ax[0, 1].plot(w, np.abs(Dnv_full), color=colors[i], linewidth=2)

        ax[1, 0].plot(w, np.abs(Dvn), '--', color=colors[i], linewidth=2)
        ax[1, 0].plot(w, np.abs(Dvn_full), color=colors[i], linewidth=2)

        ax[1, 1].plot(w, np.abs(Dvv), '--', color=colors[i], linewidth=2)
        ax[1, 1].plot(w, np.abs(Dvv_full), color=colors[i], linewidth=2)

        ### two-point function sim

        Jmat = np.random.binomial(n=1, p=connection_prob, size=(N,N)) * J / connection_prob / N
        _, spktimes = sim_lif_pop(J=Jmat, E=E, tstop=tstop, tstim=0, Estim=0)

        spk = create_spike_train(spktimes, neuron=0, tstop=tstop, dt=dt)
        f, C = welch(spk, fs=1/dt, scaling='density', window='hann', nperseg=2048)
        f *= 2*np.pi # angular frequency

        for j in range(1, N):
            spk = create_spike_train(spktimes, neuron=j, tstop=tstop, dt=dt)
            _, Ctmp = welch(spk, fs=1/dt, scaling='density', window='hann', nperseg=2048)
            C += Ctmp
        
        C /= N * 2 # factor of 2 from one-sided welch

        ax[0, 2].plot(f, C, '.', color=colors[i], linewidth=2, alpha=0.2)

        ### two-point function theory

        two_pt_a = phibar * Dnn * Dnn.conj() # diagram with the mft source
        two_pt_a_full = phibar * Dnn_full * Dnn_full.conj() # diagram with the mft source
        ax[0, 2].plot(w, np.abs(two_pt_a_full), '--', color=colors[i], linewidth=2)

        ### diagram with the correction source
        r_th = lif_rate_homog(J, E)

        _, Dnn_full, Dvn_full, Dnv_full, Dvv_full = fixed_pt_iter_propagators_1pop_true(J=J, E=E, w=w, max_its=100)
        two_pt_a_full = r_th * Dnn_full * Dnn_full.conj() # diagram with the mft source

        ax[0, 2].plot(w, np.abs(two_pt_a_full), '-', color=colors[i], linewidth=2)


        
        # dn = r_th - phibar
        
        # Dnv0 = -vbar / (1 + phibar + phi_pr * vbar)
        # two_pt_b = -J * phi_pr * dn * Dnn * Dnn.conj() * Dnv0

        # Dnv0_full = Dnv_full[np.argmin(np.abs(w))]
        # two_pt_b_full = -J * phi_pr * dn * Dnn_full * Dnn_full.conj() * Dnv0_full

        # ax[0, 2].plot(w, np.abs(two_pt_a), '--', color=colors[i], linewidth=2)
        # ax[0, 2].plot(w, np.abs(two_pt_a_full), '-', color=colors[i], linewidth=2)
        # ax[0, 2].plot(w, np.abs(two_pt_a_full + two_pt_b_full), '-', color=colors[i], linewidth=2)




    ax[0, 0].set_title(r'$\Delta_{\delta n, \tilde{n}}$', fontsize=fontsize)
    ax[0, 1].set_title(r'$\Delta_{\delta n, \tilde{v}}$', fontsize=fontsize)
    ax[1, 0].set_title(r'$\Delta_{\delta v, \tilde{n}}$', fontsize=fontsize)
    ax[1, 1].set_title(r'$\Delta_{\delta v, \tilde{v}}$', fontsize=fontsize)

    ax[1, 0].set_xlabel('Frequency (rad.)', fontsize=fontsize)
    ax[1, 1].set_xlabel('Frequency (rad.)', fontsize=fontsize)

    ax[1, 0].set_ylabel('Linear response magnitude', fontsize=fontsize)

    for axi in (ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1], ax[0, 2]):
        axi.set_xscale('log')
        # axi.set_ylim((0, 1))
        axi.set_xlim((.1, 100))
        axi.set_xticks((.1, 1, 10, 100))

    fig.tight_layout()
    sns.despine(fig)
    fig.savefig(savefile)

    return None


if __name__ == '__main__':
    # plot_fig_single_pop_weakly_coupled()
    # plot_fig_exc_inh_weakly_coupled()

    plot_fig_single_pop_bifurcation()

    # plot_fig_two_pt()