import numpy as np
import os, pickle
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, minimize_scalar
from scipy.ndimage import gaussian_filter
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

from src.model import intensity
from src.theory import rate_fn, rate_fn_neg
from src.sim import sim_lif_perturbation, sim_lif_perturbation_x, sim_lif_pop, create_spike_train
from src.phase_plane import dv, phase_plane_plot

fontsize = 10
labelsize = 9

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = ['k']+colors

root_dir = '/Users/gkocker/Documents/projects/lif_stochastic'
results_dir = os.path.join(root_dir, 'results')


def plot_fig_exc_inh_weakly_coupled(savefile=os.path.join(results_dir, 'fig_ei.pdf')):

    ### figure 3

    fig, ax = plt.subplots(3, 2, figsize=(3.4, 5))

    ### E-g phase diagram
    Npts = 1000
    g = np.linspace(0, 2, Npts)

    ax[0, 1].plot(g, np.ones(Npts,), 'k')

    Jvec = [6]
    E = np.arange(-1, 1, .01)

    for i, J in enumerate(Jvec):
        gbound = 1 - 2*(1+np.sqrt(1-E))/J

        ax[0, 1].plot(gbound, E, ':', color=colors[i], label='mean field')

        # gbound = (J-2)/J + 2*np.sqrt((1-E) / J**2)
        # ax[0, 1].plot(gbound, E, ':', color=colors[i])

        # gbound = 1 - 2/J - 4/np.sqrt(3)*np.sqrt((1-E)/(J**2))
        gbound = 1 - 9/(4*J) -np.sqrt(5) * np.sqrt(1-E) / J

        ax[0, 1].plot(gbound, E, color=colors[i], label='1 loop')

        # gbound = 1 - 2/J + 4/np.sqrt(3)*np.sqrt((1-E)/(J**2))
        # ax[0, 1].plot(gbound, E, color=colors[i])

    # ax2.legend(loc=0, frameon=False, fontsize=fontsize)
    ax[0, 1].set_xlabel('g', fontsize=fontsize)
    ax[0, 1].set_ylabel('E', fontsize=fontsize)
    ax[0, 1].legend(loc=0, frameon=False, fontsize=fontsize)

    ax[0, 1].text(x=.2, y=0, s='B', va='center', ha='center', fontsize=fontsize)
    ax[0, 1].text(x=.5, y=0, s='L', va='center', ha='center', fontsize=fontsize)
    ax[0, 1].text(x=.5, y=1.5, s='H', va='center', ha='center', fontsize=fontsize)

    ax[0, 1].set_xlim((0, 1.))
    ax[0, 1].set_ylim((-1, 2))

    ### E-J phase diagram
    g = .3
    J = np.arange(0, 10, .001)
    E_min_mft = (J*(1-g)/4)*(4-J*(1-g))
    Jpeak = np.argmin(np.abs(E_min_mft - 1))
    E_min_mft[:Jpeak] = np.nan

    E_min_1loop = (-1 - 8*(-1 + g)*J*(9 + 2*(-1 + g)*J)) / 80
    Jpeak = np.argmin(np.abs(E_min_1loop - 1))
    E_min_1loop[:Jpeak] = np.nan

    ax[1, 0].plot(J, E_min_mft, 'k:')
    ax[1, 0].plot(J, E_min_1loop, 'k')
    ax[1, 0].plot(J, np.ones(len(J)), 'k')

    ax[1, 0].set_xlabel('J', fontsize=fontsize)
    ax[1, 0].set_ylabel('E', fontsize=fontsize)
    ax[1, 0].set_ylim((-1, 2))
    ax[1, 0].set_xlim((0, 10))

    ### J-g phase diagram

    E = .5
    J = np.arange(0, 10, .001)

    ### mean field
    gbound_mft = (-2 + J)/J - 2*np.sqrt((1-E))/J
    ax[1, 1].plot(gbound_mft, J, 'k:', label='mean field')
    
    gbound_1loop = -np.sqrt(5) * np.sqrt((1-E))/J + (-9+4*J)/(4*J)
    ax[1, 1].plot(gbound_1loop, J, 'k', label='1 loop')

    g_plot = np.arange(-2, 2, .001)
    J_plot = np.arange(0, 10, .001)

    ax[1, 1].set_xlabel('g', fontsize=fontsize)
    ax[1, 1].set_ylabel('J', fontsize=fontsize)
    # ax[1, 1].set_title('E = {}'.format(E), fontsize=fontsize)
    # plt.legend(loc=0, frameon=False)
    ax[1, 1].set_xlim((0, 1))
    ax[1, 1].set_ylim((0, 10))

    ### sims for two regions

    J = 6
    g = 0.3
    E = 0.5

    ax[0, 1].plot(g, E, 'ko')
    ax[1, 0].plot(J, E, 'ko')
    ax[1, 1].plot(g, J, 'ko')

    Ne = 200
    Ni = 50
    N = Ne + Ni
    pE = 0.2
    pI = 0.8

    Jmat = np.zeros((N, N))
    Jmat[:, :Ne] = np.random.binomial(n=1, p=pE, size=(N,Ne)) * J / pE / Ne
    Jmat[:, Ne:] = np.random.binomial(n=1, p=pI, size=(N,Ni)) * (-g * J) / pI / Ni

    tstop = 20
    dt = .01
    tplot = np.arange(0, tstop, dt)
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

    Eshift = 20
    Escale = 10
    E_plot = Escale*E_plot + N + Eshift

    E = -.5
    _, spktimes = sim_lif_perturbation(J=Jmat, E=E, tstop=tstop, dt=dt, perturb_len=perturb_len, perturb_amp=perturb_amp)
    ax[2, 0].plot(spktimes[:, 0], spktimes[:, 1], 'k|', markersize=0.5)

    ax[0, 1].plot(g, E, 'ks')
    ax[1, 0].plot(J, E, 'ks')
    
    E = 0.5

    _, spktimes = sim_lif_perturbation(J=Jmat, E=E, tstop=tstop, dt=dt, perturb_len=perturb_len, perturb_amp=perturb_amp)
    ax[2, 1].plot(spktimes[:, 0], spktimes[:, 1], 'k|', markersize=0.5)

    raster_yticks = (0, Ne, N)
    raster_yticklabels = (0, Ne, N)

    for axi in ax[2]:
        axi.plot(tplot, E_plot, 'k')
        axi.set_xlim((0, tstop))
        axi.set_ylim((0, N+Eshift+3.5*Escale))
        axi.set_yticks(raster_yticks)
        axi.set_xlabel('Time (ms/{})'.format(r'$\tau$'), fontsize=fontsize)

    ### formatting

    fig.tight_layout()
    sns.despine(fig)
    fig.savefig(savefile)


def plot_fig_exc_inh_bifurcation(savefile=os.path.join(results_dir, 'fig_ei_bifurcation.pdf'), gbounds=(0, 1.), Emax=2, eps=1e-11):    

    ### figure 4 c, d

    fig, ax = plt.subplots(1, 2, figsize=(3.4, 2))

    E = 0.5
    Ne = 200
    Ni = 50
    N = Ne + Ni

    tstop = 500
    tstim = 10
    Estim = 10

    pE = 0.5
    pI = 0.8

    J = 6
    gmin, gmax = gbounds
    grange = np.linspace(gmin, gmax, 6)

    r_sim = []
    r_sim_stim = []

    for g in grange:

        Jmat = np.zeros((N, N))
        Jmat[:, :Ne] = np.random.binomial(n=1, p=pE, size=(N,Ne)) * J / pE / Ne
        Jmat[:, Ne:] = np.random.binomial(n=1, p=pI, size=(N,Ni)) * -g * J / pI / Ni

        _, spktimes = sim_lif_pop(J=Jmat, E=E, tstop=tstop, tstim=0, Estim=0)

        r_sim.append(len(spktimes) / N / tstop)
        
        _, spktimes = sim_lif_pop(J=Jmat, E=E, tstop=tstop, tstim=tstim, Estim=Estim)
        
        spktimes = spktimes[spktimes[:, 0] > 2*tstim, :]
        if len(spktimes) > 0:
            r_sim_stim.append(len(spktimes) / N / (tstop - 2*tstim))
        else:
            r_sim_stim.append(np.nan)
        
    grange_th = np.linspace(gmin, gmax, 100)

    r_th = []

    n_max = 20

    for g in grange_th:

        result1 = minimize_scalar(rate_fn_neg, args=(J*(1-g), E)) # find the local maximum of n(n) - n
        
        if result1.success and (result1.fun < 0):
            
            n_min = result1.x
            result2 = root_scalar(rate_fn, args=(J*(1-g), E), method='bisect', bracket=(n_min, n_max))
            
            if result2.converged:
                r_th.append(result2.root)
            else:
                r_th.append(np.nan)
            
        else:
            r_th.append(0)

    r_mft_low = 0 * grange_th
    v_mft_high = 0 * grange_th
    r_mft_high = v_mft_high.copy()

    r_mft_low = 0 * grange_th
    v_mft_high = 0 * grange_th
    r_mft_high = v_mft_high.copy()

    gbif = np.where( (J > 2*np.sqrt(1-E) + 2) * (grange_th <= (J-2)/J - 2 * np.sqrt(1-E)/J) )[0]

    v_mft_high[gbif] = ( J*(1-grange_th[gbif]) + np.sqrt(4*E + (grange_th[gbif]-1)*J*((grange_th[gbif]-1)*J + 4)) ) / 2
    r_mft_high = intensity(v_mft_high)

    v_1loop = (1 + 4*J*(1-grange_th) + np.sqrt(1 + 80*E + 8*J*(1-grange_th)*(2*J*(1-grange_th)-9))) / 10
    r_1loop = intensity(v_1loop)

    ax[0].plot(grange, r_sim, 'ko', alpha=0.5)
    ax[0].plot(grange, r_sim_stim, 'ko', alpha=0.5)

    ax[0].plot(grange_th, r_mft_low, 'k', linewidth=2)
    ax[0].plot(grange_th, r_mft_high, 'k:', linewidth=2, label='mean field')
    ax[0].plot(grange_th, r_1loop, 'k--', linewidth=2, label='1 loop')
    ax[0].plot(grange_th, r_th, 'k', linewidth=2, label='exact')

    ax[0].legend(loc=0, frameon=False, fontsize=fontsize)
    ax[0].set_xlabel('g', fontsize=fontsize)
    ax[0].set_ylabel('Rate ({} spk / ms)'.format(r'$\tau$'), fontsize=fontsize)


    g = 0.25
    Erange = np.linspace(-1, Emax, 6)

    r_sim = []
    r_sim_stim = []

    Jmat = np.zeros((N, N))
    Jmat[:, :Ne] = np.random.binomial(n=1, p=pE, size=(N,Ne)) * J / pE / Ne
    Jmat[:, Ne:] = np.random.binomial(n=1, p=pI, size=(N,Ni)) * -g * J / pI / Ni

    for ei, E in enumerate(Erange):

        _, spktimes = sim_lif_pop(J=Jmat, E=E, tstop=tstop, tstim=0, Estim=0)
        r_sim.append(len(spktimes) / N / tstop)
        
        if E < 1:
            _, spktimes = sim_lif_pop(J=Jmat, E=E, tstop=tstop, tstim=tstim, Estim=Estim)
            
            spktimes = spktimes[spktimes[:, 0] > 2*tstim, :]
            if len(spktimes) > 0:
                r_sim_stim.append(len(spktimes) / N / (tstop - 2*tstim))
            else:
                r_sim_stim.append(np.nan)
        else:
            r_sim_stim.append(np.nan) # already have activity, don't need stim to kick out of low state


    Erange_th = np.arange(-1, Emax, .001)

    r_th = []
    n_max = 10

    for ei, E in enumerate(Erange_th):

        result1 = minimize_scalar(rate_fn_neg, args=(J*(1-g), E))
        
        if result1.success and (result1.fun < 0):
            
            n_min = result1.x
            result2 = root_scalar(rate_fn, args=(J*(1-g), E), method='bisect', bracket=(n_min, n_max))
            
            if result2.converged:
                r_th.append(result2.root)
            else:
                r_th.append(np.nan)
            
        else:
            r_th.append(0)


    r_mft_low = np.nan * Erange_th
    r_mft_low[Erange_th < 1] = 0 
    v_mft_high = 0 * Erange_th
    r_mft_high = v_mft_high.copy()

    # Ebif = np.where( (J > 2*np.sqrt(1-Erange_th) + 2) * (g <= (J-2)/J - 2 * np.sqrt(1-Erange_th)/J) )[0]

    Ebif = np.where( (Erange_th > 1 - ( (J-2)**2)/4) * (Erange_th > 1 - (J**2)/4 * ((J-2)/J - g)**2 ) )[0]

    v_mft_high[Ebif] = ( J*(1-g) + np.sqrt(4*Erange_th[Ebif] + (g-1)*J*((g-1)*J + 4)) ) / 2

    Ebif = np.where(Erange_th > 0)[0]
    v_mft_high[Ebif] = ( J*(1-g) + np.sqrt(4*Erange_th[Ebif] + (g-1)*J*((g-1)*J + 4)) ) / 2

    r_mft_high = intensity(v_mft_high)


    v_1loop = (1 + 4*J*(1-g) + np.sqrt(1 + 80*Erange_th + 8*J*(1-g)*(2*J*(1-g)-9))) / 10
    r_1loop = intensity(v_1loop)

    # v_1loop = J*(1-g) / 2 + 2/3 * np.sqrt((3/4*J*(1-g))**2-9/4*J*(1-g) + 3*Erange_th - 3/4)
    # r_1loop = 3/4 * intensity(v_1loop)

    ax[1].plot(Erange, r_sim, 'ko', alpha=0.5)
    ax[1].plot(Erange, r_sim_stim, 'ko', alpha=0.5)
    ax[1].plot(Erange_th, r_mft_low, 'k', linewidth=2)
    ax[1].plot(Erange_th, r_mft_high, 'k:', linewidth=2)
    ax[1].plot(Erange_th, r_1loop, 'k--', linewidth=2)
    ax[1].plot(Erange_th, r_th, 'k', linewidth=2)

    ax[1].set_xlabel('E', fontsize=fontsize)
    ax[1].set_ylabel('Rate ({} spk / ms)'.format(r'$\tau$'), fontsize=fontsize)

    fig.tight_layout()
    sns.despine(fig)
    fig.savefig(savefile)


def plot_fig_paradoxical_response_example(Ne=500, Ni=200, pE=0.5, pI=0.8, J=6, g=0.5, E=2, tstop=200, dt=.01, savefile=os.path.join(results_dir, 'fig_paradox_example.pdf')):

    ### figure 5

    fig = plt.figure(figsize=(3.4, 3.7))
    gs = GridSpec(2, 3, figure=fig)

    ax1 = fig.add_subplot(gs[1, :])

    ### example paradoxical response from monostable regime
    N = Ne + Ni

    perturb_ind = range(Ne, N)
    perturb_len = int(tstop / 3)
    perturb_amp = (0, 1.5, 3)

    Jmat = np.zeros((N, N))
    Jmat[:, :Ne] = np.random.binomial(n=1, p=pE, size=(N,Ne)) * J / pE / Ne
    Jmat[:, Ne:] = np.random.binomial(n=1, p=pI, size=(N,Ni)) * -g * J / pI / Ni
    
    _, spktimes = sim_lif_perturbation_x(J=Jmat, E=E, tstop=tstop, dt=dt, perturb_ind=perturb_ind, perturb_amp=perturb_amp)

    ax1.plot(spktimes[:, 0]-50, spktimes[:, 1], 'k|', markersize=.1)

    Nt = int(tstop/dt)
    spk_i = np.zeros(Nt,)

    for i in range(Ne, N):
        spk_i += create_spike_train(spktimes, neuron=i, dt=dt, tstop=tstop)

    r_i = gaussian_filter(spk_i, sigma=2/dt) / N

    ax2 = ax1.twinx()
    tplot = np.arange(-50, tstop-50, dt)
    ax2.plot(tplot, r_i, linewidth=2, label='inh.', color=colors[2])

    t_start_perturb1 = Nt//3
    t_end_perturb1 = t_start_perturb1 + int(perturb_len / dt)
    
    t_start_perturb2 = 2*Nt//3
    
    scale = 40
    Eplot = (N + scale) * np.ones((Nt,))
    Eplot[t_start_perturb1:t_end_perturb1] += scale*1.5
    Eplot[t_end_perturb1:] += scale*3
    ax1.plot(tplot, Eplot, 'k', linewidth=2)

    ax2.set_ylabel('Rate ({} spk / ms'.format(r'$\tau$'))

    # ax2.legend(loc=0, frameon=False, fontsize=fontsize)
    ax1.set_xlim((0, tstop-50))
    ax2.set_ylim((.15, .35))
    ax2.set_yticks((.2, .3))
    # ax1.set_ylim((0, N + scale*5))
    ax1.set_yticks((0, Ne, N))

    ax1.set_ylim((0, N))
    ax1.set_yticks((0, Ne, N))

    ax1.set_ylabel('Neuron', fontsize=fontsize)
    ax2.set_ylabel('Rate ({} spk / ms)'.format(r'$\tau$'), fontsize=fontsize)
    ax2.set_xlabel('Time (ms / {})'.format(r'$\tau$'))

    sns.despine(ax=ax1, right=False)
    sns.despine(ax=ax2, right=False)

    ## phase plane
    ax3 = fig.add_subplot(gs[0, 1:])

    range_x = (0, 3.5)
    E = (2, 2)
    phase_plane_plot(dv, ax3, range_x=range_x, show=True, num_grid_points=200, num_quiv_points=7, E=E, J=J, g=g)

    E = (2, 5)
    phase_plane_plot(dv, ax3, range_x=range_x, show=True, num_grid_points=200, num_quiv_points=0, E=E, J=J, g=g)

    ax3.text(x=3.5, y=2.9, s=r'$E_i=2$', fontsize=fontsize)
    ax3.text(x=3.5, y=3.5, s=r'$E_i=5$', fontsize=fontsize)

    ax3.set_xlabel(r'$v_e$', fontsize=fontsize)
    ax3.set_ylabel(r'$v_i$', fontsize=fontsize)
    sns.despine(ax=ax3)

    fig.tight_layout()
    fig.savefig(savefile, dpi=600)


    return None


def plot_fig_paradoxical_response(Ne=500, Ni=200, pE=0.5, pI=0.8, tstop=5000, dt=.01, trans=10, Npts=12, savefile=os.path.join(results_dir, 'fig_paradox.pdf')):


    ### figure 6

    fig, ax = plt.subplots(1, 2, figsize=(3.4, 2))

    ### sims for paradoxical response
    J = 4
    N = Ne + Ni

    perturb_ind = range(Ne, N)
    perturb_amp = (0.1, )
    Nperturb = len(perturb_amp)
    perturb_len = tstop // (Nperturb + 1)

    simfile = os.path.join(results_dir, 'sim_paradox_loop_g_E_J={}_perturb_amp={}_Npts={}.pkl'.format(J, perturb_amp[0], Npts))
    print(simfile)

    if os.path.exists(simfile):
        with open(simfile, 'rb') as f:
            results = pickle.load(f)

        r_diff = results['r_diff']
        E_vec = results['E_vec']
        g_vec = results['g_vec']

        Emin, Emax = min(E_vec), max(E_vec)
        gmin, gmax = min(g_vec), max(g_vec)

    else:
        print('running sims')
        Emin, Emax = 1, 6
        gmin, gmax = 0, 3
        E_vec = np.linspace(Emin, Emax, Npts)
        g_vec = np.linspace(gmin, gmax, Npts)
        
        r_diff = np.zeros((Npts, Npts))

        for j, g in enumerate(g_vec):

            Jmat = np.zeros((N, N))
            Jmat[:, :Ne] = np.random.binomial(n=1, p=pE, size=(N,Ne)) * J / pE / Ne
            Jmat[:, Ne:] = np.random.binomial(n=1, p=pI, size=(N,Ni)) * -g * J / pI / Ni

            for i, E in enumerate(E_vec):

                # run sim, compute I rate
                _, spktimes = sim_lif_perturbation_x(J=Jmat, E=E, perturb_amp=perturb_amp, perturb_ind=perturb_ind, tstop=tstop, dt=dt)

                if len(spktimes) > 0:
                    spktimes_I = spktimes[spktimes[:, 1] >= Ne][:, 0] # only times

                    rI_pre = len ( np.where(spktimes_I < perturb_len)[0] ) / perturb_len / Ni
                    rI_post = len ( np.where(spktimes_I >= perturb_len+trans)[0] ) / (perturb_len - trans) / Ni
                    r_diff[i, j] = rI_post - rI_pre
                    
        r_diff /= Ni

        results = {}
        results['r_diff'] = r_diff
        results['E_vec'] = E_vec
        results['g_vec'] = g_vec

        with open(simfile, 'wb') as f:
            pickle.dump(results, f)

    cmax = np.amax(np.abs(r_diff)) / 3
    dE = E_vec[1] - E_vec[0]
    dg = g_vec[1] - g_vec[0]
    im1 = ax[0].imshow(r_diff, origin='lower', extent=(gmin-dg/2, gmax-dg/2, Emin-dE/2, Emax-dE/2), clim=(-cmax, cmax), cmap='bwr', aspect='auto')
    fig.colorbar(im1, ax=ax[1], shrink=0.8)

    ### contour plot of the region with monostable paradoxical responses

    J = 4
    h = 1

    Npts = 1000
    E = np.linspace(-.5, 6, Npts)
    g = np.linspace(-.5, 3, Npts)

    gg, EE = np.meshgrid(g, E)

    ### mean field boundaries
    fun1 = 2*(EE + gg*J - 1)/(gg*J) + gg*J - np.sqrt((gg*J)**2 + 4*J*gg + 4*h*EE)
    fun2 = (-gg*J + np.sqrt((gg*J)**2 + 2*J**2 + 4*J*gg - 4*J + 4*h*EE ) )/2 - (J**2/4 - J*(1-gg) + EE)/(gg*J)
    # fun2[fun1 < 0] = np.inf

    # ax[0, 0].imshow(fun1, extent=(min(g), max(g), max(E), min(E)), origin='lower', aspect='auto', clim=(-1, 1))
    ax[0].contour(g, E, fun1, [0], linestyles='dashed', colors=colors[0])
    con = ax[0].contour(g, E, fun2, [0], linestyles='dashed', colors=colors[0], label='mean field')
    con.collections[0].set_label('mean field')

    # ax[0].clabel(con, con.levels, inline=True, fmt='mean\nfield', fontsize=fontsize-2)

    ### one-loop boundaries
    fun1 = 1+80*EE - 8*J*(9-2*J - gg*(9+4*gg*J -np.sqrt(1+80*EE*h+8*J*(-9+4*J +gg*(9+2*gg*J))))) # peak
    fun2 = 10 + gg*J*(-9 - 4*gg*J + np.sqrt(1 + 80*h*EE + 8*gg*J*(9 + 2*gg*J))) - 10*EE # threshold

    ax[0].contour(g, E, fun1, [0], colors=colors[0])
    con = ax[0].contour(g, E, fun2, [0], colors=colors[0], label='1 loop')
    con.collections[0].set_label('1 loop')

    ax[0].set_xlim((0, gmax-dg/2))
    ax[0].set_ylim((0, Emax-dE/2))
    # ax[0].legend(loc=0, frameon=False, fontsize=fontsize)
    ax[0].set_xlabel('g', fontsize=fontsize)
    ax[0].set_ylabel('E', fontsize=fontsize)

    ### E vs J with fixed g

    g = 2
    h = 1

    ### sims for E vs J
    Npts = 12

    simfile = os.path.join(results_dir, 'sim_paradox_loop_J_E_g={}_perturb_amp={}_Npts={}.pkl'.format(g, perturb_amp[0], Npts))
    if os.path.exists(simfile):
        with open(simfile, 'rb') as f:
            results = pickle.load(f)

        r_diff = results['r_diff']
        E_vec = results['E_vec']
        J_vec = results['J_vec']

        Emin, Emax = min(E_vec), max(E_vec)
        Jmin, Jmax = min(J_vec), max(J_vec)

    else:
        print('running sims')
        # p = 1

        Emin, Emax = 1, 6
        Jmin, Jmax = 0, 6
        E_vec = np.linspace(Emin, Emax, Npts)
        J_vec = np.linspace(Jmin, Jmax, Npts)

        r_diff = np.full((Npts, Npts), np.nan)

        for j, J in enumerate(J_vec):

            Jmat = np.zeros((N, N))
            Jmat[:, :Ne] = np.random.binomial(n=1, p=pE, size=(N,Ne)) * J / pE / Ne
            Jmat[:, Ne:] = np.random.binomial(n=1, p=pI, size=(N,Ni)) * -g * J / pI / Ni

            for i, E in enumerate(E_vec):

            
                # run sim, compute I rate
                _, spktimes = sim_lif_perturbation_x(J=Jmat, E=E, perturb_amp=perturb_amp, perturb_ind=perturb_ind, tstop=tstop, dt=dt)

                if len(spktimes) > 0:
                    spktimes_I = spktimes[spktimes[:, 1] >= Ne][:, 0] # only times

                    rI_pre = len ( np.where(spktimes_I < perturb_len)[0] ) / perturb_len
                    rI_post = len ( np.where(spktimes_I >= perturb_len+trans)[0] ) / (perturb_len - trans)
                    r_diff[i, j] = rI_post - rI_pre
                    
        r_diff /= Ni

        results = {}
        results['r_diff'] = r_diff
        results['E_vec'] = E_vec
        results['J_vec'] = J_vec

        with open(simfile, 'wb') as f:
            pickle.dump(results, f)

    cmax = np.amax(np.abs(r_diff)) / 3
    dE = E_vec[1] - E_vec[0]
    dJ = J_vec[1] - J_vec[0]
    im2 = ax[1].imshow(r_diff, origin='lower', extent=(Jmin-dJ/2, Jmax-dJ/2, Emin-dE/2, Emax-dE/2), clim=(-cmax, cmax), cmap='bwr')

    # fig.colorbar(im2, ax=ax[1, 1], shrink=0.8)

    Npts = 1000
    E = np.linspace(-.5, 6, Npts)
    J = np.linspace(-.5, 6, Npts)

    JJ, EE = np.meshgrid(J, E)

    ### mean-field boundaries
    fun1 = 2*(EE + g*JJ - 1)/(g*JJ) + g*JJ - np.sqrt((g*JJ)**2 + 4*JJ*g + 4*h*EE)
    fun2 = (-g*JJ + np.sqrt((g*JJ)**2 + 2*JJ**2 + 4*JJ*g - 4*JJ + 4*h*EE ) )/2 - (JJ**2/4 - JJ*(1-g) + EE)/(g*JJ)

    ax[1].contour(J, E, fun1, [0], linestyles='dashed', colors=colors[0])
    ax[1].contour(J, E, fun2, [0], linestyles='dashed', colors=colors[0])

    ### one-loop boundaries
    fun1 = 1+80*EE - 8*JJ*(9-2*JJ - g*(9+4*g*JJ -np.sqrt(1+80*EE*h+8*JJ*(-9+4*JJ +g*(9+2*g*JJ))))) # peak
    fun2 = 10 + g*JJ*(-9 - 4*g*JJ + np.sqrt(1 + 80*h*EE + 8*g*JJ*(9 + 2*g*JJ))) - 10*EE # threshold

    ax[1].contour(J, E, fun1, [0], colors=colors[0])
    ax[1].contour(J, E, fun2, [0], colors=colors[0])

    # ax[1].text(s='P', x=2*max(J)/3, y=max(E)/2, ha='center', va='center', fontsize=fontsize)


    ax[1].set_xlim((0, Jmax-dJ/2))
    ax[1].set_ylim((0, Emax-dE/2))
    ax[0].set_title(r'$r_i$'+', stim - spont', fontsize=fontsize)

    # ax[1, 0].set_ylabel('E', fontsize=fontsize)
    ax[1].set_xlabel('J', fontsize=fontsize)
    
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(savefile)

    return None


if __name__ == '__main__':

    # plot_fig_exc_inh_weakly_coupled()

    # plot_fig_exc_inh_bifurcation()

    # plot_fig_paradoxical_response_example()

    plot_fig_paradoxical_response(savefile=os.path.join(results_dir, 'fig_paradox_cbar.pdf'))