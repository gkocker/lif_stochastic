import numpy as np
import os, pickle
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, minimize_scalar
from scipy.ndimage import gaussian_filter
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

from src.model import phi
from src.theory import rate_fn, rate_fn_neg
from src.sim import sim_lif_perturbation, sim_lif_perturbation_x, sim_lif_pop, create_spike_train

fontsize = 10
labelsize = 9

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = ['k']+colors


def plot_fig_exc_inh_weakly_coupled(savefile='../results/fig../results/fig3.pdf'):

    # fig, ax = plt.subplots(2, 9, figsize=(3.4, 3.7))
    fig = plt.figure(figsize=(3.4, 3.7))
    nrows = 11
    gs = GridSpec(2, nrows, figure=fig)

    ax1 = fig.add_subplot(gs[0, :nrows//2])

    ### E-J phase diagram

    Npts = 1000
    E = np.linspace(-1, 1, Npts)
    Jbound = 2 + 2*np.sqrt(1-E)

    J = np.linspace(0, 8, Npts)
    JJ, EE = np.meshgrid(J, E)

    gbound = (JJ-2)/JJ-2*np.sqrt(1-EE)/JJ

    for i in range(gbound.shape[1]):
        ind = np.where(J < 2 + 2*np.sqrt(1-E[i]) - .01)
        gbound[i,ind] = np.nan

    ax1.plot(Jbound, E, 'k')
    im = ax1.imshow(gbound, origin='lower', extent=(0,max(J),-1,1), aspect=max(J)/2.5, alpha=0.5, clim=(0, .6), cmap='Greys')

    ax1.plot(J, np.ones(Npts), 'k')

    ax1.text(s='H', x=5, y=1.1, fontsize=fontsize, horizontalalignment='center')
    ax1.text(s='L', x=1.5, y=0.4, fontsize=fontsize, verticalalignment='center')
    ax1.text(s='L\nor\nB', x=5., y=.5, fontsize=fontsize, verticalalignment='center')

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.01)
    fig.colorbar(im, cax=cax, orientation='vertical', label='Max g for B', drawedges=False, ticks=[0, 0.3, 0.6])

    # cax = fig.add_subplot(gs[0, nrows//2])
    # plt.colorbar(im, cax=cax, label='Max g for B', drawedges=False, ticks=[0, 0.6], shrink=0.9)

    ax1.set_xlabel('J', fontsize=fontsize)
    ax1.set_ylabel('E', fontsize=fontsize)
    ax1.set_xlim((0, max(J)))
    ax1.set_ylim((-1, 2))

    # ### J-g phase diagram
    # # ax2 = fig.add_subplot(222)
    # ax2 = fig.add_subplot(gs[0, nrows//2+1:])
    # Npts = 1000
    # J = np.linspace(0, 10, Npts)

    # Evec = [-.5, 0, .5, .99]

    # for E in Evec:

    #     Jmin = 2 + 2*np.sqrt(1-E)
    #     gbound = (J-2)/J - 2*np.sqrt(1-E)/J

    #     # plt.plot([Jmin, Jmin], [0, 6], 'k')
    #     ax2.plot(J, gbound, label=r'$E={}$'.format(E))

    # ax2.legend(loc=0, frameon=False, fontsize=fontsize)
    # ax2.set_xlabel('J', fontsize=fontsize)
    # ax2.set_ylabel('g', fontsize=fontsize)
    # ax2.set_ylim((0, 1.5))

    ### E-g phase diagram
    ax2 = fig.add_subplot(gs[0, nrows//2+1:])
    Npts = 1000
    g = np.linspace(0, 2, Npts)

    ax2.plot(g, np.ones(Npts,), 'k')

    Jvec = [6]
    E = np.arange(-1, 1, .01)

    for J in Jvec:
        gbound = (J-2)/J - 2*np.sqrt((1-E) / J**2)
        ax2.plot(gbound, E, 'k', label='J={}'.format(J))

    # ax2.legend(loc=0, frameon=False, fontsize=fontsize)
    ax2.set_xlabel('g', fontsize=fontsize)
    ax2.set_ylabel('E', fontsize=fontsize)

    ax2.text(x=.2, y=0, s='B', va='center', ha='center', fontsize=fontsize)
    ax2.text(x=.5, y=0, s='L', va='center', ha='center', fontsize=fontsize)
    ax2.text(x=.5, y=1.5, s='H', va='center', ha='center', fontsize=fontsize)

    ax2.set_xlim((0, 1))
    ax2.set_ylim((-1, 2))

    ### sims for two regions

    ax3 = fig.add_subplot(gs[1, :nrows//2])

    J = 6
    g = 0.3
    E = 0.5

    Ne = 200
    Ni = 50
    N = Ne + Ni
    pE = 0.2
    pI = 0.8

    Jmat = np.zeros((N, N))
    Jmat[:, :Ne] = np.random.binomial(n=1, p=pE, size=(N,Ne)) * J / pE / Ne
    Jmat[:, Ne:] = np.random.binomial(n=1, p=pI, size=(N,Ni)) * (-g * J) / pI / Ni

    print(np.mean(Jmat[:, :Ne]))
    print(np.mean(Jmat[:, Ne:]))

    tstop = 20
    dt = .01
    tplot = np.arange(0, tstop, dt)
    E = -.5
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

    _, spktimes = sim_lif_perturbation(J=Jmat, E=E, tstop=tstop, dt=dt, perturb_len=perturb_len, perturb_amp=perturb_amp)
    ax3.plot(spktimes[:, 0], spktimes[:, 1], 'k|', markersize=0.5)
    ax2.plot(J, g, 'ko')

    ax4 = fig.add_subplot(gs[1, nrows//2+1:])
    E = 0.5
    _, spktimes = sim_lif_perturbation(J=Jmat, E=E, tstop=tstop, dt=dt, perturb_len=perturb_len, perturb_amp=perturb_amp)
    ax4.plot(spktimes[:, 0], spktimes[:, 1], 'k|', markersize=0.5)


    raster_yticks = (0, Ne, N)
    raster_yticklabels = (0, Ne, N)

    for axi in (ax3, ax4):
        axi.plot(tplot, E_plot, 'k')
        axi.set_xlim((0, tstop))
        axi.set_ylim((0, N+Eshift+3.5*Escale))
        axi.set_yticks(raster_yticks)
        axi.set_xlabel('Time (ms/{})'.format(r'$\tau$'), fontsize=fontsize)
    
    ax3.set_ylabel('Neuron', fontsize=fontsize)
    ax3.set_yticklabels(raster_yticklabels)
    ax4.set_yticklabels([])

    for axi in (ax1, ax2, ax3, ax4):
        axi.tick_params(axis='x', labelsize=labelsize)
        axi.tick_params(axis='y', labelsize=labelsize)

    # fig.tight_layout()
    sns.despine(fig)
    fig.savefig(savefile)


def plot_fig_exc_inh_bifurcation(savefile='../results/fig../results/fig3_bif.pdf', gbounds=(0, 1.8), Emax=2, eps=1e-11):    

    fig, ax = plt.subplots(1, 2, figsize=(3.4, 2))

    E = 1.2
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
    r_mft_high = phi(v_mft_high)

    ax[0].plot(grange, r_sim, 'ko', alpha=0.5)
    ax[0].plot(grange, r_sim_stim, 'ko', alpha=0.5)

    ax[0].plot(grange_th, r_mft_low, 'k', linewidth=2)
    ax[0].plot(grange_th, r_mft_high, 'k--', linewidth=2, label='1st ord.')
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


    Erange_th = np.arange(-1, Emax, .01)

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

    Ebif = np.where( (J > 2*np.sqrt(1-Erange_th) + 2) * (g <= (J-2)/J - 2 * np.sqrt(1-Erange_th)/J) )[0]

    v_mft_high[Ebif] = ( J*(1-g) + np.sqrt(4*Erange_th[Ebif] + (g-1)*J*((g-1)*J + 4)) ) / 2

    Ebif = np.where(Erange_th > 0)[0]
    v_mft_high[Ebif] = ( J*(1-g) + np.sqrt(4*Erange_th[Ebif] + (g-1)*J*((g-1)*J + 4)) ) / 2

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


def dv(v, E, J, g, p=1):
    
    if len(E) == 1:
        Ee = E
        Ei = E
    elif len(E) == 2:
        Ee, Ei = E
    else:
        raise Exception('input E has length {}, needs to be 1 or 2'.format(len(E)))
    
    ve, vi = v
    
    dve = -ve - ve*phi(ve) + J*phi(ve, p=p) - g*J*phi(vi, p=p) + Ee
    dvi = -vi - vi*phi(vi) + J*phi(ve, p=p) - g*J*phi(vi, p=p) + Ei
    
    return np.array([dve, dvi])


def dv_wc(v):
    
    if len(E) == 1:
        Ee = E
        Ei = E
    elif len(E) == 2:
        Ee, Ei = E
    else:
        raise Exception('input E has length {}, needs to be 1 or 2'.format(len(E)))
    
    ve, vi = v
    
    dve = -ve + J*phi(ve, p=p) - g*J*phi(vi, p=p) + Ee
    dvi = -vi + J*phi(ve, p=p) - g*J*phi(vi, p=p) + Ei
    
    return np.array([dve, dvi])


def phase_plane_plot(model, ax, range_x = (-1,1), range_y = None,
                     num_grid_points = 50, num_quiv_points = None, show = False, E=(2, 2), g=0.5, J=6):
    '''
    Simple implementation of the phase plane plot in matplotlib.
    
    Input:
    -----
      *model* : function
        function that takes numpy.array as input with two elements
        representing two state variables
      *range_x* = (-1, 1) : tuple
        range of x axis
      *range_y* = None : tuple
        range of y axis; if None, the same range as *range_x*
      *num_grid_points* = 50 : int
        number of samples on grid
      *show* = False : bool
        if True it shows matplotlib plot
    '''
    if range_y is None:
        range_y = range_x
    
    if num_quiv_points is None:
        num_quiv_points = num_grid_points
    
    x_ = np.linspace(range_x[0], range_x[1], num_quiv_points)                                                             
    y_ = np.linspace(range_y[0], range_y[1], num_quiv_points)                                                             

    grid = np.meshgrid(x_, y_)

    dfmat = np.zeros((num_quiv_points, num_quiv_points, 2))
    for nx in range(num_quiv_points):
        for ny in range(num_quiv_points):
            df = model([grid[0][nx,ny], grid[1][nx,ny]], E=E, J=J, g=g)
            dfmat[nx, ny, 0] = df[0]
            dfmat[nx, ny, 1] = df[1]

    ax.quiver(grid[0], grid[1], dfmat[:, :, 0], dfmat[:, :, 1], headwidth=5)    
    
    x_ = np.linspace(range_x[0], range_x[1], num_grid_points)                                                             
    y_ = np.linspace(range_y[0], range_y[1], num_grid_points)                                                             

    grid = np.meshgrid(x_, y_)

    dfmat = np.zeros((num_grid_points, num_grid_points, 2))
    for nx in range(num_grid_points):
        for ny in range(num_grid_points):
            df = model([grid[0][nx,ny], grid[1][nx,ny]], E=E, J=J, g=g)
            dfmat[nx, ny, 0] = df[0]
            dfmat[nx, ny, 1] = df[1]
    
    ax.contour(grid[0], grid[1], dfmat[:, :, 0], [0], colors=colors[1], label='E')
    ax.contour(grid[0], grid[1], dfmat[:, :, 1], [0], colors=colors[2], label='I')


def plot_fig_exc_inh_asymmetric_input(savefile='../results/fig4.pdf'):


    fig = plt.figure(figsize=(3.4, 3.7))
    gs = GridSpec(2, 3, figure=fig)

    ax1 = fig.add_subplot(gs[1, :])

    ### example paradoxical response from monostable regime
    J = 6
    g = 0.5
    E = 2

    p = 1
    pE = 0.5
    pI = 0.8

    Ne = 500
    Ni = 200
    N = Ne + Ni

    tstop = 200
    dt = .01

    perturb_ind = range(Ne, N)
    perturb_len = int(tstop / 3)
    perturb_amp = (0, 1.5, 3)

    Jmat = np.zeros((N, N))
    Jmat[:, :Ne] = np.random.binomial(n=1, p=pE, size=(N,Ne)) * J / pE / Ne
    Jmat[:, Ne:] = np.random.binomial(n=1, p=pI, size=(N,Ni)) * -g * J / pI / Ni
    
    _, spktimes = sim_lif_perturbation_x(J=Jmat, E=E, tstop=tstop, dt=dt, perturb_ind=perturb_ind, perturb_amp=perturb_amp)

    ax1.plot(spktimes[:, 0]-50, spktimes[:, 1], 'k|', markersize=.1)

    Nt = int(tstop/dt)
    spk_e = np.zeros(Nt,)
    spk_i = np.zeros(Nt,)

    for i in range(Ne):
        spk_e += create_spike_train(spktimes, neuron=i, dt=dt, tstop=tstop)

    for i in range(Ne, N):
        spk_i += create_spike_train(spktimes, neuron=i, dt=dt, tstop=tstop)

    r_e = gaussian_filter(spk_e, sigma=2/dt) / N
    r_i = gaussian_filter(spk_i, sigma=2/dt) / N

    ax2 = ax1.twinx()
    tplot = np.arange(-50, tstop-50, dt)
    # ax2.plot(tplot, r_e, linewidth=2, label='exc.', color=colors[1])
    ax2.plot(tplot, r_i, linewidth=2, label='inh.', color=colors[2])

    # t_start_perturb1 = Nt//3
    # t_end_perturb1 = t_start_perturb1 + int(perturb_len / dt)
    
    # t_start_perturb2 = 2*Nt//3
    # t_end_perturb2 = t_start_perturb2 + int(perturb_len / dt)
    
    # scale = 40
    # Eplot = (N + scale) * np.ones((Nt,))
    # Eplot[t_start_perturb1:t_end_perturb1] += scale*1.5
    # Eplot[t_end_perturb1:] += scale*3
    # ax1.plot(tplot, Eplot, 'k', linewidth=2)

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

    # E = (2, 3)
    # phase_plane_plot(dv, ax3, range_x=range_x, show=True, num_grid_points=200, num_quiv_points=0, E=E, J=J, g=g)

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


def plot_fig_paradoxical_response(savefile='../results/fig5.pdf'):

    fig, ax = plt.subplots(2, 2, figsize=(3.4, 3.7))

    ### contour plot of the region with monostable paradoxical responses

    J = 4
    h = 1

    Npts = 1000
    E = np.linspace(0, 6, Npts)
    g = np.linspace(0, 3, Npts)

    gg, EE = np.meshgrid(g, E)

    fun1 = 2*(EE + gg*J - 1)/(gg*J) + gg*J - np.sqrt((gg*J)**2 + 4*J*gg + 4*h*EE)
    fun2 = np.sqrt((gg*J)**2 + 4*J*gg + 4*h*EE + 2*J*(J-2)) - 2*(EE + J*(1+gg) - J**2 / 4)/(gg*J) + gg*J

    fun2[fun1 < 0] = np.inf

    # ax[0, 0].imshow(fun1, extent=(min(g), max(g), max(E), min(E)), origin='lower', aspect='auto', clim=(-1, 1))
    ax[0, 0].contour(g, E, fun1, [0], colors=colors[0])
    ax[0, 0].contour(g, E, fun2, [0], colors=colors[0])

    J = 2
    fun1 = 2*(EE + gg*J - 1)/(gg*J) + gg*J - np.sqrt((gg*J)**2 + 4*J*gg + 4*h*EE)
    fun2 = np.sqrt((gg*J)**2 + 4*J*gg + 4*h*EE + 2*J*(J-2)) - 2*(EE + J*(1+gg) - J**2 / 4)/(gg*J) + gg*J

    fun2[fun1 < 0] = np.inf

    # ax[0, 0].contour(g, E, fun1, [0], colors=colors[0])
    ax[0, 0].contour(g, E, fun2, [0], colors=colors[0], linestyles='dashed')
    ax[0, 0].set_ylim((0, max(E)))
    ax[0, 0].text(s='P', x=2*max(g)/3, y=max(E)/2, ha='center', va='center', fontsize=fontsize)

    ### sims for paradoxical response
    J = 4

    p = 1
    pE = 0.5
    pI = 0.8

    Ne = 500
    Ni = 200
    N = Ne + Ni

    tstop = 200
    dt = .01
    trans = 10

    perturb_ind = range(Ne, N)
    perturb_amp = (0.1, )
    Nperturb = len(perturb_amp)
    perturb_len = tstop // (Nperturb + 1)

    Npts = 12
    simfile = '../results/sim_paradox_loop_g_E_J={}_perturb_amp={}_Npts={}.pkl'.format(J, perturb_amp[0], Npts)

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
    im1 = ax[0, 1].imshow(r_diff, origin='lower', extent=(gmin-dg/2, gmax-dg/2, Emin-dE/2, Emax-dE/2), clim=(-cmax, cmax), cmap='bwr', aspect='auto')
    fig.colorbar(im1, ax=ax[0, 1], shrink=0.8)

    ax[0, 1].set_xlim((gmin-dg/2, gmax-dg/2))
    ax[0, 1].set_ylim((0, Emax-dE/2))
    ax[0, 0].set_ylim((0, Emax-dE/2))

    ax[0, 1].set_title(r'$r_i$'+', stim - spont', fontsize=fontsize)
    ax[0, 1].set_xlabel('g', fontsize=fontsize)
    ax[0, 0].set_xlabel('g', fontsize=fontsize)
    ax[0, 0].set_ylabel('E', fontsize=fontsize)

    ### E vs J with fixed g

    g = 2
    h = 1

    Npts = 1000
    E = np.linspace(0, 6, Npts)
    J = np.linspace(0, 6, Npts)

    JJ, EE = np.meshgrid(J, E)

    fun1 = 2*(EE + g*JJ - 1)/(g*JJ) + g*JJ - np.sqrt((g*JJ)**2 + 4*JJ*g + 4*h*EE)
    fun2 = np.sqrt((g*JJ)**2 + 4*JJ*g + 4*h*EE + 2*JJ*(JJ-2)) - 2*(EE + JJ*(1+g) - JJ**2 / 4)/(g*JJ) + g*JJ

    ax[1, 0].contour(J, E, fun1, [0], colors=colors[0])
    ax[1, 0].contour(J, E, fun2, [0], colors=colors[0])

    ax[1, 0].text(s='P', x=2*max(J)/3, y=max(E)/2, ha='center', va='center', fontsize=fontsize)


    ### sims for E vs J
    Npts = 12

    simfile = '../results/sim_paradox_loop_J_E_g={}_perturb_amp={}_Npts={}.pkl'.format(g, perturb_amp[0], Npts)
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
        p = 1
        pE = 0.5
        pI = 0.8

        Ne = 500
        Ni = 200
        N = Ne + Ni

        tstop = 200
        dt = .01
        trans = 10

        Npts = 12
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
    im2 = ax[1, 1].imshow(r_diff, origin='lower', extent=(Jmin-dJ/2, Jmax-dJ/2, Emin-dE/2, Emax-dE/2), clim=(-cmax, cmax), cmap='bwr')

    fig.colorbar(im2, ax=ax[1, 1], shrink=0.8)

    ax[1, 1].set_xlim((Jmin-dJ/2, Jmax-dJ/2))
    ax[1, 1].set_ylim((0, Emax-dE/2))
    ax[1, 0].set_ylim((0, Emax-dE/2))

    ax[1, 0].set_ylabel('E', fontsize=fontsize)
    ax[1, 0].set_xlabel('J', fontsize=fontsize)
    ax[1, 1].set_xlabel('J', fontsize=fontsize)

    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(savefile)

    return None


if __name__ == '__main__':
    # plot_fig_single_pop_weakly_coupled()
    # plot_fig_exc_inh_weakly_coupled()

    # plot_fig_exc_inh_bifurcation()

    plot_fig_exc_inh_asymmetric_input(savefile='../results/fig4_raster.png')
    # plot_fig_paradoxical_response(savefile='../results/fig5_cbar.pdf')