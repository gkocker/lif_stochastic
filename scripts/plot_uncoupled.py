import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root, root_scalar, minimize_scalar
from scipy.signal import fftconvolve
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

from src.model import intensity, intensity_match_linear_reset_mft
from src.theory import lif_linear_full_fI, lif_linear_1loop_fI
from src.sim import sim_lif_pop, create_spike_train

from plot_fluctuations import calc_avg_spectrum

fontsize = 10
labelsize = 9

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = ['k']+colors

root_dir = '/Users/gkocker/Documents/projects/lif_stochastic'
results_dir = os.path.join(root_dir, 'results')


def sim_lif_1neuron(J, E, tstop=100, dt=.01, B=1, v_th=1, p=1, v_r=0, tstim=0, Estim=0):

    Nt = int(tstop / dt)
    Ntstim = int(tstim / dt)

    if len(np.shape(J)) > 1:
        N = np.shape(J)[0]
    else:
        N = 1
    
    if len(np.shape(E)) == 0:
        E = E * np.ones(N,)
    elif len(E) < N:
        raise Exception('Need either a scalar or length N input E')

    v = np.zeros((Nt,N))
    n = np.zeros(N,)
    spktimes = []

    for t in range(1, Nt):

        if t < Ntstim:
            Et = Estim
        else:
            Et = E

        v[t] = v[t-1] + dt*(-v[t-1] + Et) + np.dot(J, n) - n*(v[t-1]-v_r)

        lam = intensity(v[t], B=B, v_th=v_th, p=p)
        if lam > 1/dt:
            lam = 1/dt
            
        n = np.random.binomial(n=1, p=dt*lam)

        spkind = np.where(n > 0)[0]
        for i in spkind:
            spktimes.append([t, i])
            
    spktimes = np.array(spktimes).astype(np.float32)
    return v, spktimes


def sim_linear_reset_1neuron(J, E, intensity_fun, tstop=100, dt=.01, B=1, v_th=1, p=1, v_r=0, tstim=0, Estim=0):

    Nt = int(tstop / dt)
    Ntstim = int(tstim / dt)

    if len(np.shape(J)) > 1:
        N = np.shape(J)[0]
    else:
        N = 1
    
    if len(np.shape(E)) == 0:
        E = E * np.ones(N,)
    elif len(E) < N:
        raise Exception('Need either a scalar or length N input E')

    v = np.zeros((Nt,N))
    n = np.zeros(N,)
    spktimes = []

    for t in range(1, Nt):

        if t < Ntstim:
            Et = Estim
        else:
            Et = E

        v[t] = v[t-1] + dt*(-v[t-1] + Et) + np.dot(J, n) - n*v_th

        lam = intensity_fun(v[t], B=B, v_th=v_th, p=p)
        if lam > 1/dt:
            lam = 1/dt
            
        n = np.random.binomial(n=1, p=dt*lam)

        spkind = np.where(n > 0)[0]
        for i in spkind:
            spktimes.append([t, i])
            
    spktimes = np.array(spktimes).astype(np.float32)
    return v, spktimes


def sim_1neuron_match_times_linear_reset(J, E, spktimes, tstop=100, dt=.01, B=1, v_th=1, p=1):
    
    Nt = int(tstop / dt)
    
    if len(np.shape(J)) > 1:
        N = np.shape(J)[0]
    else:
        N = 1
    
    if len(np.shape(E)) == 0:
        E = E * np.ones(N,)
    elif len(E) < N:
        raise Exception('Need either a scalar or length N input E')

    v = np.zeros((Nt,))
    n = np.zeros(1,)

    spkind = 0
    
    for t in range(1, Nt):

    #     v[t] = (1-n)*v[t-1] + dt*(1-n)*(-v[t-1] + b[t-1])
        v[t] = v[t-1] + dt*(-v[t-1] + E + np.dot(J, n))
            
        if (spkind < len(spktimes)) and (spktimes[spkind, 0] == t):
            v[t] -= E
            spkind += 1
            
    return v


def plot_fig_intro_uncoupled(savefile=os.path.join(results_dir, 'fig1.pdf')):

    fig = plt.figure(figsize=(3.4, 3.7)) # single column, double column is ~7in

    gs = fig.add_gridspec(4, 2)

    ax1 = fig.add_subplot(gs[0, 0]) # first voltage trace
    ax2 = fig.add_subplot(gs[1, 0], sharey=ax1, sharex=ax1) # second voltage trace
    ax3 = fig.add_subplot(gs[:2, 1]) # rate-input curves for the three models
    ax4 = fig.add_subplot(gs[2:4, 0]) # mean field, one-loop, and renewal rate-input curves for the stochastic LIF
    ax5 = fig.add_subplot(gs[2:4, 1]) # one-loop vs mean field for the three models

    E = 3
    v_th = 1
    p = 1
    dt = .01
    tstop = 20

    v, spktimes = sim_lif_1neuron((0), E, tstop=tstop, dt=dt, p=p, v_th=v_th)
    v_lin = sim_1neuron_match_times_linear_reset((0), E, spktimes, tstop=tstop, dt=dt, p=p, v_th=v_th)

    tplot = np.arange(0, tstop, dt)
    ax1.plot(spktimes[:, 0]*dt, 1.2*E*np.ones(len(spktimes)), 'k|', markersize=6)
    ax1.plot(tplot, v, color=colors[0], linewidth=1)
    ax2.plot(tplot, v_lin, color=colors[1], linewidth=1)

    ax1.xaxis.set_visible(False)
    ax1.set_xticks(())
    ax1.set_yticks(())
    ax2.set_yticks(())
    ax2.set_xlabel('Time (ms/{})'.format(r'$\tau$'), fontsize=fontsize)
    ax2.set_ylabel('Membrane voltage\n(norm.)', fontsize=fontsize)

    dE = .5
    v_th = 1
    Erange = np.arange(v_th-dE, 4+dE, dE)
    tstop = 1000
    dt = .1

    p = 1

    ### plot the mean-field rates for three models: the LIF, the linear reset, and the linear reset with matched mean-field voltage

    rates_lif = []
    rates_linear = []
    rates_linear_match_v = []
    var_lif = []
    var_linear = []
    var_linear_match_v = []

    for E in Erange:
        v, spktimes = sim_lif_1neuron((0), E, tstop=tstop, dt=dt, p=1)
        rates_lif.append(len(spktimes) / tstop)

        if len(spktimes) > 0:
            spktimes[:, 0] *= dt
            wsim, Csim = calc_avg_spectrum(spktimes, N=1, tstop=tstop, dt=dt)
            var_lif.append(Csim[0])
        else:
            var_lif.append(0)

        v, spktimes = sim_linear_reset_1neuron((0), E, intensity, tstop=tstop, dt=dt, p=1)
        rates_linear.append(len(spktimes) / tstop)

        if len(spktimes) > 0:
            spktimes[:, 0] *= dt
            wsim, Csim = calc_avg_spectrum(spktimes, N=1, tstop=tstop, dt=dt)
            var_linear.append(Csim[0])
        else:
            var_linear.append(0)

        v, spktimes = sim_linear_reset_1neuron((0), E, intensity_match_linear_reset_mft, tstop=tstop, dt=dt, p=1)
        rates_linear_match_v.append(len(spktimes) / tstop)
        if len(spktimes) > 0:
            spktimes[:, 0] *= dt
            wsim, Csim = calc_avg_spectrum(spktimes, N=1, tstop=tstop, dt=dt)
            var_linear_match_v.append(Csim[0])
        else:
            var_linear_match_v.append(0)

    Erange_th = np.arange(min(Erange), max(Erange), .01)
    rates_lif_mft, rates_lif_1loop = lif_linear_1loop_fI(Erange_th, p=p, v_th=v_th)
    rates_lif_full = lif_linear_full_fI(Erange_th)
    rates_linear_mft = intensity((Erange_th+1)/2)
    rates_linear_match_mft = intensity_match_linear_reset_mft(np.sqrt(Erange_th))

    ax3.plot(Erange, rates_lif, 'o', color=colors[0], alpha=0.3)
    ax3.plot(Erange, rates_linear, 'o', color=colors[1], alpha=0.3)
    ax3.plot(Erange, rates_linear_match_v, 'o', color=colors[2], alpha=0.3)

    ax3.plot(Erange_th, rates_lif_mft, color=colors[0])
    ax3.plot(Erange_th, rates_linear_mft, color=colors[1])
    ax3.plot(Erange_th, rates_linear_match_mft, color=colors[2])
    ax3.set_ylabel('Rate (norm.)', fontsize=fontsize)

    ### plot the mean-field, 1-loop, and exact rates for the LIF
    vbar_1loop = (1+np.sqrt(1+80*Erange_th)) / 10
    rates_lif_1loop = intensity(vbar_1loop)
    # rates_lif_1loop[vbar_1loop < 1] = 0

    ax4.plot(Erange, rates_lif, 'o', color=colors[0], alpha=0.3)
    ax4.plot(Erange_th, rates_lif_mft, ':', color=colors[0], label='mean field')
    ax4.plot(Erange_th, rates_lif_1loop, '--', color=colors[0], label='1 loop')
    ax4.plot(Erange_th, rates_lif_full, '-', color=colors[0], label='exact')

    ax4.set_xlabel('E', fontsize=fontsize)
    ax4.set_ylabel('Rate (norm.)', fontsize=fontsize)
    ax4.legend(loc=0, frameon=False, fontsize=fontsize)

    ### plot one-loop correction for the three models
    # C = 9 - 36*Erange_th + np.sqrt(3)*np.sqrt(59 - 216*Erange_th + 432*(Erange_th**2))
    r_match_1loop = (np.sqrt(Erange_th)+1/4)*(np.sqrt(Erange_th)-1)    
    r_match_1loop[Erange_th < 1] = 0.

    # r_match_1loop = (1+phibar) * phi_2/2 * phibar * phi_1**2 / (1+phibar+phi_1*vbar)**3
    # r_match_1loop[vbar < 1] = 0

    ax5.plot(Erange_th, rates_lif_1loop - rates_lif_mft, color=colors[0])
    ax5.plot(Erange_th, 0*Erange_th, color=colors[1])
    ax5.plot(Erange_th, r_match_1loop - rates_linear_match_mft, color=colors[2])

    ax5.set_xlabel('E', fontsize=fontsize)
    ax5.set_ylabel('1 loop correction (norm.)', fontsize=fontsize)

    for axi in (ax1, ax2, ax3, ax4, ax5):
        axi.tick_params(axis='x', labelsize=labelsize)
        axi.tick_params(axis='y', labelsize=labelsize)

    for axi in (ax2, ax3, ax4, ax5):
        axi.set_xticks((1, 3))

    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(savefile)


if __name__ == '__main__':
    plot_fig_intro_uncoupled()