import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import welch
import seaborn as sns

from src.theory import lif_rate_homog, fixed_pt_iter_propagators_1pop_true
from src.sim import sim_lif_pop, create_spike_train

fontsize = 10
labelsize = 8

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = ['k']+colors

root_dir = '/Users/gkocker/Documents/projects/lif_stochastic'
results_dir = os.path.join(root_dir, 'results')

def calc_avg_spectrum(spktimes, N=None, tstop=None, dt=.01):

    if N is None:
        N = int(np.amax(spktimes[:, 1]))
    
    if tstop is None:
        tstop = np.amax(spktimes[:, 0])
    
    spk = create_spike_train(spktimes, neuron=0, dt=dt, tstop=tstop)
    spk -= np.mean(spk)
    wsim, Csim = welch(spk, fs=1/dt, scaling='density', window='bartlett', nperseg=2048, return_onesided=False, detrend=False)

    for i in range(1, N):
        spk = create_spike_train(spktimes, neuron=i, dt=dt, tstop=tstop)
        spk -= np.mean(spk)
        _, Ctmp = welch(spk, fs=1/dt, scaling='density', window='bartlett', nperseg=2048, return_onesided=False, detrend=False)

        Csim +=  Ctmp 
        
    Csim /= N

    return wsim, Csim


def calc_isi_dist(spktimes, N=None, smax=10, ds=0.5):

    if N is None:
        N = int(np.amax(spktimes[:, 1]))

    isi_bins = np.arange(0, smax, ds)
    p_isi = np.zeros((len(isi_bins)-1, ))

    for i in range(N):
        spki = spktimes[spktimes[:, 1]==i][:, 0]
        isi_i = np.diff(spki)
        
        p_isi += np.histogram(isi_i, bins=isi_bins, density=True)[0]

    p_isi /= N

    return isi_bins, p_isi


def calc_avg_isi_cv2(spktimes, N=None):

    if N is None:
        N = int(np.amax(spktimes[:, 1]))

    cv_isi = []

    if len(spktimes) > 0:
        for i in range(N):
            spki = spktimes[spktimes[:, 1]==i, 0]
            isi_i = np.diff(spki)
            cv_isi.append( np.var(isi_i) / np.mean(isi_i)**2)
    else:
        cv_isi.append(np.nan)

    return np.mean(cv_isi)    


def plot_ei_fluctuations(Ne=200, Ni=50, pE=0.5, pI=0.8, tstop=500, tstim=10, Estim=10, savefile=os.path.join(results_dir, 'fig_corrs.pdf'), gbounds=(0, .75), Emax=2):

    '''
    Plot figure 7.
    '''

    fig, ax = plt.subplots(3, 2, figsize=(3.4, 5))

    ### isi density and power spectrum for example parameters
    g = 0.3
    J = 6
    E = 1.2

    dt = .001 # want high freq
    tstop = tstop + 2*tstim # want low freq
    N = Ne + Ni

    Jmat = np.zeros((N, N))
    Jmat[:, :Ne] = np.random.binomial(n=1, p=pE, size=(N,Ne)) * J / pE / Ne
    Jmat[:, Ne:] = np.random.binomial(n=1, p=pI, size=(N,Ni)) * -g * J / pI / Ni

    _, spktimes = sim_lif_pop(J=Jmat, E=E, tstop=tstop, tstim=tstim, Estim=Estim, dt=dt)

    # truncate the transient stim that activates the network
    spktimes = spktimes[spktimes[:, 0] > 2*tstim, :]
    spktimes[:, 0] -= 2*tstim
    tstop -= 2*tstim

    wsim, Csim = calc_avg_spectrum(spktimes, N=N, tstop=tstop, dt=dt)
    isi_bins, p_isi = calc_isi_dist(spktimes, N=N, ds=0.25)

    isi_bin_diff = isi_bins[1] - isi_bins[0]
    ax[0, 0].plot(isi_bins[1:] - isi_bin_diff/2, p_isi, 'ko', alpha=0.5)

    wsim = np.fft.fftshift(wsim)
    Csim = np.fft.fftshift(Csim)
    ax[0, 1].plot(wsim*2*np.pi, Csim, 'ko', alpha=0.5)

    ### theory - rate
    r_th = lif_rate_homog(J*(1-g), E)
    if not (r_th > 0): raise Exception('Need positive rate')

    ### theory - isi dist
    EJn = E + J*(1-g)*r_th

    t0 = np.log(EJn) - np.log(EJn-1)

    ds = .001
    smax = 100
    smin = 0

    s = np.arange(smin, smax, ds)
    s0 = np.arange(t0, smax, ds)

    p1 = EJn*(1-np.exp(-s0)) - 1
    p2 = EJn*np.exp(-s0) + (EJn-1)*(s0-1-t0)
        
    p = np.zeros((len(s)))
    p[s > t0-ds] = p1 * np.exp(-p2)

    ax[0, 0].plot(s, p, 'k', linewidth=2)


    ### theory - power spectrum
    pw = np.fft.fft(p) # fourier transform of isi density
    w = np.fft.fftfreq(len(s))*2*np.pi/ds
    pw *= ds*np.exp(-complex(0,1)*w*smin) # phase factor

    Cw = r_th * (1 - np.abs(pw)**2) / (np.abs(1 - pw)**2)
    w = np.fft.fftshift(w)
    Cw = np.fft.fftshift(Cw)

    ax[0, 1].plot(w, Cw, 'k', linewidth=2, label='exact')

    ### theory - approximate power spectrum, tree 
    vbar = (J*(1-g) + np.sqrt((J*(1-g))**2 + 4*(E-J*(1-g)))) / 2
    f = vbar - 1
    if vbar < 1:
        f = 0
    Cw_tree = f * (vbar**2 + w**2) / (4*vbar**2 + w**2)
    # vbar = np.sqrt(J*(1-g)*r_th + E)
    # Cw_tree = (vbar - 1) * (vbar**2 + w**2) / (4*vbar**2 + w**2)
    ax[0, 1].plot(w, Cw_tree, 'k:', linewidth=2, label='mean field')

    ### theory - one-loop effective action
    vbar = (J*(1-g))/2 + 2/3*np.sqrt((3/4*J*(1-g))**2 - 9/4*J*(1-g) + 3*E - 3/4)
    f = 3/4*(vbar - 1)
    if vbar < 1:
        f = 0
    Cw_tree = f * (vbar**2 + w**2) / (4*vbar**2 + w**2)
    # vbar = np.sqrt(J*(1-g)*r_th + E)
    # Cw_tree = (vbar - 1) * (vbar**2 + w**2) / (4*vbar**2 + w**2)
    ax[0, 1].plot(w, Cw_tree, 'k--', linewidth=2, label='1 loop')

    ### theory - resummed propagators
    # w, Dnn_full, Dvn_full, Dnv_full, Dvv_full = fixed_pt_iter_propagators_1pop_true(J*(1-g), E)
    # Cw_tree = (vbar - 1) * Dnn_full * Dnn_full.conj()
    # ax[0, 1].plot(w, Cw_tree, 'k:', linewidth=2, label='tree, re-summed')

    ### spike train variance as function of g

    E = 0.5

    dt = .01 # want long timescale variance

    J = 6
    gmin, gmax = gbounds
    grange = np.linspace(gmin, gmax, 6)

    r_sim = []
    r_sim_stim = []
    var_sim = []
    var_sim_stim = []

    for gi, g in enumerate(grange):

        print('sims, g, {}/{}'.format(gi+1,len(grange)))

        Jmat = np.zeros((N, N))
        Jmat[:, :Ne] = np.random.binomial(n=1, p=pE, size=(N,Ne)) * J / pE / Ne
        Jmat[:, Ne:] = np.random.binomial(n=1, p=pI, size=(N,Ni)) * -g * J / pI / Ni

        _, spktimes = sim_lif_pop(J=Jmat, E=E, tstop=tstop, dt=dt, tstim=0, Estim=0)

        if len(spktimes) > 0:
            r_sim.append(len(spktimes) / N / np.amax(spktimes[:, 0]))
            _, Csim = calc_avg_spectrum(spktimes, tstop=tstop, dt=dt, N=N)
            var_sim.append(Csim[1])
        else:
            r_sim.append(0)
            var_sim.append(0)

        _, spktimes = sim_lif_pop(J=Jmat, E=E, tstop=tstop, dt=dt, tstim=tstim, Estim=Estim)
        spktimes = spktimes[spktimes[:, 0] > 2*tstim, :]
        spktimes[:, 0] -= 2*tstim

        if len(spktimes) > 0:
            r_sim_stim.append(len(spktimes) / N / (tstop - 2*tstim))
            _, Csim = calc_avg_spectrum(spktimes, tstop=tstop-2*tstim, dt=dt, N=N)
            var_sim_stim.append(Csim[1])
        else:
            r_sim_stim.append(np.nan)
            var_sim_stim.append(np.nan)

    grange_th = np.linspace(gmin, gmax, 200)

    r_th = []
    var_spk = []

    for gi, g in enumerate(grange_th):

        r_th_i = lif_rate_homog(J*(1-g), E)
        r_th.append(r_th_i)

        if r_th_i > 0:

            EJn = E + J*(1-g)*r_th_i
            t0 = np.log(EJn) - np.log(EJn-1)
            s0 = np.arange(t0, smax, ds)

            p1 = EJn*(1-np.exp(-s0)) - 1
            p2 = EJn*np.exp(-s0) + (EJn-1)*(s0-1-t0)
                
            p = np.zeros((len(s)))
            p[s > t0-ds] = p1 * np.exp(-p2)

            pw = np.fft.fft(p) # fourier transform of isi density
            w = np.fft.fftfreq(len(s))*2*np.pi/ds
            pw *= ds*np.exp(-complex(0,1)*w*smin) # phase factor

            Cw = r_th_i * (1 - np.abs(pw)**2) / (np.abs(1 - pw)**2)
            var_spk.append(Cw[1]) # 0 is the delta peak

        else:
            var_spk.append(0)

    r_th = np.array(r_th)
    var_spk = np.array(var_spk)

    vbar = (J*(1-grange_th) + np.sqrt((J*(1-grange_th))**2 + 4*(E-J*(1-grange_th)))) / 2
    f_mft = vbar - 1
    f_mft[vbar < 1] = 0
    f_mft[np.isnan(vbar)] = 0
    var_spk_tree = f_mft / 4

    f_low = 0*grange_th
    var_spk_low = 0*grange_th
    # f_low[vbar > 1] = np.nan

    vbar = (J*(1-grange_th))/2 + 2/3*np.sqrt((3/4*J*(1-grange_th))**2 - 9/4*J*(1-grange_th) + 3*E - 3/4)
    f_1loop = 3/4*(vbar - 1)
    f_1loop[vbar < 1] = 0
    f_1loop[np.isnan(vbar)] = 0
    var_spk_1loop = f_1loop / 4
    # Cw_tree = f * (vbar**2 + w**2) / (4*vbar**2 + w**2)

    # var_spk_tree = (np.sqrt(J*(1-grange_th)*r_th + E)-1) / 4    

    ax[1, 0].plot(grange, r_sim, 'ko', alpha=0.5)
    ax[1, 0].plot(grange, r_sim_stim, 'ko', alpha=0.5)
    ax[1, 0].plot(grange_th, f_mft, 'k:', label='mean field')
    ax[1, 0].plot(grange_th, f_1loop, 'k--', label='1 loop')
    ax[1, 0].plot(grange_th, r_th, 'k', label='exact')
    ax[1, 0].plot(grange_th, f_low, 'k')

    ax[2, 0].plot(grange, var_sim, 'ko', alpha=0.5)
    ax[2, 0].plot(grange, var_sim_stim, 'ko', alpha=0.5)
    ax[2, 0].plot(grange_th, var_spk_tree, 'k:', linewidth=2, label='mean field')
    ax[2, 0].plot(grange_th, var_spk_1loop, 'k--', linewidth=2, label='1 loop')
    ax[2, 0].plot(grange_th, var_spk, 'k', linewidth=2, label='exact')
    ax[2, 0].plot(grange_th, var_spk_low, 'k', linewidth=2)

    ### spike train variance as function of E
    g = 0.25
    Erange = np.linspace(-1, Emax, 6)

    r_sim = []
    r_sim_stim = []
    var_sim = []
    var_sim_stim = []

    Jmat = np.zeros((N, N))
    Jmat[:, :Ne] = np.random.binomial(n=1, p=pE, size=(N,Ne)) * J / pE / Ne
    Jmat[:, Ne:] = np.random.binomial(n=1, p=pI, size=(N,Ni)) * -g * J / pI / Ni

    for ei, E in enumerate(Erange):
        
        print('sims, E, {}/{}'.format(ei+1,len(Erange)))

        _, spktimes = sim_lif_pop(J=Jmat, E=E, tstop=tstop, dt=dt, tstim=0, Estim=0)

        if len(spktimes) > 0:
            r_sim.append(len(spktimes) / N / np.amax(spktimes[:, 0]))
            _, Csim = calc_avg_spectrum(spktimes, tstop=np.amax(spktimes[:, 0])+dt, dt=dt, N=N)
            var_sim.append(Csim[1])
        else:
            r_sim.append(0)
            var_sim.append(0)
        
        if E < 1:
            _, spktimes = sim_lif_pop(J=Jmat, E=E, tstop=tstop, dt=dt, tstim=tstim, Estim=Estim)
            spktimes = spktimes[spktimes[:, 0] > 2*tstim, :]
            spktimes[:, 0] -= 2*tstim

            if len(spktimes) > 0:
                r_sim_stim.append(len(spktimes) / N / (tstop - 2*tstim))
                _, Csim = calc_avg_spectrum(spktimes, tstop=tstop-2*tstim, dt=dt, N=N)
                var_sim_stim.append(Csim[1])
            else:
                r_sim_stim.append(np.nan)
                var_sim_stim.append(np.nan)
        else:
            r_sim_stim.append(np.nan) # already have activity, don't need stim to kick out of low state
            var_sim_stim.append(np.nan)

        fig2, ax2 = plt.subplots(1, 1, figsize=(4, 4))
        ax2.plot(spktimes[:, 0], spktimes[:, 1], 'k|', markersize=0.5)
        fig2.savefig(os.path.join(results_dir, 'raster_J={}_g={}_E={}.pdf'.format(J, np.round(g, decimals=1), np.round(E, decimals=1))))


    Erange_th = np.arange(-1, Emax, .005)

    r_th = []
    var_spk = []

    for ei, E in enumerate(Erange_th):

        r_th_i = lif_rate_homog(J*(1-g), E)
        if np.isnan(r_th_i):
            r_th_i = 0
        r_th.append(r_th_i)

        if r_th_i > 0:
            
            EJn = E + J*(1-g)*r_th_i
            t0 = np.log(EJn) - np.log(EJn-1)
            s0 = np.arange(t0, smax, ds)

            p1 = EJn*(1-np.exp(-s0)) - 1
            p2 = EJn*np.exp(-s0) + (EJn-1)*(s0-1-t0)
                
            p = np.zeros((len(s)))
            p[s > t0-ds] = p1 * np.exp(-p2)

            pw = np.fft.fft(p) # fourier transform of isi density
            w = np.fft.fftfreq(len(s))*2*np.pi/ds
            pw *= ds*np.exp(-complex(0,1)*w*smin) # phase factor

            Cw = r_th_i * (1 - np.abs(pw)**2) / (np.abs(1 - pw)**2)
            var_spk.append(Cw[1]) # 0 is the delta peak

        else:
            var_spk.append(0)

    r_th = np.array(r_th)
    var_spk = np.array(var_spk)

    dE = Erange_th[1] - Erange_th[0]

    vbar = (J*(1-g) + np.sqrt((J*(1-g))**2 + 4*(Erange_th-J*(1-g)))) / 2
    f_mft = vbar - 1
    f_mft[vbar < 1] = 0
    f_mft[np.isnan(vbar)] = 0
    var_spk_tree = f_mft / 4
    f_low = np.nan * Erange_th
    f_low[Erange_th < 1] = 0
    # var_spk_tree = (np.sqrt(J*(1-g)*r_th + Erange_th)-1) / 4   

    vbar = (J*(1-g))/2 + 2/3*np.sqrt((3/4*J*(1-g))**2 - 9/4*J*(1-g) + 3*Erange_th - 3/4)
    f_1loop = 3/4*(vbar - 1)
    f_1loop[vbar < 1] = 0
    f_1loop[np.isnan(vbar)] = 0
    var_spk_1loop = f_1loop / 4

    var_spk[Erange_th <= dE] = 0
    # var_spk_tree[Erange_th < dE] = 0 
    var_spk_low = np.nan * Erange_th
    var_spk_low[Erange_th < 1] = 0

    ax[1, 1].plot(Erange, r_sim, 'ko', alpha=0.5)
    ax[1, 1].plot(Erange, r_sim_stim, 'ko', alpha=0.5)
    ax[1, 1].plot(Erange_th, f_mft, 'k:')
    ax[1, 1].plot(Erange_th, f_1loop, 'k--')
    ax[1, 1].plot(Erange_th, f_low, 'k')
    ax[1, 1].plot(Erange_th, r_th, 'k')

    ax[2, 1].plot(Erange, var_sim, 'ko', alpha=0.5)
    ax[2, 1].plot(Erange, var_sim_stim, 'ko', alpha=0.5)
    ax[2, 1].plot(Erange_th, var_spk_tree, 'k:', linewidth=2, label='mean field')
    ax[2, 1].plot(Erange_th, var_spk_1loop, 'k--', linewidth=2, label='1 loop')
    ax[2, 1].plot(Erange_th, var_spk, 'k', linewidth=2, label='exact')
    ax[2, 1].plot(Erange_th, var_spk_low, 'k', linewidth=2)

    ax[0, 0].set_xlim((0, 3))
    ax[0, 1].set_xlim((0, 40))
    # ax[0, 1].set_xscale('log')
    ax[0, 1].set_ylim((0, 2.5))

    ax[1, 0].set_xlim((gmin, gmax))
    ax[2, 0].set_xlim((gmin, gmax))

    ax[1, 1].set_xlim((-1, Emax))
    ax[2, 1].set_xlim((-1, Emax))

    ax[0, 0].set_xlabel('Interspike interval {}'.format(r'$s$'), fontsize=fontsize)
    ax[0, 0].set_ylabel('Density {}'.format(r'$p(s)$'), fontsize=fontsize)

    ax[0, 1].set_xlabel('Frequency, {} Hz'.format(r'$\tau$'), fontsize=fontsize)
    ax[0, 1].set_ylabel('Spectral density, {} Hz'.format(r'$\tau$'), fontsize=fontsize)
    ax[1, 0].set_ylabel('Rate (norm.)', fontsize=fontsize)

    ax[2, 0].set_xlabel('g', fontsize=fontsize)
    ax[2, 0].set_ylabel('Pop. variance', fontsize=fontsize)
    ax[2, 1].set_xlabel('E', fontsize=fontsize)

    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(savefile)


def plot_propagators_resum(savefile=os.path.join(results_dir, 'fig_propagators.pdf'), Nvec=50, Emin=-1, Emax=2, Jmin=0, Jmax=10):
    
    '''
    Plot figure 8.
    '''

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(3.4, 5))

    J = 6
    E = 0.5
    g = 0
    
    w, Dnn_full, Dvn_full, Dnv_full, Dvv_full = fixed_pt_iter_propagators_1pop_true(J=J, E=E, max_its=10)

    ax[0, 0].plot(w, np.abs(Dnn_full), linewidth=2, color=colors[0], label=r'$\Delta_{\dot{n}, \tilde{n}}$')
    ax[0, 0].plot(w, np.abs(Dvn_full), linewidth=2, color=colors[1], label=r'$\Delta_{v, \tilde{n}}$')
    ax[0, 0].plot(w, np.abs(Dnv_full), '--', linewidth=2, color=colors[2], label=r'$\Delta_{\dot{n}, \tilde{v}}$')
    ax[0, 0].plot(w, np.abs(Dvv_full), linewidth=2, color=colors[3], label=r'$\Delta_{v, \tilde{v}}$')   

    r_th = lif_rate_homog(J*(1-g), E, n_max=10)
    if not (r_th > 0): raise Exception('Need non-zero rate')

    vbar = np.sqrt(J*r_th + E)
    if vbar > 1:
        hazardbar = vbar - 1
        hazard_pr = 1
    else:
        hazardbar = 0
        hazard_pr = 0

    ### define the bare propagators
    Dnn = (1 + hazardbar +1j * w) / (1 + hazardbar + hazard_pr * vbar + 1j*w)
    Dvn = -hazard_pr / (1 + hazardbar + hazard_pr * vbar + 1j*w)
    Dnv = -vbar / (1 + hazardbar + hazard_pr * vbar + 1j*w)
    Dvv = -1 / (1 + hazardbar + hazard_pr * vbar + 1j*w)

    scale = 1000
    ax[0, 1].plot(w, scale*(np.abs(Dnn_full) - np.abs(Dnn)), linewidth=2, color=colors[0])
    ax[0, 1].plot(w, scale*(np.abs(Dvn_full) - np.abs(Dvn)), linewidth=2, color=colors[1])
    ax[0, 1].plot(w, scale*(np.abs(Dnv_full) - np.abs(Dnv)), linewidth=2, color=colors[2])
    ax[0, 1].plot(w, scale*(np.abs(Dvv_full) - np.abs(Dvv)), linewidth=2, color=colors[3])   

    ax[0, 0].set_xlim((0, 20))
    ax[0, 1].set_xlim((0, 20))
    ax[0, 0].legend(loc=0, frameon=False, fontsize=fontsize)
    ax[0, 0].set_xlabel('Frequency, {} Hz'.format(r'$\tau$'), fontsize=fontsize)
    ax[0, 1].set_xlabel('Frequency, {} Hz'.format(r'$\tau$'), fontsize=fontsize)
    ax[0, 0].set_ylabel('Response magnitude, '+r'$\vert \Delta \vert$', fontsize=fontsize)
    ax[0, 1].set_ylabel(r'$\vert \Delta \vert -  \vert \bar{\Delta} \vert$', fontsize=fontsize)

    Evec = np.linspace(Emin, Emax, Nvec)
    Jvec = np.linspace(Jmin, Jmax, Nvec)

    Dnn_diff = np.full((Nvec, Nvec), np.nan)
    Dnv_diff = np.full((Nvec, Nvec), np.nan)
    Dvn_diff = np.full((Nvec, Nvec), np.nan)
    Dvv_diff = np.full((Nvec, Nvec), np.nan)

    for i, E in enumerate(Evec):
        for j, J in enumerate(Jvec):

            try:
                r_th, w, Dnn_full, Dvn_full, Dnv_full, Dvv_full = fixed_pt_iter_propagators_1pop_true(J=J, E=E, return_rate=True)
            except:
                continue

            vbar = np.sqrt(J*r_th + E)
            if vbar > 1:
                hazardbar = vbar - 1
                hazard_pr = 1
            else:
                hazardbar = 0
                hazard_pr = 0

            ### define the bare propagators
            Dnn = (1 + hazardbar +1j * w) / (1 + hazardbar + hazard_pr * vbar + 1j*w)
            Dvn = -hazard_pr / (1 + hazardbar + hazard_pr * vbar + 1j*w)
            Dnv = -vbar / (1 + hazardbar + hazard_pr * vbar + 1j*w)
            Dvv = -1 / (1 + hazardbar + hazard_pr * vbar + 1j*w)

            w0ind = np.argmin(np.abs(w))
            Dnn_diff[i, j] = -np.abs(Dnn[w0ind]) + np.abs(Dnn_full[w0ind])
            Dnv_diff[i, j] = -np.abs(Dnv[w0ind]) + np.abs(Dnv_full[w0ind])
            Dvn_diff[i, j] = -np.abs(Dvn[w0ind]) + np.abs(Dvn_full[w0ind])
            Dvv_diff[i, j] = -np.abs(Dvv[w0ind]) + np.abs(Dvv_full[w0ind])

    # cmax = max([np.amax(np.abs(Dnn_diff)), np.amax(np.abs(Dnv_diff)), np.amax(np.abs(Dvn_diff)), np.amax(np.abs(Dvv_diff))])
    cmax = 0.01

    dE = Evec[1] - Evec[0]
    dJ = Jvec[1] - Jvec[0]

    im = ax[1, 0].imshow(np.abs(Dnn_diff), clim=(0, cmax), extent=(Jmin-dJ/2, Jmax-dJ/2, Emin-dE/2, Emax-dE/2), cmap='cividis', origin='lower', aspect='auto')
    ax[1, 1].imshow(np.abs(Dvn_diff), clim=(0, cmax), extent=(Jmin-dJ/2, Jmax-dJ/2, Emin-dE/2, Emax-dE/2), cmap='cividis', origin='lower', aspect='auto')
    ax[2, 0].imshow(np.abs(Dnv_diff), clim=(0, cmax), extent=(Jmin-dJ/2, Jmax-dJ/2, Emin-dE/2, Emax-dE/2), cmap='cividis', origin='lower', aspect='auto')
    ax[2, 1].imshow(np.abs(Dvv_diff), clim=(0, cmax), extent=(Jmin-dJ/2, Jmax-dJ/2, Emin-dE/2, Emax-dE/2), cmap='cividis', origin='lower', aspect='auto')

    fig.colorbar(im, ax=ax[1:, 1], shrink=0.6)

    ax[1, 0].set_title('Spike-spike, '+r'$\Delta_{\dot{n}, \tilde{n}}$', fontsize=fontsize)
    ax[1, 1].set_title('Voltage-spike, '+r'$\Delta_{v, \tilde{n}}$', fontsize=fontsize)
    ax[2, 0].set_title('Spike-voltage, '+r'$\Delta_{\dot{n}, \tilde{v}}$', fontsize=fontsize)
    ax[2, 1].set_title('Voltage-voltage, '+r'$\Delta_{v, \tilde{v}}$', fontsize=fontsize)

    ax[2, 0].set_xlabel('J', fontsize=fontsize)
    ax[2, 1].set_xlabel('J', fontsize=fontsize)
    ax[1, 0].set_ylabel('E', fontsize=fontsize)
    ax[2, 0].set_ylabel('E', fontsize=fontsize)

    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(savefile)


def plot_fig_propagators_old(savefile='../results/fig_propagators1.pdf'):

    '''
    This is an old version of "plot_propagators_resum."
    It plots the four propagators for a network with threshold-linear hazard functions.
    It plots the bare and dressed (re-summed) propagators.
    '''

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
            
        hazardbar = hazard(vbar)
        if vbar >= 1:
            hazard_pr = 1
        else:
            hazard_pr = 0

        ### bare propagators

        Dnn = (1 + hazardbar +1j * w) / (1 + hazardbar + hazard_pr * vbar + 1j*w)
        Dvn = -hazard_pr / (1 + hazardbar + hazard_pr * vbar + 1j*w)
        Dnv = -vbar / (1 + hazardbar + hazard_pr * vbar + 1j*w)
        Dvv = -1 / (1 + hazardbar + hazard_pr * vbar + 1j*w)

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

        two_pt_a = hazardbar * Dnn * Dnn.conj() # diagram with the mft source
        two_pt_a_full = hazardbar * Dnn_full * Dnn_full.conj() # diagram with the mft source
        ax[0, 2].plot(w, np.abs(two_pt_a_full), '--', color=colors[i], linewidth=2)

        ### diagram with the correction source
        r_th = lif_rate_homog(J, E)

        _, Dnn_full, Dvn_full, Dnv_full, Dvv_full = fixed_pt_iter_propagators_1pop_true(J=J, E=E, w=w, max_its=100)
        two_pt_a_full = r_th * Dnn_full * Dnn_full.conj() # diagram with the mft source

        ax[0, 2].plot(w, np.abs(two_pt_a_full), '-', color=colors[i], linewidth=2)


        
        # dn = r_th - hazardbar
        
        # Dnv0 = -vbar / (1 + hazardbar + hazard_pr * vbar)
        # two_pt_b = -J * hazard_pr * dn * Dnn * Dnn.conj() * Dnv0

        # Dnv0_full = Dnv_full[np.argmin(np.abs(w))]
        # two_pt_b_full = -J * hazard_pr * dn * Dnn_full * Dnn_full.conj() * Dnv0_full

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

    plot_ei_fluctuations()
    
    # plot_propagators_resum()