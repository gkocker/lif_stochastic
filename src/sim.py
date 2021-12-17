import numpy as np
from src.model import phi

def sim_pif(J, E, tstop=100, dt=.01, B=1, v_th=1, p=1):

    Nt = int(tstop / dt)
    
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

        v[t] = v[t-1] + dt*E + np.dot(J, n) - n*v[t-1]

        lam = phi(v[t], B=B, v_th=v_th, p=p)
        if lam > 1/dt:
            lam = 1/dt
            
        n = np.random.binomial(n=1, p=dt*lam)

        spkind = np.where(n > 0)[0]
        for i in spkind:
            spktimes.append([t, i])
            
    return v, spktimes


def sim_lif_pop(J, E, tstop=100, dt=.01, B=1, v_th=1, p=1, v_r=0, tstim=0, Estim=0, v0=0):

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
    v[0] = v0
    
    n = np.zeros(N,)
    spktimes = []

    for t in range(1, Nt):

        if t < Ntstim:
            Et = Estim
        else:
            Et = E

        v[t] = v[t-1] + dt*(-v[t-1] + Et) + np.dot(J, n) - n*(v[t-1]-v_r)

        lam = phi(v[t], B=B, v_th=v_th, p=p)
        lam[lam > 1/dt] = 1/dt
            
        n = np.random.binomial(n=1, p=dt*lam)

        spkind = np.where(n > 0)[0]
        for i in spkind:
            spktimes.append([t*dt, i])
            
    spktimes = np.array(spktimes)
    return v, spktimes


def sim_lif_pop_fully_connected(J, E, N=1000, tstop=100, dt=.01, B=1, v_th=1, p=1, v_r=0, tstim=0, Estim=0):

    Nt = int(tstop / dt)
    Ntstim = int(tstim / dt)
    
    if len(np.shape(E)) == 0:
        E = E * np.ones(N,)
    elif len(E) < N:
        raise Exception('Need either a scalar or length N input E')

    v = np.zeros((Nt,N))
    v[0] = np.random.randn(N) / np.sqrt(N)
    n = np.zeros(N,)
    spktimes = []

    for t in range(1, Nt):

        if t < Ntstim:
            Et = Estim
        else:
            Et = E

        v[t] = v[t-1] + dt*(-v[t-1] + Et) + J*np.sum(n) - n*(v[t-1]-v_r)

        lam = phi(v[t], B=B, v_th=v_th, p=p)
        # lam[lam > 1/dt] = 1/dt

        # n = np.random.binomial(n=1, p=dt*lam)
        n = np.random.poisson(lam=dt*lam)

        spkind = np.where(n > 0)[0]
        for i in spkind:
            spktimes.append([t*dt, i])
            
    spktimes = np.array(spktimes)
    return v, spktimes


def sim_lif_perturbation(J, E, tstop=100, dt=.01, B=1, v_th=1, p=1, v_r=0, perturb_start=None, perturb_len=10, perturb_amp=1.5, perturb_ind=None):

    '''
    simulate an LIF network. at times tstop/4 a postive perturbation is applied, and at 3/4 tstop a negative perturbation

    J: connectivity matrix, NxN
    E: resting potential
    '''

    Nt = int(tstop / dt)

    if len(np.shape(J)) > 1:
        N = np.shape(J)[0]
    else:
        N = 1
    
    if len(np.shape(E)) == 0:
        E0 = E * np.ones(N,)
    elif len(E) == N:
        E0 = np.array(E)
    else:
        raise Exception('Need either a scalar or length N input E')

    print(E0.shape)

    if perturb_ind is None:
        perturb_ind = range(N)

    if perturb_start is None:
        t_start_perturb1 = Nt//4
        t_end_perturb1 = t_start_perturb1 + int(perturb_len / dt)
        
        t_start_perturb2 = 3*Nt//4
        t_end_perturb2 = t_start_perturb2 + int(perturb_len / dt)
    
    else:
        t_start_perturb1 = int(perturb_start / dt)
        t_end_perturb1 = t_start_perturb1 + int(perturb_len / dt)

        t_start_perturb2 = Nt+1
        t_end_perturb2 = Nt+1

    v = np.zeros((Nt,N))
    v[0] = np.random.rand(N,)
    n = np.zeros(N,)
    spktimes = []

    for t in range(1, Nt):
        
        E = E0.copy()

        if (t >= t_start_perturb1) and (t < t_end_perturb1):
            E[perturb_ind] += perturb_amp
        elif (t >= t_start_perturb2) and (t < t_end_perturb2):
            E[perturb_ind] -= perturb_amp

        v[t] = v[t-1] + dt*(-v[t-1] + E) - n*(v[t-1]-v_r) + np.dot(J, n)

        lam = phi(v[t], B=B, v_th=v_th, p=p)
        lam[lam > 1/dt] = 1/dt
            
        n = np.random.binomial(n=1, p=dt*lam)

        spkind = np.where(n > 0)[0]
        for i in spkind:
            spktimes.append([t*dt, i])
            
    spktimes = np.array(spktimes)
    return v, spktimes


def sim_lif_perturbation_x(J, E, tstop=100, dt=.01, B=1, v_th=1, p=1, v_r=0, perturb_amp=1.5, perturb_ind=None):

    '''
    simulate an LIF network. the resting potential E undergoes a series of step perturbations.
    
    J: connectivity matrix, NxN
    E: resting potential
    perturb_amp: list of step amplitudes
    perturb_ind: indices of neurons that experience the perturbation
    '''

    Nt = int(tstop / dt)

    if len(np.shape(J)) > 1:
        N = np.shape(J)[0]
    else:
        N = 1

    if len(np.shape(perturb_amp)) == 0:
        Nperturb = 1
    else:
        Nperturb = len(perturb_amp)

    perturb_len = Nt // (Nperturb + 1)

    if len(np.shape(E)) == 0:
        E0 = E * np.ones(N,)
    elif len(E) == N:
        E0 = np.array(E)
    else:
        raise Exception('Need either a scalar or length N input E')

    if perturb_ind is None:
        perturb_ind = range(N)

    v = np.zeros((Nt,N))
    v[0] = np.random.rand(N,)
    n = np.zeros(N,)
    spktimes = []

    for t in range(1, Nt):
        
        E = E0.copy()
        if t > perturb_len:
            E[perturb_ind] += perturb_amp[(t - perturb_len) // perturb_len]

        v[t] = v[t-1] + dt*(-v[t-1] + E) - n*(v[t-1]-v_r) + np.dot(J, n)

        lam = phi(v[t], B=B, v_th=v_th, p=p)
        lam[lam > 1/dt] = 1/dt
            
        n = np.random.binomial(n=1, p=dt*lam)

        spkind = np.where(n > 0)[0]
        for i in spkind:
            spktimes.append([t*dt, i])
            
    spktimes = np.array(spktimes)
    return v, spktimes


def create_spike_train(spktimes, neuron=0, dt=.01, tstop=100):
    
    '''
    create a spike train from a list of spike times and neuron indices
    spktimes: Nspikes x 2, first column is times and second is neurons
    dt and tstop should match the simulation that created spktimes
    '''
    
    spktimes_tmp = spktimes[spktimes[:, 1] == neuron][:, 0]
    
    Nt = int(tstop/dt)
    spktrain = np.zeros((Nt,))
    
    spk_indices = spktimes_tmp / dt
    spk_indices = spk_indices.astype('int')

    spktrain[spk_indices] = 1/dt
    
    return spktrain


if __name__=='__main__':
    pass