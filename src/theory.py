import numpy as np
import mpmath
from scipy.optimize import root_scalar, minimize_scalar, fsolve
from scipy.signal import fftconvolve
from src.model import phi

def dv_mft_pif(v, J, E, B=1, v_th=1, p=1):
    
    phiv = phi(v, B=B, v_th=v_th, p=p)
    return -v * phiv + E + np.dot(J, phiv)


def dv_mft_lif(v, J, E, B=1, v_th=1, p=1):
    
    phiv = phi(v, B=B, v_th=v_th, p=p)
    return -v * (1 + phiv) + E + np.dot(J, phiv)


def lif_mft_rhs(v, E=0, B=1, v_th=1, p=2):
    return -v - v*phi(v, B=B, v_th=v_th, p=p) + E


def lif_linear_1loop_fI(Erange, B=1, v_th=1, p=1):

    '''
    1 loop calculation for B=1, v_th=1, p=1; other parameters need new calc
    '''

    Epos = Erange.copy()
    Epos[Epos < 0] = 0.
    
    NE = len(Erange)
    vbar = np.zeros((NE,))
    phibar = np.zeros((NE,))
    
    for i, e in enumerate(Erange):
        vbar[i] = fsolve(lif_mft_rhs, 1, args=(e,1,v_th,p))
        phibar[i] = phi(vbar[i], p=p, B=B, v_th=v_th)            

    vbar_pos = vbar.copy() - v_th
    vbar_pos[vbar_pos < 0] = 0

    return phibar, phibar + 1/(8 * np.pi) * vbar_pos/vbar


def lif_linear_loop_fI(Erange, B=1, v_th=1, p=1):

    '''
    re-summed loop calculation for B=1, v_th=1, p=1; other parameters need new calc
    '''

    r_1loop = []
    r_mft = []

    for i, e in enumerate(Erange):

        vbar = fsolve(lif_mft_rhs, 1, args=(e,1,v_th,p))
        phibar = phi(vbar, p=p, B=B, v_th=v_th)    

        w, Dnn_full, Dvn_full, Dnv_full, Dvv_full = fixed_pt_iter_propagators_1pop(J=0, E=e, max_its=100)

        if phibar > 0:
            phi_pr = 1
        else:
            phi_pr = 0
        
        dw = w[1] - w[0]
        integral = np.sum( Dvn_full * Dnn_full.conj() ) * dw

        ind0 = np.argmin(np.abs(w))
        r_1loop_tmp = phibar * Dnv_full[ind0] / (2*np.pi)**2 * integral

        r_mft.append(phibar)
        r_1loop.append(phibar + np.real(r_1loop_tmp))

    return r_mft, r_1loop


def rate_fn(n, J, E):

    if n < 0: n = 0 

    EJn = E + J*n - 1

    try:
        gam = np.complex(mpmath.gamma(EJn) - mpmath.gammainc(EJn, EJn))
        rhs = -np.log(1-1/(EJn+1)) + gam * (np.e / EJn)**EJn
        rhs = rhs**-1
    except:
        rhs = np.nan
        
    return np.real(rhs - n)


def rate_fn_neg(n, J, E):

    if n < 0: n = 0 

    EJn = E + J*n - 1

    try:
        gam = np.complex(mpmath.gamma(EJn) - mpmath.gammainc(EJn, EJn))
        rhs = -np.log(1-1/(EJn+1)) + gam * (np.e / EJn)**EJn
        rhs = rhs**-1
    except:
        rhs = np.nan
        
    return np.real(n - rhs)


def lif_linear_full_fI(Erange, J=0, eps=1e-6):
    
    '''
    f-I curve
    exact firing rate with a threshold-linear transfer function (non-dim. for reset at 0 and threshold at 1)
    '''

    r_th = []

    for ei, E in enumerate(Erange):
    
        result = minimize_scalar(rate_fn, args=(J, E))
        if result.success and np.real(result.fun) < eps:
            r_th.append(result.x)
        else:
            r_th.append(0)

    return np.array(r_th)


def fixed_pt_iter_propagators_1pop(J, E, max_its=100, w=np.linspace(-200, 200, 2**15)):

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

    ### define the bare propagators
    Dnn = (1 + phibar +1j * w) / (1 + phibar + phi_pr * vbar + 1j*w)
    Dvn = -phi_pr / (1 + phibar + phi_pr * vbar + 1j*w)
    Dnv = -vbar / (1 + phibar + phi_pr * vbar + 1j*w)
    Dvv = -1 / (1 + phibar + phi_pr * vbar + 1j*w)

    ### solve for the full n, \tilde{n} propagator
    dw = w[1] - w[0]
    pi2 = (2*np.pi)**2

    Dnn_full = Dnn

    for i in range(max_its):
        
        bub = (Dvv / Dnv) * (Dnn_full - Dnn) + Dvn
        bubint = dw * fftconvolve(Dnn_full, bub, mode='same')
        Dnn_full = Dnn + phi_pr / pi2 * Dnv * bub * bubint

    ### solve for the full v, \tilde{n} propagator: it is the bubble diagram
    Dvn_full = bub

    ### solve for the full v, \tilde{v} propagator
    Dvv_full = Dvv
    for i in range(max_its):
        Dvv_full = Dvv + phi_pr / pi2 * Dvv * Dvv_full * bubint

    ### solve for the full n, \tilde{v} propagator
    Dnv_full = Dnv + phi_pr / pi2 * Dnv * Dvv_full * bubint

    return w, Dnn_full, Dvn_full, Dnv_full, Dvv_full


def fixed_pt_iter_propagators_1pop_true(J, E, max_its=10, w=np.linspace(-200, 200, 2**15), return_rate=False, n_max=20):

    ### define the mean-field theory
    r_th = lif_rate_homog(J, E, n_max=n_max)

    vbar = np.sqrt(J*r_th + E)
    if vbar > 1:
        phibar = vbar - 1
        phi_pr = 1
    else:
        phibar = 0
        phi_pr = 0
        raise Exception('need positive rate')

    ### define the bare propagators
    Dnn = (1 + phibar +1j * w) / (1 + phibar + phi_pr * vbar + 1j*w)
    Dvn = -phi_pr / (1 + phibar + phi_pr * vbar + 1j*w)
    Dnv = -vbar / (1 + phibar + phi_pr * vbar + 1j*w)
    Dvv = -1 / (1 + phibar + phi_pr * vbar + 1j*w)

    ### solve for the full n, \tilde{n} propagator
    dw = w[1] - w[0]
    pi2 = (2*np.pi)**2

    Dnn_full = Dnn

    for i in range(max_its):
        
        bub = (Dvv / Dnv) * (Dnn_full - Dnn) + Dvn
        bubint = dw * fftconvolve(Dnn_full, bub, mode='same')
        Dnn_full = Dnn + phi_pr / pi2 * Dnv * bub * bubint

    ### solve for the full v, \tilde{n} propagator: it is the bubble diagram
    Dvn_full = bub

    ### solve for the full v, \tilde{v} propagator
    Dvv_full = Dvv
    for i in range(max_its):
        Dvv_full = Dvv + phi_pr / pi2 * Dvv * Dvv_full * bubint

    ### solve for the full n, \tilde{v} propagator
    Dnv_full = Dnv + phi_pr / pi2 * Dnv * Dvv_full * bubint

    if return_rate:
        return r_th, w, Dnn_full, Dvn_full, Dnv_full, Dvv_full
    else:
        return w, Dnn_full, Dvn_full, Dnv_full, Dvv_full


def rate_1pop_1loop(J, E):

    # 1-loop approximation around the mean-field theory with the bare propagators
    # assume a threshold-linear transfer function with gain 1

    if ((J * (4-J)/4) < E) and (J > 2):
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

    int_factor = -np.pi * (2 + 2*phibar + phi_pr*vbar) / (1 + phibar + phi_pr*vbar)
    pi2 = (2*np.pi)**-2
    pi3 = (2*np.pi)**-3
    Dnv = -vbar / (1+phibar+vbar)
    Dvv = -1 / (1+phibar+vbar)

    r_1loop = phibar * pi2 * Dnv * int_factor / (1 + J*Dnv*(phi_pr*pi3*Dvv*int_factor) )

    return vbar, phibar, r_1loop


def lif_rate_homog(J, E, n_max=20):

    result1 = minimize_scalar(rate_fn_neg, args=(J, E)) # find the local maximum of n(n) - n
    if result1.success and (result1.fun < 0):
        
        n_min = result1.x
        result2 = root_scalar(rate_fn, args=(J, E), method='bisect', bracket=(n_min, n_max))
        
        if result2.converged:
            return result2.root
        else:
            return np.nan

    else:
        return 0


if __name__ == '__main__':
    pass