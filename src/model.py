import numpy as np

def hazard(v, B=1, v_th=1, p=1):
    x = v - v_th 

    if len(np.shape(x)) > 0:
        x[x < 0] = 0
    elif x < 0:
        x = 0
    else: pass
    
    return B * x**p


def hazard_match_linear_reset_mft(v, B=1, v_th=1, p=1):

    x = v - v_th 

    if len(np.shape(x)) > 0:
        x[x < 0] = 0
    elif x < 0:
        x = 0
    else: pass
    
    return B * np.sqrt(v) * x**p

if __name__ == '__main__':
    pass