import numpy as np
from src.model import intensity
import matplotlib.pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = ['k']+colors

def dv(v, E, J, g, p=1):
    
    if len(E) == 1:
        Ee = E
        Ei = E
    elif len(E) == 2:
        Ee, Ei = E
    else:
        raise Exception('input E has length {}, needs to be 1 or 2'.format(len(E)))
    
    ve, vi = v
    
    dve = -ve - ve*intensity(ve) + J*intensity(ve, p=p) - g*J*intensity(vi, p=p) + Ee
    dvi = -vi - vi*intensity(vi) + J*intensity(ve, p=p) - g*J*intensity(vi, p=p) + Ei
    
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
    
    dve = -ve + J*intensity(ve, p=p) - g*J*intensity(vi, p=p) + Ee
    dvi = -vi + J*intensity(ve, p=p) - g*J*intensity(vi, p=p) + Ei
    
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
