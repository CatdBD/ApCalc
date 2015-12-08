import circ_app
import numpy as np

__author__ = 'Caffe'

def run():
    sizex = 20
    sizey = 20
    arr_offsetx = 3.8
    arr_offsety = -4.4
    midx = (sizex / 2.) - 0.5 - arr_offsetx
    midy = (sizey / 2.) - 0.5 - arr_offsety
    scale = 2.0
    xarr, yarr = np.indices([sizex, sizey], float)
    ID = np.reshape(np.linspace(1, sizex * sizey, sizex * sizey), (sizex, sizey))
    xarr -= midx
    yarr -= midy
    xarr /= scale
    yarr /= scale
    arrs = xarr, yarr
    circle_loc = 2.3, -1.1
    rcirc = 3.8

    xx, yy, AA = circ_app.circular_aperture(arrs, circle_loc, rcirc, normalize=True, plot=True)

