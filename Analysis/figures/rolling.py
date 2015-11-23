# coding=utf-8
from __future__ import print_function
import numpy as np
import sys
import matplotlib, matplotlib.pyplot as plt
import time
import matplotlib.gridspec as gridspec
import matplotlib.patches
import scipy.stats
from matplotlib.path import Path
from matplotlib import rc
from pylab import *

fig = plt.figure(figsize=(15,6), dpi=300)
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['axes.labelsize'] = 16
rc('text', usetex=False)
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14

gs = gridspec.GridSpec(1, 10)
gs.update(wspace=0.0, hspace=0.125) 
ax0 = fig.add_subplot(gs[0:3])
ax1 = fig.add_subplot(gs[4:])

x = np.arange(-3, 4)
y = np.power(0.9, np.abs(x))
y /= y.sum()
y2 = np.ones(6)/6

ax0.plot(x,y, 'o-')
ax0.set_xlim(-4,4)
ax0.set_xlabel("Pixel offset")
ax0.set_ylabel("Filter strength")

xs = np.arange(0, 100)
ys = 100 * 0.9**np.abs(xs-50)
ys2 = ys + np.random.normal(size=xs.size)*np.sqrt(ys)
out = np.convolve(ys2, y, mode='same')
out2 = np.convolve(ys2, y2, mode='same')

ax1.plot(xs, ys,'k--',  linewidth=1, alpha=0.6, label="Signal")
ax1.plot(xs,ys2, 'k', linewidth=1,  alpha=0.9, label="Sn+Noise")
ax1.plot(xs, out, 'b-',  linewidth=1, label="Output")
ax1.plot(xs, out2, 'r-', linewidth=1, label="Boxcar")

ax1.set_xlabel("Pixel")
ax1.set_ylabel("Signal")
ax1.legend(loc=2)
ax1.set_ylim(-1, max(ys2.max(), ys.max())*1.1)
fig.savefig("rolling.png", bbox_inches='tight', dpi=100, transparent=True)
fig.savefig("rolling.pdf", bbox_inches='tight', transparent=True)
