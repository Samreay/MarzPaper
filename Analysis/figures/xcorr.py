# coding=utf-8
from __future__ import print_function
from sam import *
import numpy as np
import sys
from multiprocessing import Pool
import matplotlib, matplotlib.pyplot as plt
import time
import matplotlib.gridspec as gridspec
import matplotlib.patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats
from matplotlib.path import Path
from matplotlib import rc
import matplotlib.patches as mpatches

from pylab import *

data = np.loadtxt("intensity.dat")
lambdas = data[0, :]
intensity = data[1, :]
intensity /= intensity.max()

data = np.loadtxt('xcors.dat')
zs = data[0, :]
xcors = data[1, :]

data = np.loadtxt('templateSpec.dat')
templateLambdas = data[0, :]
templateSpec = data[1, :]
templateSpec /= templateSpec.max()

fig = plt.figure(figsize=(15,8), dpi=300)
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['axes.labelsize'] = 16
rc('text', usetex=False)
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14

gs = gridspec.GridSpec(105, 1)
gs.update(wspace=0.125, hspace=0) 
ax0 = fig.add_subplot(gs[0:35])
ax1 = fig.add_subplot(gs[45:65])
ax2 = fig.add_subplot(gs[65:85], sharex=ax1)
ax3 = fig.add_subplot(gs[85:105], sharex=ax1)


ax0.plot([0.3107, 0.3107], [-10, 30],'--',  color='#1976D2', linewidth=2)
ax0.plot([0.7592, 0.7592], [-10, 30], '--', color='#D32F2F', linewidth=2)
ax0.plot(zs, xcors, label="Xcor function", color="#111111")
ax0.text(0.325, 13, "Primary Peak", color="#1976D2")
ax0.text(0.765, 13, "Secondary Peak", color="#D32F2F")

ax1.plot(lambdas, intensity, color='#111111', label="Input Spectrum") #D0322069-3152339


ax2.plot(templateLambdas * (1 + 0.3107), templateSpec, color='#1976D2', label="Primary Peak")

ax3.plot(templateLambdas * (1 + 0.7592), templateSpec, color='#D32F2F', label="Secondary Peak")

ax0.set_xlim(0, 1.5)
ax0.set_ylim(-3, 16)
ax1.set_xlim(4000, 8800)
ax2.set_xlim(4000, 8800)
ax3.set_xlim(4000, 8800)

p1 = mpatches.Patch(color='red', alpha=0.0, label='Xcor function')
p2 = mpatches.Patch(color='red', alpha=0.0, label='Input Spectrum')
p3 = mpatches.Patch(color='red', alpha=0.0, label='Primary Template Match, $z=0.3107$')
p4 = mpatches.Patch(color='red', alpha=0.0, label='Secondary Template Matchh, $z=0.7592$')

l0 = ax0.legend(framealpha=0.0, handles=[p1])

l1 = ax1.legend(framealpha=0.0, handles=[p2])
l2 = ax2.legend(framealpha=0.0, handles=[p3])
l2.get_texts()[0].set_color("#1976D2")

l3 = ax3.legend(framealpha=0.0, handles=[p4])
l3.get_texts()[0].set_color("#D32F2F")

ax0.set_xlabel('Redshift')

ax0.set_ylabel("Xcor strength")
ax3.set_xlabel("Wavelength (Å)".decode('utf-8'))
figtext(0.084,0.47,r"Normalised intensity", fontdict={'fontsize':16},rotation=90)

ax1.get_xaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)

ax1.set_ylim(-0.05, 1)
ax2.set_ylim(-0.05, 1)
ax3.set_ylim(-0.05, 1)

ax0.yaxis.set_major_locator(plt.MaxNLocator(4))
ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
ax3.yaxis.set_major_locator(plt.MaxNLocator(4))

ax3.xaxis.set_major_locator(plt.MaxNLocator(7))
ax0.xaxis.set_major_locator(plt.MaxNLocator(15))

fig.savefig("xcors.png", bbox_inches='tight', dpi=300, transparent=True)
fig.savefig("xcors.pdf", bbox_inches='tight', transparent=True)