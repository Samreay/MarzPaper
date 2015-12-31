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
from pylab import *

from templateData import *

lambdas = [lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7, lambda8, lambda9, lambda10, lambda11, lambda12]
specs = [spec1, spec2, spec3, spec4, spec5, spec6, spec7, spec8, spec9, spec10, spec11, spec12]
labels = ["A Star", "K Star", "M3 Star", "M5 Star", "G Star", 'Early Type Absorption Galaxy', 'Intermediate Type Galaxy','Late Type Emission Galaxy','Composite Galaxy','High Redshift Star Forming Galaxy','Transitional Galaxy','Quasar']
fig = plt.figure(figsize=(15,19), dpi=300)
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['axes.labelsize'] = 16
rc('text', usetex=False)
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14

gs = gridspec.GridSpec(12, 1)
gs.update(wspace=0.025, hspace=0.25) 


for i, l, s, label in zip(range(len(lambdas)), lambdas, specs, labels):
    ax = fig.add_subplot(gs[i])
    s2 = np.array(s)/max(s)
    ax.plot(l, s2, color="#555555")
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_xlim(min(l), max(l))
    ax.set_ylim(min(s2), 1.4*max(s2))
    ax.text(0.99, 0.92, label, fontsize=14, horizontalalignment='right', transform = ax.transAxes, verticalalignment='top', color="#1E88E5")
    if i == 11:
        ax.set_xlabel("Wavelength (Å)")


figtext(0.07,0.6,"Normalised Intensity", fontdict={'fontsize':16},rotation=90)
#fig.savefig("templates.png", bbox_inches='tight', dpi=300, transparent=True)
fig.savefig("templates.pdf", bbox_inches='tight', transparent=True)