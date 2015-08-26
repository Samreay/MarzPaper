# coding=utf-8
import numpy as np
import matplotlib, matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
from pylab import *

vals = np.loadtxt("vars.dat")
#vals = np.loadtxt("vars_2.dat")
lambdas = vals[0,:]
initial = vals[1,:]
afterPoly = vals[2,:]
afterSmooth = vals[3,:]
varianceInitial = vals[4,:]
varianceAfter = vals[5,:]
afterError = vals[6,:]

poly = (initial -afterPoly) / (initial.max() - initial.min())

 
initial /= (initial.max() - initial.min())
afterPoly /= (afterPoly.max() - afterPoly.min())
afterSmooth /= (afterSmooth.max() - afterSmooth.min())
afterError /= (afterError.max() - afterError.min())


z = 2.4793
lines = np.array([1215.670, 1240.14, 1400.0, 1549.06,1908.73]) * (1 + z)
lineLabels = [r'$\mathrm{Ly}_\alpha$', r'$\left[\mathrm{N}_\mathrm{V}\right]$', r'$\mathrm{Si}_4$', r'$\mathrm{C}_{\mathrm{IV}}$', r'$\mathrm{C}_{\mathrm{III}}$']






fig = plt.figure(figsize=(15,6), dpi=300)
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['axes.labelsize'] = 16
rc('text', usetex=False)
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14

gs = gridspec.GridSpec(4, 1)
gs.update(wspace=0.025, hspace=0.0) 
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1], sharex=ax0)
ax2 = fig.add_subplot(gs[2], sharex=ax0)
ax3 = fig.add_subplot(gs[3], sharex=ax0)



ax0.fill_between(lambdas, initial - varianceInitial, initial + varianceInitial, facecolor='red', edgecolor='none', alpha=0.3)
ax1.fill_between(lambdas, afterPoly - varianceInitial, afterPoly + varianceInitial, facecolor='red', edgecolor='none', alpha=0.3)
ax2.fill_between(lambdas, afterSmooth - varianceAfter, afterSmooth + varianceAfter, facecolor='red', edgecolor='none', alpha=0.3)

'''
ax0.fill_between(lambdas, initial.min(),initial.min()+ varianceInitial, facecolor='red', edgecolor='none', alpha=0.3)
ax1.fill_between(lambdas, afterPoly.min(), afterPoly.min()+varianceInitial, facecolor='red', edgecolor='none', alpha=0.3)
ax2.fill_between(lambdas, afterSmooth.min(), afterSmooth.min()+varianceAfter, facecolor='red', edgecolor='none', alpha=0.3)
'''
ax0.plot(lambdas, initial, color="#444444")
ax1.plot(lambdas, afterPoly, color="#444444")
ax2.plot(lambdas, afterSmooth, color="#444444")
ax3.plot(lambdas, afterError, color="#444444")

for line, label in zip(lines, lineLabels):
    ax0.plot([line, line], [-1000, 1000], ':', color='#888888')
    if label == r'$\mathrm{Ly}_\alpha$':
        ax0.text(0.95* line, max(initial) * 1, label, fontsize=16)
    else:
        ax0.text(line*1.005, max(initial) * 1, label, fontsize=16)
    ax1.plot([line, line], [-1000, 1000], ':', color='#888888')
    ax2.plot([line, line], [-1000, 1000], ':', color='#888888')
    ax3.plot([line, line], [-1000, 1000], ':', color='#888888')

ax0.get_yaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)

ax0.get_xaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)

xmin=lambdas.min()
xmax=lambdas.max()
s = 1.25
ax0.set_ylim(initial.min(), initial.max()*s)
ax1.set_ylim(afterPoly.min(), afterPoly.max()*s)
ax2.set_ylim(afterSmooth.min(), afterSmooth.max()*s)
ax3.set_ylim(afterError.min(), afterError.max()*s)
ax0.set_xlim(xmin, xmax)

ax0.text(0.99 * xmax, -0.1 + max(initial), 'Starting spectrum', fontsize=14, horizontalalignment='right')
ax1.text(0.99 * xmax, -0.1 + max(afterPoly), 'Aftter polynomial subtraction', fontsize=14, horizontalalignment='right')
ax2.text(0.99 * xmax, -0.1 + max(afterSmooth), 'After tapering and smoothing of spectrum and variance', fontsize=14, horizontalalignment='right')
ax3.text(0.99 * xmax, -0.1 + max(afterError), 'After division by variance', fontsize=14, horizontalalignment='right')
ax3.set_xlabel("Wavelength (Å)".decode('utf-8'))
figtext(0.1,0.65,r"Spectrum intensity", fontdict={'fontsize':15},rotation=90)

fig.savefig("quasarProcess.png", bbox_inches='tight', dpi=600, transparent=True)
fig.savefig("quasarProcess.pdf", bbox_inches='tight', transparent=True)