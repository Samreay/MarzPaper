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

def plotSubtractions():
    data = np.loadtxt("continuumSub")
    lambdas = data[0,:]
    start = data[1, :]
    poly = data[2, :]
    medians = data[3, :]
    smoothed = data[4, :]
    subtracted = data[5, :]
    
    z = 0.194
    lines = np.array([3933.663, 3968.468, 5175.3, 5894.0]) * (1 + z)
    lineLabels = [r'$\mathrm{H}$', r'$\mathrm{K}$', r'$\mathrm{Mg}$', r'$\mathrm{Na}$']
    	
    fig = plt.figure(figsize=(15,6), dpi=300)
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams['axes.labelsize'] = 16
    rc('text', usetex=False)
    matplotlib.rcParams['xtick.labelsize'] = 14
    matplotlib.rcParams['ytick.labelsize'] = 14
    
    gs = gridspec.GridSpec(3, 1)
    gs.update(wspace=0.025, hspace=0.0) 
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    
    ax0.plot(lambdas, start, color="#555555")
    ax0.plot(lambdas, start-poly, '--', color="#1E88E5", linewidth=2)
    for line, label in zip(lines, lineLabels):
        ax0.plot([line, line], [-1000, 1000], ':', color='#888888')
        if label == r'$\mathrm{H}$':
            ax0.text(0.98* line, max(start) * 1, label, fontsize=16)
        else:
            ax0.text(line*1.005, max(start) * 1, label, fontsize=16)
        ax1.plot([line, line], [-1000, 1000], ':', color='#888888')
        ax2.plot([line, line], [-1000, 1000], ':', color='#888888')
        
    ax1.plot(lambdas, poly, color="#555555")
    #ax1.plot(lambdas, medians, color='#1A237E')
    ax1.plot(lambdas, smoothed, '--', color="#F44336", linewidth=2)
    ax2.plot(lambdas, subtracted, color="#444444")
    
    ax0.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
    
    #ax0.yaxis.get_major_ticks()[0].label1.set_visible(False)
    #ax1.yaxis.get_major_ticks()[0].label1.set_visible(False)
    #ax2.yaxis.get_major_ticks()[0].label1.set_visible(False)
    ax0.get_yaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    
    xmin = min(lambdas)
    xmin = 4000
    xmax = max(lambdas)
    ax0.set_xlim(xmin, xmax)
    ax0.set_ylim(min(start)*1.1, 1.2*max(start))
    ax1.set_ylim(min(poly)*1.1, 1.1*max(poly))
    ax2.set_ylim(min(subtracted)*1.1, 1.1*max(subtracted))
    ax1.set_xlim(xmin, xmax)
    ax2.set_xlim(xmin, xmax)
    
    ax0.get_xaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    
    #ax0.text(0.99 * xmax, 0.9*max(start), '(a)', fontsize=14, horizontalalignment='right')
    #ax1.text(0.99 * xmax, 0.8*max(poly), '(b)', fontsize=14, horizontalalignment='right')
    #ax2.text(0.99 * xmax, 0.7*max(subtracted), '(c)', fontsize=14, horizontalalignment='right')
    ax0.text(0.99 * xmax, min(start), 'Starting spectrum and polynomial fit (dashed)', fontsize=14, horizontalalignment='right')
    ax1.text(0.99 * xmax, min(poly), 'Polynomial subtraction and smoothed median filter (dashed)', fontsize=14, horizontalalignment='right')
    ax2.text(0.99 * xmax, min(subtracted), 'Final spectrum after smoothed median subtraction', fontsize=14, horizontalalignment='right')
    
    
    ax2.set_xlabel("Wavelength (Å)".decode('utf-8'))
    figtext(0.1,0.65,r"Spectrum intensity", fontdict={'fontsize':15},rotation=90)
        
    fig.savefig("continuum.png", bbox_inches='tight', dpi=600, transparent=True)
    fig.savefig("continuum.pdf", bbox_inches='tight', transparent=True)

plotSubtractions()