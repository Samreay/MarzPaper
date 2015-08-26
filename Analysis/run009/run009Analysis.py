import os
import numpy as np
import asciitable

import matplotlib, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
from pylab import *


def loadData():
    directory = os.path.dirname(os.path.realpath(__file__))
    
    marzDir = os.path.join(directory, 'marz')
    autozDir = os.path.join(directory, 'autoz')
    runzDir = os.path.join(directory, 'runz')
    
    
    autozID = 3
    autozNum = 0
    autozZ = 8
    autozTID = 10
    runzID = 1
    runzZ = 19
    runzXZ = 16
    runzEZ = 11
    runzQOP = 22
    marzZ = 8
    marzT = 5
    marzTID = 6
    marzID = 1
    objects = {}
    
    for filename in os.listdir(autozDir):
        with open(os.path.join(autozDir, filename), 'r') as f:
            data = asciitable.read(f.read())
            for i in data:
                key = i[autozID]
                if objects.get(key):
                    objects[key]['ATID'] = i[autozTID]
                    objects[key]['AZ'] = i[autozZ]
                    objects[key]['FILE'] = filename
                    objects[key]['ID'] = i[autozID]
                else:
                    objects[key] = {'AZ': i[autozZ], 'ATID': i[autozTID], 'FILE': filename, 'Num': i[autozNum]}
    for filename in os.listdir(runzDir):
        with open(os.path.join(runzDir, filename), 'r') as f:
            for line in f:
                if line.startswith("#"): continue
                i = line.split()
                key = i[runzID]
                if len(i) < runzQOP: continue
                if objects.get(key):
                    objects[key]['RZ'] = float(i[runzZ])
                    objects[key]['QOP'] = int(i[runzQOP])
                    objects[key]['RXZ'] = float(i[runzXZ])
                    objects[key]['REZ'] = float(i[runzEZ])
                else:
                    objects[key] = {'RZ': float(i[runzZ]), 'QOP': int(i[runzQOP]), 'RXZ': float(i[runzXZ]), 'REZ': float(i[runzEZ])}
                    
    for filename in os.listdir(marzDir):
        if os.path.isfile(os.path.join(marzDir, filename)):
            with open(os.path.join(marzDir, filename), 'r') as f:
                for line in f:
                    if line.startswith("#"): continue
                    i = line.split(",")
                    if len(i) <= marzTID: continue
                    idd = i[marzID].strip()
                    if objects.get(idd):
                        objects[idd]['MZ'] = float(i[marzZ])
                        objects[idd]['MTID'] = int(i[marzTID].strip())
                        objects[idd]['TYPE'] = i[marzT].strip()
                    else:
                        objects[idd] = {'MZ': float(i[marzZ]), 'MTID': i[marzTID].strip(), 'TYPE': i[marzT].strip()}
                        
                        
    results = []
    
    for item in objects:
        i = objects[item]
        if i.get('MZ') is not None and i.get('AZ') is not None and i.get('RZ') is not None and i.get('RXZ') is not None:
            results.append([i.get('QOP'), i.get('RZ'), i.get('MZ'), i.get('AZ'), i.get('RXZ'), i.get('REZ'), i.get('MTID')])
            
    return np.array(results)
    
    
def plotOffset(res):
    qop4 = res[res[:,0] == 4]
    use = qop4[qop4[:,1] < 1]
    #t = 7
    #use = use[use[:, -1] == t]
    
    
    marzDiff = - use[:, 2] + use[:, 1]
    autozDiff = -use[:, 3] + use[:, 1]
    runzDiff = -use[:, 4] + use[:, 1]
    
    bin=numpy.arange(-0.001,0.002,0.00004)
    fig = plt.figure(figsize=(30,7), dpi=300)
    matplotlib.rcParams.update({'font.size': 12})
    #matplotlib.rcParams['axes.labelsize'] = 20
    rc('text', usetex=False)
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    
    ax = fig.add_subplot(1,1,1)
    ax.xaxis.set_ticks(numpy.arange(-0.001,0.002,0.0001))
    ax.hist(marzDiff, bin, color='b', label='marz', alpha=0.5)
    ax.hist(autozDiff, bin, color='g', label='autoz', alpha=0.5)
    ax.hist(runzDiff, bin, color='r', label='runz', alpha=0.2)
    
    
    
    
    
def plotQop4Comparison(res):
    qop4 = res[res[:,0] >= 4]
    
    thresh = 0.01
    
    autozNum = 100.0 *  (np.abs(qop4[:,3] - qop4[:,1]) < thresh).sum() / qop4[:,1].size
    marzNum  = 100.0 *  (np.abs(qop4[:,2] - qop4[:,1]) < thresh).sum() / qop4[:,1].size
    runzXNum = 100.0 *  (np.abs(qop4[:,4] - qop4[:,1]) < thresh).sum() / qop4[:,1].size
    runzENum = 100.0 *  (np.abs(qop4[:,5] - qop4[:,1]) < thresh).sum() / qop4[:,1].size
    
    fig = plt.figure(figsize=(7,7), dpi=300)
    matplotlib.rcParams.update({'font.size': 12})
    #matplotlib.rcParams['axes.labelsize'] = 20
    rc('text', usetex=False)
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    
    
    
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.03, hspace=0.03) 
        
    ax0 = fig.add_subplot(gs[0,0])   
    ax1 = fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])
    
    opts = {'alpha':0.3, 's':2}
    ax0.scatter(qop4[:,4], qop4[:,1], label="Runz xcor: %0.1f%%" % runzXNum, color='#E53935', **opts)
    ax1.scatter(qop4[:,5], qop4[:,1], label="Runz ELM: %0.1f%%" % runzENum, color='#AB47BC', **opts)
    ax2.scatter(qop4[:,3], qop4[:,1], label="Autoz: %0.1f%%" % autozNum, color='#4CAF50', **opts)
    ax3.scatter(qop4[:,2], qop4[:,1], label="Marz: %0.1f%%" % marzNum, color='#2196F3', **opts)
    
    xlims = [0, 5]
    ylims = [0, 5]
    ax0.set_xlim(xlims)
    ax0.set_ylim(ylims)
    ax1.set_xlim(xlims)
    ax1.set_ylim(ylims)
    ax2.set_xlim(xlims)
    ax2.set_ylim(ylims)
    ax3.set_xlim(xlims)
    ax3.set_ylim(ylims)
    
    
    ax0.set_xticklabels([])
    ax1.set_xticklabels([])
    ax3.set_yticklabels([])
    ax1.set_yticklabels([])
    
    ax0.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax0.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax3.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax3.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax2.yaxis.get_major_ticks()[-1].label1.set_visible(False)
    ax2.xaxis.get_major_ticks()[-1].label1.set_visible(False)
    '''
    ax0.legend()
    ax1.legend()
    ax2.legend()
    ax3.legend()
    '''
    ax0.text(xlims[1] * 0.97, ylims[1]*0.9, "Runz xcor: %0.1f%%" % runzXNum, horizontalalignment='right', fontsize=14)
    ax1.text(xlims[1] * 0.97, ylims[1]*0.9, "Runz ELM: %0.1f%%" % runzENum, horizontalalignment='right', fontsize=14)
    ax2.text(xlims[1] * 0.97, ylims[1]*0.9, "Autoz: %0.1f%%" % autozNum, horizontalalignment='right', fontsize=14)
    ax3.text(xlims[1] * 0.97, ylims[1]*0.9, "Marz: %0.1f%%" % marzNum, horizontalalignment='right', fontsize=14)
    ax3.yaxis.get_major_ticks()[0].label1.set_visible(False)
    figtext(0.35,0.05,"Automatic redshift",fontdict={'fontsize':16})
    figtext(0.03,0.6,"Actual redshift", fontdict={'fontsize':16},rotation=90)
    
    #fig.savefig("run009Comp.png", bbox_inches='tight', dpi=600, transparent=True)
    #fig.savefig("run009Comp.pdf", bbox_inches='tight', dpi=1200, transparent=True)

    
res = loadData()
#plotQop4Comparison(res)
plotOffset(res)

