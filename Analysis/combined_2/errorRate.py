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
                        objects[idd]['MTID'] = i[marzTID].strip()
                        objects[idd]['TYPE'] = i[marzT].strip()
                    else:
                        objects[idd] = {'MZ': float(i[marzZ]), 'MTID': i[marzTID].strip(), 'TYPE': i[marzT].strip()}
                        
                        
    results = []
    
    for item in objects:
        i = objects[item]
        if i.get('MZ') is not None and i.get('AZ') is not None and i.get('RZ') is not None and i.get('RXZ') is not None:
            results.append([i.get('QOP'), i.get('RZ'), i.get('MZ'), i.get('AZ'), i.get('RXZ'), i.get('REZ')])
            
    return np.array(results)
    
    
    
    
    
def plotErrorRate(res):
    halpha = 6562.80;
    hbeta = 4861.325;
    o2 = 3728.5;
    lya = 1215.67;
    mg2 = 2798.75;
    c3 = 1908.73;
    o3d = 5006.91; 
    c4 = 1549.06;
    
    qop4 = res[res[:,0] >= 3]
    
    thresh = 0.01
    
    autozNum = 1.0* (np.abs(qop4[:,3] - qop4[:,1]) < thresh).sum()
    marzNum  = 1.0* (np.abs(qop4[:,2] - qop4[:,1]) < thresh).sum()
    runzxNum = 1.0* (np.abs(qop4[:,4] - qop4[:,1]) < thresh).sum()
    runzeNum = 1.0* (np.abs(qop4[:,5] - qop4[:,1]) < thresh).sum()
    
    
    autoz =  ((1 + qop4[:,3]) / (1 + qop4[:,1]))[np.abs(qop4[:,3] - qop4[:,1]) > thresh]
    marz  =  ((1 + qop4[:,2]) / (1 + qop4[:,1]))[np.abs(qop4[:,2] - qop4[:,1]) > thresh]
    runzx =  ((1 + qop4[:,4]) / (1 + qop4[:,1]))[np.abs(qop4[:,4] - qop4[:,1]) > thresh]
    runze =  ((1 + qop4[:,5]) / (1 + qop4[:,1]))[np.abs(qop4[:,5] - qop4[:,1]) > thresh]
    
    msr = (100.0 * (1.0-(1.0*marz.size / qop4[:,0].size )))
    asr = (100.0 * (1.0-(1.0*autoz.size / qop4[:,0].size )))
    rxsr = (100.0 * (1.0-(1.0*runzx.size / qop4[:,0].size )))
    resr = (100.0 * (1.0-(1.0*runze.size / qop4[:,0].size )))
    print("marz %0.3f%% success rate, %0.3f%% fail rate" % (msr, 100 - msr))
    print("autoz %0.3f%% success rate, %0.3f%% fail rate" % (asr, 100 - asr))
    print("runzx %0.3f%% success rate, %0.3f%% fail rate" % (rxsr, 100 - rxsr))
    print("runze %0.3f%% success rate, %0.3f%% fail rate" % (resr, 100 - resr))
    
    fig = plt.figure(figsize=(7,7), dpi=300)
    matplotlib.rcParams.update({'font.size': 12})
    #matplotlib.rcParams['axes.labelsize'] = 20
    rc('text', usetex=False)
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    
    
    
    gs = gridspec.GridSpec(4, 1)
    gs.update(wspace=0.0, hspace=0.0) 
        
    ax0 = fig.add_subplot(gs[0])   
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    
    bins = np.linspace(0.2,1.5,200)
    width = 1 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    
    hist0 = 100 * np.histogram(runzx, bins=bins)[0] / autozNum
    hist1 = 100 * np.histogram(runze, bins=bins)[0] / marzNum
    hist2 = 100 * np.histogram(autoz, bins=bins)[0] / runzxNum
    hist3 = 100 * np.histogram(marz,  bins=bins)[0] / runzeNum
    
    maxes = max([max(hist0), max(hist1), max(hist2), max(hist3)])
    

    ax0.plot([halpha/o3d, halpha/o3d], [1.1 * maxes, 0], ':', color='#555555')
    ax1.plot([halpha/o3d, halpha/o3d], [1.1 * maxes, 0], ':', color='#555555')
    ax2.plot([halpha/o3d, halpha/o3d], [1.1 * maxes, 0], ':', color='#555555')
    ax3.plot([halpha/o3d, halpha/o3d], [1.1 * maxes, 0], ':', color='#555555')
    ax2.text(1.23, 2, r'$\mathrm{H}\alpha\mathrm{/O[III]}$', fontsize=14)
    
    ax0.plot([o2/o3d, o2/o3d], [1.1 * maxes, 0], ':', color='#555555')
    ax1.plot([o2/o3d, o2/o3d], [1.1 * maxes, 0], ':', color='#555555')
    ax2.plot([o2/o3d, o2/o3d], [1.1 * maxes, 0], ':', color='#555555')
    ax3.plot([o2/o3d, o2/o3d], [1.1 * maxes, 0], ':', color='#555555')
    ax0.text(0.63, 0.92, r'$\mathrm{O[II]/O[III]}$', fontsize=14)
    
    ax0.plot([mg2/halpha, mg2/halpha], [1.1 * maxes, 0], ':', color='#555555')
    ax1.plot([mg2/halpha, mg2/halpha], [1.1 * maxes, 0], ':', color='#555555')
    ax2.plot([mg2/halpha, mg2/halpha], [1.1 * maxes, 0], ':', color='#555555')
    ax3.plot([mg2/halpha, mg2/halpha], [1.1 * maxes, 0], ':', color='#555555')
    ax0.text(0.31, 0.92, r'$\mathrm{MgII/H}\alpha$', fontsize=14)

    ax0.plot([o2/halpha, o2/halpha], [1.1 * maxes, 0], ':', color='#555555')
    ax1.plot([o2/halpha, o2/halpha], [1.1 * maxes, 0], ':', color='#555555')
    ax2.plot([o2/halpha, o2/halpha], [1.1 * maxes, 0], ':', color='#555555')
    ax3.plot([o2/halpha, o2/halpha], [1.1 * maxes, 0], ':', color='#555555')
    ax3.text(0.455, 1.5, r'$\mathrm{O[II]/H}\alpha$', fontsize=14)
    
    a0 = ax0.bar(center, hist0, align='center', width=width, edgecolor="none", facecolor="#E53935", label="Runz xcor")
    a1 = ax1.bar(center, hist1, align='center', width=width, edgecolor="none", facecolor="#AB47BC", label="Runz ELM")
    a2 = ax2.bar(center, hist2, align='center', width=width, edgecolor="none", facecolor="#4CAF50", label="Autoz")
    a3 = ax3.bar(center, hist3, align='center', width=width, edgecolor="none", facecolor="#2196F3", label="Marz")
    '''
    
    ax0.plot(center, hist0, linewidth=2, color="#E53935")
    ax1.plot(center, hist1, linewidth=2, color="#AB47BC")
    ax2.plot(center, hist2, linewidth=2, color="#4CAF50")
    ax3.plot(center, hist3, linewidth=2, color="#2196F3")
    '''
    

    
    
    ax0.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax3.yaxis.set_major_locator(plt.MaxNLocator(4))
    
    ax0.yaxis.get_major_ticks()[0].label1.set_visible(False)
    ax1.yaxis.get_major_ticks()[0].label1.set_visible(False)
    ax2.yaxis.get_major_ticks()[0].label1.set_visible(False)

    
    ax3.set_xlabel(r"$(1 + z_A)/(1 + z_M)$", fontsize=18)    
    
    ax0.get_xaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    
    xmax = 1.6
    ax0.set_ylim(0, 1.1 * max(hist0))
    ax1.set_ylim(0, 1.1 * max(hist1))
    ax2.set_ylim(0, 1.1 * max(hist2))
    ax3.set_ylim(0, 1.1 * max(hist3))
    ax0.set_xlim(0.2, xmax)
    ax1.set_xlim(0.2, xmax)
    ax2.set_xlim(0.2, xmax)
    ax3.set_xlim(0.2, xmax)
    
    ax0.text(0.98 * xmax, 0.9 * max(hist0), 'Runz xcor', fontsize=14, horizontalalignment='right')
    ax1.text(0.98 * xmax, 0.9 * max(hist1), 'Runz ELM', fontsize=14, horizontalalignment='right')
    ax2.text(0.98 * xmax, 0.9 * max(hist2), 'Autoz', fontsize=14, horizontalalignment='right')
    ax3.text(0.98 * xmax, 0.9 * max(hist3), 'Marz', fontsize=14, horizontalalignment='right')
    
    
    
    figtext(0.03,0.7,r"probability to misidentify [%]", fontdict={'fontsize':15},rotation=90)

    
    #fig.savefig("errorRateqop3.png", bbox_inches='tight', dpi=600, transparent=True)
    fig.savefig("errorRateqop3.pdf", bbox_inches='tight', transparent=True)
    
    
res = loadData()
plotErrorRate(res)

'''
l = [1025.72,1215.67,1240.14,1400.0,1549.06,1908.73,2798.75,3728.48,3869.81,3933.66,3968.46,4102.92,4304.4,4341.69,4861.32,4958.91,5006.84,5175.3,5894.0,6549.84,6562.80,6585.23,6718.32,6732.71]
for i in range(len(l)):
    for j in range(len(l)):
        if abs((l[i] / l[j]) - 1.311) < 0.005:
            print(l[i], l[j], l[i]/l[j])
'''