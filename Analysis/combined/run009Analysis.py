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
    #marzDir = os.path.join(directory, 'marz_2')
    #marzDir = os.path.join(directory, 'marz_64k')
    #marzDir = os.path.join(directory, 'marz_5_3points')
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
    runzDiff = -use[:, 5] + use[:, 1]
    runzDiff2 = -use[:, 4] + use[:, 1]
    
    t=2e-3
    md = marzDiff[(np.abs(marzDiff) < t)]
    ad = autozDiff[(np.abs(autozDiff) < t)]
    rd = runzDiff[(np.abs(runzDiff) < t)]
    rdx = runzDiff2[(np.abs(runzDiff2) < t)]
    print("Marz (mean, median, var):\t %.1e,\t %.1e,\t %.1e " % (np.mean(md), np.median(md), np.var(md)))
    print("Autoz (mean, median, var):\t %.1e,\t %.1e,\t %.1e " % ( np.mean(ad), np.median(ad), np.var(ad)))
    print("Runze (mean, median, var):\t %.1e,\t %.1e,\t %.1e " % ( np.mean(rd), np.median(rd), np.var(rd)))
    print("Runzx (mean, median, var):\t %.1e,\t %.1e,\t%.1e " % ( np.mean(rdx), np.median(rdx), np.var(rdx)))
    
    bin=numpy.arange(-0.001,0.002,0.00004)
    fig = plt.figure(figsize=(17,7), dpi=300)
    matplotlib.rcParams.update({'font.size': 12})
    #matplotlib.rcParams['axes.labelsize'] = 20
    rc('text', usetex=False)
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    
    ax = fig.add_subplot(1,1,1)
    ax.xaxis.set_ticks(numpy.arange(-0.001,0.002,0.0002))
    ax.hist(marzDiff, bin, color='b', label='marz', alpha=0.5)
    ax.hist(autozDiff, bin, color='g', label='autoz', alpha=0.5)
    ax.hist(runzDiff, bin, color='r', label='runz', alpha=0.2)
    ax.set_xlabel("$\Delta z$")
    ax.set_ylabel("Num spectra")
    ax.set_title("QOP4 spectra for run009 and 2dFlens data")
    
    
res = loadData()
#plotQop4Comparison(res)
plotOffset(res)


