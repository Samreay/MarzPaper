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
    runzDir = os.path.join(directory, 'runz')
    
    
    
    runzID = 1
    runzZ = 19
    runzXZ = 16
    runzEZ = 11
    runzQOP = 22
    marzZ = 8
    marzT = 5
    marzTID = 6
    marzID = 1
    marzQOP = 13
    objects = {}
    
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
                        objects[idd]['MQOP1'] = np.floor(float(i[marzQOP])) / 100.0
                        objects[idd]['MQOP2'] = (float(i[marzQOP]) - np.floor(float(i[marzQOP]))) * 100.0
                    else:
                        objects[idd] = {'MZ': float(i[marzZ]), 'MTID': i[marzTID].strip(), 'TYPE': i[marzT].strip(), 'MQOP1': np.floor(float(i[marzQOP])) / 100.0, 'MQOP2': (float(i[marzQOP]) - np.floor(float(i[marzQOP]))) * 100.0}
                        
                        
    results = []
    
    for item in objects:
        i = objects[item]
        if i.get('MZ') is not None and i.get('QOP'):
            results.append([i.get('QOP'),  i.get('MQOP1'), i.get('MQOP2'), i.get('RZ'), i.get('MZ'),])
            
    return np.array(results)
    
res = loadData()

fig = plt.figure(figsize=(21,15), dpi=300)
matplotlib.rcParams.update({'font.size': 12})
#matplotlib.rcParams['axes.labelsize'] = 20
rc('text', usetex=False)
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12

#ax = fig.add_subplot(111)

qops = [4,3,2,1]
colours = ["#4CAF50", "#2196F3", "#E5A9A5", "#E53935", "#673AB7"]
qopArr = [res[res[:,0] == p] for p in qops]

sub = 2.4
n2 = 0.8
data = [(x[:,1]-sub) * ((x[:,1]/x[:,2])**n2 - 0.5)  for x in qopArr]

i = 1
for q, c, d, arr in zip(qops, colours, data, qopArr):
    #ax.scatter(arr[:, 1], d, color=c,  alpha=0.6, s=7)
    hist, bins = np.histogram(d, bins=200)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    

    ax0 = fig.add_subplot(4,1,i)
    i += 1
    ax0.bar(center, hist, align='center', width=width, facecolor=c)
    #ax0.set_xlim(-20, 300)

