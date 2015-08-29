import os
import numpy as np
import asciitable

import matplotlib, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
from pylab import *


def loadData():
    directory = os.path.dirname(os.path.realpath(__file__))
    
    marzDir = os.path.join(directory, 'marz_new')
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
    marzQOP = 13
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
                        objects[idd]['MQOP'] = int(i[marzQOP])
                    else:
                        objects[idd] = {'MZ': float(i[marzZ]), 'MTID': i[marzTID].strip(), 'TYPE': i[marzT].strip(), 'MQOP': int(i[marzQOP])}
                        
                        
    results = []
    
    for item in objects:
        i = objects[item]
        if i.get('MZ') is not None and i.get('AZ') is not None and i.get('RZ') is not None and i.get('RXZ') is not None:
            results.append([i.get('QOP'), i.get('RZ'), i.get('MZ'), i.get('AZ'), i.get('RXZ'), i.get('REZ'), i.get('MQOP')])
            
    return np.array(results)
    
    
    
    
    
def plotErrorRate(res):
    qop6 = res[res[:,0] == 6]
    qop4 = res[res[:,0] == 4]
    qop3 = res[res[:,0] == 3]
    qop2 = res[res[:,0] == 2]
    qop1 = res[res[:,0] == 1]
    
    possible = [1,2,3,4,6]
    ind = np.arange(len(possible))
    qop4ratio = 100.0 * np.array([(qop4[:,6] == x).sum() for x in possible]) / qop4[:,6].size
    qop3ratio = 100.0 * np.array([(qop3[:,6] == x).sum() for x in possible]) / qop3[:,6].size
    qop2ratio = 100.0 * np.array([(qop2[:,6] == x).sum() for x in possible]) / qop2[:,6].size
    qop1ratio = 100.0 * np.array([(qop1[:,6] == x).sum() for x in possible]) / qop1[:,6].size
    qop6ratio = 100.0 * np.array([(qop6[:,6] == x).sum() for x in possible]) / qop6[:,6].size
    
    print(qop4ratio)
    print(qop3ratio)
    print(qop2ratio)
    print(qop1ratio)
    print(qop6ratio)
    fig = plt.figure(figsize=(7,7), dpi=300)
    matplotlib.rcParams.update({'font.size': 12})
    #matplotlib.rcParams['axes.labelsize'] = 20
    rc('text', usetex=False)
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    
    ax = fig.add_subplot(111)
    
    ax.bar(ind, qop4ratio, color="#4CAF50", label="QOP4")
    ax.bar(ind, qop3ratio, color="#2196F3", label="QOP3", bottom=qop4ratio)
    ax.bar(ind, qop2ratio, color="#FF9800", label="QOP2", bottom=qop4ratio+qop3ratio)
    ax.bar(ind, qop1ratio, color="#E53935", label="QOP1", bottom=qop4ratio+qop3ratio+qop2ratio)
    ax.bar(ind, qop6ratio, color="#673AB7", label="QOP6", bottom=qop4ratio+qop3ratio+qop2ratio+qop1ratio)
    
    plt.xticks(ind + 0.5, ('1','2','3','4','6'))
    ax.set_xlabel("Marz estimated QOP", fontsize=14)
    ax.set_ylabel("Assignment chance [%]", fontsize=14)
    ax.legend(loc="upper right")
    

    
    #fig.savefig("autoQOP.png", bbox_inches='tight', dpi=600, transparent=True)
    fig.savefig("autoQOP.pdf", bbox_inches='tight', transparent=True)
    
    
    
res = loadData()
plotErrorRate(res)

