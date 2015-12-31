import os
import numpy as np
from astropy.io import ascii

import matplotlib, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
from pylab import *


directory = os.path.dirname(os.path.realpath(__file__))
marzDir = os.path.join(directory, 'marz110')
runzDir = os.path.join(directory, 'runz')
autozDir = os.path.join(directory, 'autoz')

def loadData():
    
    #marzDir = os.path.join(directory, 'marz')
    #marzDir = os.path.join(directory, 'marz_64')
    #marzDir = os.path.join(directory, 'marz_128')
    #marzDir = os.path.join(directory, 'marz_5dp')
    #marzDir = os.path.join(directory, 'marz_5dp')
    #marzDir = os.path.join(directory, 'marz_4window')
    #marzDir = os.path.join(directory, 'marz_5window')
    #marzDir = os.path.join(directory, 'marz_64k')
    #marzDir = os.path.join(directory, 'marz_5_3points')

    
    
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
    marzID = 1
    marzTID = 6
    marzID = 1
    marzQOP = -4
    objects = {}
    helios = []
    
    for filename in os.listdir(autozDir):
        with open(os.path.join(autozDir, filename), 'r') as f:
            data = ascii.read(f.read())
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
            count = 0
            helio = 0.0
            for line in f:                
                if line.startswith("#"):
                    if count == 1:
                        try:
                            string = line.split("vel corr = ")[1].split(" km/s")[0]
                            helio = float(string)
                            helios.append(helio)
                        except Exception:
                            helio = 0.0
                    count += 1
                    continue
                i = line.split()
                key = i[runzID]
                if len(i) < runzQOP: continue
                if objects.get(key):
                    objects[key]['COR'] = helio/3.0e5
                    objects[key]['RZ'] = float(i[runzZ]) 
                    objects[key]['QOP'] = int(i[runzQOP])
                    objects[key]['RXZ'] = float(i[runzXZ])
                    objects[key]['REZ'] = float(i[runzEZ])
                else:
                    objects[key] = {'RZ': float(i[runzZ]), 'QOP': int(i[runzQOP]), 'RXZ': float(i[runzXZ]) , 
                    'REZ': float(i[runzEZ]), 'COR': helio/3.0e5}
                    
    for filename in os.listdir(marzDir):
        if os.path.isfile(os.path.join(marzDir, filename)): # and "150513" in filename:
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
                        objects[idd]['MID'] = i[marzID].strip()
                        objects[idd]['MQOP'] = int(i[marzQOP].strip())
                        
                    else:
                        objects[idd] = {'MZ': float(i[marzZ]), 'MTID': i[marzTID].strip(), 'TYPE': i[marzT].strip(), 
                        'MID': i[marzID].strip(), 'MQOP': int(i[marzQOP].strip())}
                        
                        
    results = []
    names = []    
    for item in objects:
        i = objects[item]
        if i.get('MZ') is not None and i.get('RZ') is not None and i.get('RXZ') is not None:
            results.append([i.get('MQOP'), i.get('AZ'), i.get('RZ'), i.get('MZ'), i.get('RXZ'), i.get('REZ'), 
                            i.get('COR'), i.get('MTID')])
            names.append(i.get('MID'))
    print np.mean(helios)
    print np.var(helios)
            
    return np.array(results), np.array(names)
    
    
def plotOffset(res, names):
    t=1e-3
    qop4 = res[res[:,0] >= 4]
    use = qop4
    selection1 = qop4[:,2] < 1
    use = qop4[selection1]
    names = names[selection1]
    #print(use.size)
    for tt in np.arange(12) + 1:
        selection2 = use[:, -1] == tt
        use2 = use[selection2]
        marzDiff2 = - use2[:, 3] + use2[:, 2]
        md2 = marzDiff2[(np.abs(marzDiff2) < t)]
        print("t=%d\tMarz (mean, median, var):\t %.1e,\t %.1e,\t %.1e " % (tt, np.mean(md2), np.median(md2), np.sqrt(np.var(md2))))
        
    #print(use.size)
    
    marzDiff = - use[:, 3] + use[:, 2] #- use[:,6]
    autozDiff = - use[:, 1] + use[:, 2]
    runzDiff = -use[:, 5] + use[:, 2]
    runzDiff2 = -use[:, 4] + use[:, 2]
    #print(names[np.abs(marzDiff - 0.0002) < 0.00005])
    print(marzDiff.size)
    md = marzDiff[(np.abs(marzDiff) < t)]
    rd = runzDiff[(np.abs(runzDiff) < t)]
    ad = autozDiff[(np.abs(autozDiff) < t)]
    rdx = runzDiff2[(np.abs(runzDiff2) < t)]
    print("Totals")
    print("Autoz (mean, median, var):\t %.1e,\t %.1e,\t %.1e " % (np.mean(ad), np.median(ad), np.std(ad)))
    print("Marz (mean, median, var):\t %.1e,\t %.1e,\t %.1e " % (np.mean(md), np.median(md), np.std(md)))
    print("Runze (mean, median, var):\t %.1e,\t %.1e,\t %.1e " % ( np.mean(rd), np.median(rd), np.std(rd)))
    print("Runzx (mean, median, var):\t %.1e,\t %.1e,\t%.1e " % ( np.mean(rdx), np.median(rdx), np.std(rdx)))
    
    bin=numpy.arange(-0.001,0.001,0.00005)
    fig = plt.figure(figsize=(7,7), dpi=300)
    matplotlib.rcParams.update({'font.size': 12})
    #matplotlib.rcParams['axes.labelsize'] = 20
    rc('text', usetex=False)
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    
    ax = fig.add_subplot(1,1,1)
    ax.xaxis.set_ticks(numpy.arange(-0.001,0.002,0.0004))
    ax.hist(autozDiff, bin, color='g', label='Autoz', alpha=0.5)
    ax.hist(marzDiff, bin, color='b', label='Marz', alpha=0.6)
    ax.hist(runzDiff, bin, color='r', label='Runz ELM', alpha=0.2)
    #ax.hist(runzDiff2, bin, color='k', label='runz', alpha=0.1)
    #ax.set_ylim(0, 20)    
    ax.legend(frameon=False)
    ax.set_xlabel("$\Delta z$")
    ax.set_ylabel("Num spectra")
    #ax.set_title(marzDir)
    #ax.set_title("Redshift distribution")
    hh = 100
    ax.text(-0.00071, 540-hh, r"$\bar{\Delta z}\pm\sigma$", fontsize=16)
    ax.text(-0.00075, 500-hh, r"$\mathrm{Autoz}$", fontsize=16, horizontalalignment='right')
    ax.text(-0.00075, 470-hh, r"$\mathrm{Marz}$", fontsize=16, horizontalalignment='right')
    dd = 5
    ax.text(-0.00071, 500-hh, r"$(%0.f \pm %0.f)\times 10^{-%d} $" % (np.mean(ad)*10**dd, np.std(ad)*10**dd, dd), fontsize=16)
    ax.text(-0.00071, 470-hh, r"$(%0.f \pm %0.f)\times 10^{-%d} $" % (np.mean(md)*10**dd, np.std(md)*10**dd, dd), fontsize=16)

    
    ax.text(-0.00071, 400-hh, r"$\tilde{\Delta z}$", fontsize=16)
    ax.text(-0.00075, 370-hh, r"$\mathrm{Autoz}$", fontsize=16, horizontalalignment='right')
    ax.text(-0.00075, 340-hh, r"$\mathrm{Marz}$", fontsize=16, horizontalalignment='right')
    d1 = 5
    d2 = 6
    ax.text(-0.00071, 370-hh, r"$%0.f\times 10^{-%d} $" % (np.median(ad)*10**d1, d1), fontsize=16)
    ax.text(-0.00071, 340-hh, r"$%0.f\times 10^{-%d} $" % (np.median(md)*10**d1, d1), fontsize=16)
    
    fig.savefig("systematic2.png", bbox_inches='tight', dpi=600, transparent=True)
    fig.savefig("systematic2.pdf", bbox_inches='tight', transparent=True)

res, names = loadData()
#plotQop4Comparison(res)
plotOffset(res, names)


