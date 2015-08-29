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

#from pympler.tracker import SummaryTracker




class Fitted(object):
    def __init__(self, debug=False):
        self._debug = debug
    def getIndex(self, param):
        for i, x in enumerate(self.getParams()):
            if x[0] == param:
                return i + 3
        return None
    def debug(self, string):
        if self._debug:
            print(string)
            sys.stdout.flush()
    def getNumParams(self):
        raise NotImplementedError
    def getChi2(self, params):
        raise NotImplementedError
    def getParams(self):
        raise NotImplementedError
        
class ThresholdFitter(Fitted):
    def __init__(self):
        self.params = [('sub', 0, 5, 'sub'), ('pow', 0, 2, 'pow'), ('q2', 0, 5, 'q2'), ('q3', 0, 10, 'q3'), ('q4', 0, 15, 'q4')]
        self.res = self.loadData()
        self.qops = [4,3,2,1]
        self.colours = ["#4CAF50", "#2196F3", "#E5A9A5", "#E53935", "#673AB7"]
        self.qopArr = [self.res[self.res[:,0] == p] for p in self.qops]

        
    def getParams(self):
        return self.params

    def loadData(self):
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
                results.append([i.get('QOP'),  i.get('MQOP1'), i.get('MQOP2'), i.get('RZ'), i.get('MZ')])
            
        return np.array(results)

    def getNumParams(self):
        return len(self.params)

    def getChi2(self, params):
        allParams = self.getParams()
        for i, p in enumerate(params):
            if p < allParams[i][1] or p > allParams[i][2]:
                return None
        data = [(x[:,1]-params[0])**params[1] * (x[:,1]/x[:,2]) for x in self.qopArr]        
        #print(x[:,1])
        #print(x[:,2])
        #print("ARGH")
        b2 = params[2]
        b3 = params[3]
        b4 =  params[4]

        qop4Vals = data[0]
        qop3Vals = data[1]
        qop2Vals = data[2]
        qop1Vals = data[3]


        qop4chi2 = - 5.0 * (qop4Vals > b4).sum()/qop4Vals.size     +  10.0 * (qop3Vals > b4).sum()/qop3Vals.size  + 50.0 *  (qop2Vals > b4).sum()/qop2Vals.size + 100.0 * (qop1Vals > b4).sum()/qop1Vals.size 
        qop3chi2 = - 3.0 * (qop3Vals > b3).sum()/qop3Vals.size + ((qop4Vals > b3) & (qop4Vals < b4)).sum()/qop4Vals.size + 5.0 * (qop2Vals > b3).sum()/qop2Vals.size + 10 * (qop1Vals > b3).sum()/qop1Vals.size
        qop2chi2 = - 1.0 * ((qop2Vals > b2) & (qop2Vals < b3)).sum()/qop2Vals.size + 5.0 * ((qop3Vals > b2) & (qop3Vals < b3)).sum()/qop3Vals.size + 10 * ((qop4Vals > b2) & (qop4Vals < b3)).sum()/qop4Vals.size    
        qop1chi2 = - 1.0 * ((qop1Vals < b2)).sum()/qop1Vals.size  + 2 * (qop2Vals < b2).sum()/qop2Vals.size + 5 * (qop3Vals < b2).sum()/qop3Vals.size + 10 * (qop4Vals < b2).sum()/qop4Vals.size
        #chi2 = (data[0] < b1).sum()/(1.0*data[0].size) +  1.3 * (data[1] > b1).sum()/(1.0*data[1].size) + (data[1] < b2).sum()/(1.0*data[1].size) +  1.2*(data[2] > b2).sum()/(1.0*data[2].size) + 0.2 * (data[2] < b3).sum()/(1.0*data[2].size) +  1.4*(data[3] > b2).sum()/(1.0*data[3].size) + 0.1*(data[3] > b3).sum()/(1.0*data[3].size)
        #print(b4)
        #print(chi2)
        #print(1.0 * (qop4Vals > b4).sum()/qop4Vals.size)
        #print(10.0 * (qop3Vals > b4).sum()/qop3Vals.size)
        #print(50.0 *  (qop2Vals > b4).sum()/qop2Vals.size)
        #print(100.0 * (qop1Vals > b4).sum()/qop1Vals.size)
        #raise Exception("fuck")
        chi2 = 1.5*qop4chi2 + qop3chi2 + 0.5*qop2chi2 + 0.25*qop1chi2   
        return chi2
    
class CambMCMCManager(object):
    def __init__(self, uid, fitter, debug=False):
        self.uid = uid
        self.fitter = fitter
        self._debug = debug
        self.configureMCMC()
        self.configureSaving()
        self.steps = {}
        
    """ Stuff that people all call """
    def configureMCMC(self, numStarts=50, sigmaCalibrationLength=50, calibrationLength=1000, numCalibrations=7, desiredAcceptRatio=0.4, thinning=2, maxSteps=100000):
        self.sigmaCalibrationLength = sigmaCalibrationLength
        self.calibrationLength = calibrationLength
        self.numCalibrations = numCalibrations
        self.desiredAcceptRatio = desiredAcceptRatio
        self.thinning = thinning
        self.maxSteps = maxSteps
        self.numStarts = numStarts
        
    def configureSaving(self, stepsPerSave=100):
        self.stepsPerSave = stepsPerSave
        
    def debug(self, string):
        if self._debug:
            print(string)
    
    
    def doWalk(self, index=0, outputToFile=False):
        cambMCMC = CambMCMC(self)
        result = cambMCMC(index, outputToFile)
        return result
        
    def consolidateData(self):
        self.steps = {}
        self.finalSteps = None
        index = 0
        while (True):
            steps = self.loadSteps(index, fillZeros=False)
            if steps is None:
                break
            else:
                self.steps[index] = steps
            index += 1
        toUse = []
        for x,v in self.steps.iteritems():
            if v[:,0].size > self.calibrationLength * self.numCalibrations:
                dataArray = v[self.calibrationLength * self.numCalibrations :: self.thinning, :]
                dataArray[:,2] = x
                toUse.append(dataArray)
        if len(toUse) > 0:
            self.finalSteps = np.concatenate(tuple(toUse))
            
        if self.finalSteps is not None:
            self.debug("Consolidated data contains %d data points" % self.finalSteps.shape[0])
        else:
            self.debug("Consolidated data is empty")
    
    def testConvergence(self):
        finals = [self.finalSteps[self.finalSteps[:,2] == i, 3:] for i in np.unique(self.finalSteps[:,2])]
        variances = [np.var(f, axis=0, ddof=1) for f in finals]
        w = np.mean(variances, axis=0)
    
        allMean = np.mean(self.finalSteps[:,3:], axis=0)
        walkMeans = [np.mean(f, axis=0) for f in finals]
        b = np.zeros(len(allMean))
        
        for i, walkMean in enumerate(walkMeans):
            b += 1.0 * (1.0 / (len(walkMeans) - 1)) * finals[i][:,0].size * np.power((walkMean - allMean), 2)
    
        meanN = 0
        for f in finals:
            meanN += f[:,0].size
        meanN *= 1.0 / len(finals)
        
        estVar = (1 - (1.0 / meanN)) * w + (1.0 / meanN) * b
        r = np.sqrt(estVar / w)
        print("estvar and r coming up")
        print(estVar)
        print(r)
        return estVar, r
        
    def getParameterBounds(self, numBins=50):
        interpVals = np.array([0.15865, 0.5, 0.84135])
        res = []
        string = "\\begin{align}\n"
        if self.finalSteps is None:
            return None
        for paramList in self.fitter.getParams():
            param = paramList[0]
            data = self.finalSteps[:, self.fitter.getIndex(param)]
            hist, bins = np.histogram(data, bins=numBins)
            centers = (bins[:-1] + bins[1:]) / 2
            dist = 1.0 * hist.cumsum() / hist.sum()
            bounds = interp1d(dist, centers, bounds_error=False)(interpVals)
            maxL = data[self.finalSteps[:,0].argmin()]
            maxLchi2 = self.finalSteps[:,0].min()
            res.append((param, maxL, bounds))
            string += "%s &= %0.3f^{+%0.3f}_{-%0.3f} \\\\ \n" % (paramList[3].replace("$", ""), bounds[1], bounds[2]-bounds[1], bounds[1]-bounds[0])
            print("%s, maxL=%0.4f with chi2=%0.3f, 1sigmaDistBounds=(%0.4f, %0.4f, %0.4f)" % (param, maxL, maxLchi2, bounds[0], bounds[1], bounds[2]))
        string += "\\end{align}\n"
        print(string)
        return res

    def plotWalk(self, param1, param2, final=True, size=(13,9)):
        
        fig = plt.figure(figsize=size, dpi=300)
        matplotlib.rcParams.update({'font.size': 14})
        ax0 = fig.add_subplot(1,1,1)
        
        if final:
            for i in np.unique(self.finalSteps[:, 2]):
                d = self.finalSteps[self.finalSteps[:,2] == i]
                ax0.plot(d[:, self.fitter.getIndex(param1)], d[:, self.fitter.getIndex(param2)], 'o', markersize=5, alpha=0.2)
        else:
            for x in self.steps:
                ax0.plot(self.steps[x][self.steps[x][:,0] >= 0, self.fitter.getIndex(param1)], self.steps[x][self.steps[x][:,0] >= 0, self.fitter.getIndex(param2)], 'o-', markersize=5, alpha=0.05)
        ax0.set_xlabel(param1)
        ax0.set_ylabel(param2)
    def getHistogram(self, param, bins=50, size=(13,9)):
        ''' Because doSteps won't finish, you run it and then ask for results when you are happy with the dataset'''
        

        data = self.finalSteps[:, self.fitter.getIndex(param)]
        hist, bins = np.histogram(data, bins=bins)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        
        fig = plt.figure(figsize=size, dpi=300)
        matplotlib.rcParams.update({'font.size': 14})
        ax0 = fig.add_subplot(1,1,1)
        ax0.bar(center, hist, align='center', width=width)
        ax0.set_xlabel(param)
        ax0.set_ylabel("Counts")
        
    def plotResults(self, size=(20,20), filename=None, useGaussianKernelDensityEstimation=False):
        fig = plt.figure(figsize=size, dpi=300)
        matplotlib.rcParams.update({'font.size': 14})
        matplotlib.rcParams['axes.labelsize'] = 20
        rc('text', usetex=False)
        matplotlib.rcParams['xtick.labelsize'] = 14
        matplotlib.rcParams['ytick.labelsize'] = 14
        params = []
        for p in self.fitter.getParams():
            if p[0] != "sigmav":
                params.append(p)
        
        n = len(params)
        gs = gridspec.GridSpec(n, n)
        gs.update(wspace=0.0, hspace=0.0) 
        
        for i in range(n):
            for j in range(i + 1):
                ax = fig.add_subplot(gs[i,j])
                pi = params[i]
                pj = params[j]
                if i == j:
                    self.plotBars(ax, pi[0], False, i==n-1)
                else:
                    if useGaussianKernelDensityEstimation:
                        self.plotContourGKDE(ax, pi[0], pj[0], j==0, i==n-1)
                    else:
                        self.plotContour(ax, pi[0], pj[0], j==0, i==n-1)
                if i == n - 1:
                    ax.set_xlabel(pj[3])
                if j == 0:
                    ax.set_ylabel(pi[3])
                    
        if filename is not None:
            fig.savefig("%s.png" % filename, bbox_inches='tight', dpi=300, transparent=True)



    def plotBars(self, ax, param, showylabel, showxlabel, bins=50):
        self.debug("Plotting bars for %s" % param)
        data = self.finalSteps[:, self.fitter.getIndex(param)]
        hist, bins = np.histogram(data, bins=bins)
        width = 1 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        
        ax.bar(center, hist, align='center', width=width, edgecolor="none", facecolor="#333333")
        ax.set_xlim([min(data), max(data)])
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        if not showylabel:
            ax.set_yticklabels([])
        if not showxlabel:
            ax.set_xticklabels([])
    
    def convert_to_stdev(self, sigma):
        """
        From astroML
        
        Given a grid of log-likelihood values, convert them to cumulative
        standard deviation.  This is useful for drawing contours from a
        grid of likelihoods.
        """
        #sigma = np.exp(logL)
    
        shape = sigma.shape
        sigma = sigma.ravel()
    
        # obtain the indices to sort and unsort the flattened array
        i_sort = np.argsort(sigma)[::-1]
        i_unsort = np.argsort(i_sort)
    
        sigma_cumsum = 1.0* sigma[i_sort].cumsum()
        sigma_cumsum /= sigma_cumsum[-1]
    
        return sigma_cumsum[i_unsort].reshape(shape)
        
    def plotContour(self, ax, param1, param2, showylabel, showxlabel, levels=[0, 0.6827, 0.9545]):
        self.debug("Plotting contour for %s and %s" % (param1, param2))
        
        ys = self.finalSteps[:, self.fitter.getIndex(param1)]
        xs = self.finalSteps[:, self.fitter.getIndex(param2)]
        minIndex = self.finalSteps[:,0].argmin()
        bins = np.sqrt(len(xs) / 30);
        bins = 25
        L_MCMC, xBins, yBins = np.histogram2d(xs, ys, bins=bins)
        L_MCMC[L_MCMC == 0] = 1E-16  # prevents zero-division errors
        vals = self.convert_to_stdev(L_MCMC.T)
        
        ax.contourf(0.5 * (xBins[:-1] + xBins[1:]), 0.5 * (yBins[:-1] + yBins[1:]), vals, levels=levels, colors=("#BBDEFB", "#90CAF9"))
        ax.contour(0.5 * (xBins[:-1] + xBins[1:]), 0.5 * (yBins[:-1] + yBins[1:]), vals, levels=levels, colors="k", alpha=0.5)
        ax.scatter(xs[minIndex], ys[minIndex], color='k', s=20)
        ax.set_xlim([min(xs), max(xs)])
        ax.set_ylim([min(ys), max(ys)])
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        if not showylabel:
            ax.set_yticklabels([])            
        if not showxlabel:
            ax.set_xticklabels([])
        
    def plotContourGKDE(self, ax, param1, param2, showylabel, showxlabel, contourNGrid=100, contourFractions=[0.6827, 0.9545]):
        self.debug("Plotting gkde contour for %s and %s" % (param1, param2))
        ykde = self.finalSteps[:, self.fitter.getIndex(param1)]
        xkde = self.finalSteps[:, self.fitter.getIndex(param2)]
        gkde = scipy.stats.gaussian_kde([xkde,ykde])
        #bw = np.power(len(ykde), -1/(4+self.fitter.getNumParams()))
        #gkde.covariance_factor = lambda: bw
        #gkde._compute_covariance()
        xgrid, ygrid = numpy.mgrid[min(xkde):max(xkde):contourNGrid * 1j, min(ykde):max(ykde):contourNGrid * 1j]
        zvals = numpy.array(gkde.evaluate([xgrid.flatten(), ygrid.flatten()])).reshape(xgrid.shape)
        contours = self.contour_enclosing(ax, xkde, ykde, contourFractions, xgrid, ygrid, zvals)
        ax.set_xlim([min(xkde), max(xkde)])
        ax.set_ylim([min(ykde), max(ykde)])
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        if not showylabel:
            ax.set_yticklabels([])            
        if not showxlabel:
            ax.set_xticklabels([])
        
        
    def contour_enclosing(self, axes, x, y, fractions, xgrid, ygrid, zvals, nstart=100):
        """Plot contours encompassing specified fractions of points x,y.
        """
        # Generate a large set of contours initially.
        contours = axes.contour(xgrid, ygrid, zvals, nstart, extend='both')
        # Set up fracs and levs for interpolation.
        levs = contours.levels
        fracs = numpy.array(self.fracs_inside_contours(x,y,contours))
        sortinds = numpy.argsort(fracs)
        levs = levs[sortinds]
        fracs = fracs[sortinds]
        # Find the levels that give the specified fractions.
        levels = scipy.interp(fractions, fracs, levs)
    
        # Remove the old contours from the graph.
        for coll in contours.collections:
            coll.remove()
        # Reset the contours
        contours.__init__(axes, xgrid, ygrid, zvals, levels)
        return contours
        
    def frac_inside_poly(self, x,y,polyxy):
        """Calculate the fraction of points x,y inside polygon polyxy.
        
        polyxy -- list of x,y coordinates of vertices.
    
        """
        xy = numpy.vstack([x,y]).transpose()
        path = Path(polyxy) 
        return float(sum(path.contains_points(xy)))/len(x)
    
    def fracs_inside_contours(self, x, y, contours):
        """Calculate the fraction of points x,y inside each contour level.
    
        contours -- a matplotlib.contour.QuadContourSet
        """
        fracs = []
        for (icollection, collection) in enumerate(contours.collections):
            path = collection.get_paths()[0]
            pathxy = path.vertices
            frac = self.frac_inside_poly(x,y,pathxy)
            fracs.append(frac)
        return fracs


    def getSaveFileName(self, index, cov=False):
        return "cambMCMC_%s_%s_%d.txt" % (self.uid, 'cov' if cov else 'steps', index)
        
    def loadSteps(self, index, fillZeros=True):
        try:
            res = np.loadtxt(self.getSaveFileName(index))
            self.debug("Save file found containing %d steps" % res[:,0].size)
            if (fillZeros and res[:,0].size < self.maxSteps):
                zeros = np.zeros((self.maxSteps - res[:,0].size, 3 + self.fitter.getNumParams()))
                zeros[:, 0] -= 1
                return np.concatenate((res, zeros))
            else:
                return res
        except:
            self.debug("No save file found")
            if fillZeros:
                zeros = np.zeros((self.maxSteps, 3 + self.fitter.getNumParams()))
                zeros[:, 0] -= 1
                return zeros
            else:
                return None









    
class CambMCMC(object):
    
    def __init__(self, parent):
        self._debug = parent._debug
        self.parent = parent
        self.uid = parent.uid
        self.outputFile = None
        self.fitter = parent.fitter        
        self.sigmaCalibrationLength = parent.sigmaCalibrationLength
        self.calibrationLength = parent.calibrationLength
        self.numCalibrations = parent.numCalibrations
        self.desiredAcceptRatio = parent.desiredAcceptRatio
        self.thinning = parent.thinning
        self.maxSteps = parent.maxSteps
        self.stepsPerSave = parent.stepsPerSave
        self.numStarts = parent.numStarts
        
    """ Stuff the class uses to do everything """
    def debug(self, text):
        if self._debug:
            if self.outputFile is not None:
                print(text, file = self.outputFile)
            else:
                print(text)
                sys.stdout.flush()
    

    def __call__(self, index=0, fileLogging=False):

        if fileLogging:
            self.outputFile = open('cambMCMC_%s_%d_log.txt' % (self.uid, index), 'w+', 8000)
        self.debug("doMCMC for walk %d" % index)
        steps = self.parent.loadSteps(index)
        sigma, sigmaRatio, rotation = self.initialiseSigma(steps)       
        
        for i, step in enumerate(steps):
            #if (i != 0 and i % self.sigmaCalibrationLength == 0):
            #    tracker = SummaryTracker()
            if step[0] >= 0:
                continue
            if i == 0:
                self.debug("Starting at beginning")
                oldParams = None
                oldChi2 = 9e9
                for x in range(self.numStarts):
                    p = self.getInitialParams()
                    c = self.fitter.getChi2(p)
                    self.debug("Got chi2 %0.2f when starting at params %s" % (c, p))
                    if c < oldChi2:
                        oldChi2 = c
                        oldParams = p
                self.debug("Best starting location of chi2 %0.2f found at %s" % (oldChi2, oldParams))
            else:
                sigmaRatio = steps[i-1, 2]
                oldParams = steps[i-1, 3:]
                oldChi2 = steps[i-1, 0]
                
            isBurnin = (i <= self.numCalibrations * self.calibrationLength)
        
            if (i != 0 and isBurnin and i % self.sigmaCalibrationLength == 0):
                sigma, sigmaRatio = self.adjustSigma(steps, i, sigma, sigmaRatio)        
                
            if (i != 0 and isBurnin and i % self.calibrationLength == 0 and i <= (self.numCalibrations-1) * self.calibrationLength):
                sigma, sigmaRatio, rotation = self.adjustCovariance(steps, i / self.calibrationLength, sigmaRatio)
                
            steps[i, :], oldweight = self.doStep(oldParams, oldChi2, sigma, sigmaRatio, rotation, isBurnin)
            steps[i - 1, 1] = oldweight
            self.debug("Done step %d" % i)
            
            if ((i + 1) % self.stepsPerSave == 0 and i > 0):
                self.saveData(index, steps)

                
        if self.outputFile is not None:
            self.outputFile.close()
            
        return steps
        
    def doStep(self, oldParams, oldChi2, sigma, sigmaRatio, rotation, isBurnin):
        attempts = 1
        while(True):        
            params = self.getPotentialPoint(oldParams, sigmaRatio * sigma, rotation)
            newChi2 = self.fitter.getChi2(params)

            if newChi2 is None:
                self.debug("REJECT! Outside allowed bounds!")
            else:
                prob = np.exp((oldChi2 - newChi2)/2)
                if prob > 1 or prob > np.random.uniform():
                    self.debug("ACCEPT! chi2s %0.5f vs %0.5f" % (newChi2, oldChi2))
                    return np.concatenate((np.array([newChi2, 0.0, sigmaRatio]), params)), attempts
                else:
                    self.debug("REJECT! chi2s %0.5f vs %0.5f" % (newChi2, oldChi2))
                    attempts += 1
                    if isBurnin and attempts >= 20 and attempts % 10 == 0:
                        sigmaRatio *= 0.9
                
        
    def saveData(self, index, steps):
        self.debug("\nSaving data")
        try:
            np.savetxt(self.parent.getSaveFileName(index), steps[steps[:,0] >= 0])
            self.debug("Saved data\n\n\n")
        except Exception as e:
            self.debug(e.strerror)
            
    def getPotentialPoint(self, oldParams, sigma, rotation):
        rotParams = np.dot(oldParams, rotation)
        newRotParams = rotParams + sigma * np.random.normal(size=sigma.size)
        newParams = np.dot(rotation, newRotParams)
        return newParams
        
    def initialiseSigma(self, steps):
        if (steps is not None and steps[steps[:,0] >= 0, 0].size > self.calibrationLength):
            self.debug("Calculating rotation matrix from existing steps")
            i = min(self.numCalibrations * self.calibrationLength, self.calibrationLength * (steps[steps[:,0] >= 0,0].size / self.calibrationLength)) / self.calibrationLength
            return self.adjustCovariance(steps, i, 0.5)
            
        else:
            self.debug("Initialising sigma and rotation to defaults")
            toReturn = (0.001 * np.array([(x[2] - x[1]) for x in self.fitter.getParams()]), 1, np.identity(self.fitter.getNumParams()))
            self.debug("Setting sigma to %s" % toReturn[0])
            return toReturn
            
    def adjustSigma(self, steps, index, sigma, sigmaRatio):
        subsection = steps[index - self.sigmaCalibrationLength : index]
        desiredAvg = 1 / self.desiredAcceptRatio
        actualAvg = np.average(subsection[:, 1])
        n = steps[0,3:].size
        update = 1 + 2*(((desiredAvg/actualAvg) - 1)/(n))
        #update = min(1/0.7, max(0.7, update))
        ps = sigmaRatio
        sigmaRatio *= update
        self.debug("Adjusting sigma: want accept every %0.2f, got %0.2f. Updating ratio from %0.3f to %0.3f" % (desiredAvg, actualAvg, ps, sigmaRatio))
        
        
        self.debug("Adjusting sigma: setting sigma to %s" % (sigma))
        return sigma, sigmaRatio
        
    def adjustCovariance(self, steps, index, sigmaRatio):
        if index == 1 or True:
            sigmaRatio = 0.2
        weightedAvgs, covariance = self.getCovariance(steps, index * self.calibrationLength)
        evals, evecs = self.diagonalise(covariance)
        sigma = np.sqrt(np.abs(evals)) * 2.3 / np.sqrt(evals.size)
        self.debug("[Covariance] sigma: setting sigma to %s" % (sigma))
        return sigma, sigmaRatio, evecs

    def getCovariance(self, steps, index):
        self.debug("Getting covariance")
        subset = steps[np.floor(index/2):index, :]
        weightedAvg = np.sum(subset[:, 3:] * subset[:, 1].reshape((subset[:,1].size, 1)), axis=0) / np.sum(subset[:,1])
        deviations = subset[:, 3:] - weightedAvg
        n = deviations[0,:].size
        covariance = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                covariance[i,j] = np.sum(deviations[:, i] * deviations[:, j] * subset[:, 1])
        covariance /= np.sum(subset[:,1])
        
        #self.debug("Found covariance: \n%s" % covariance)
        return weightedAvg, covariance
        
    def diagonalise(self, covariance):
        evals, evecs = np.linalg.eig(covariance)
        #self.debug("Diagonlised eigenvectors: \n%s" % evecs)
        return (evals, evecs)
        
    def getInitialParams(self):
        return [np.random.uniform(x[1], x[2]) for x in self.fitter.getParams()]
        
    
                
                
                
            
if __name__ == '__main__':
   
    
    finalFitter = ThresholdFitter()
    text = "thresholdMCMC"
       
    cambMCMCManager = CambMCMCManager(text, finalFitter, debug=False)
    cambMCMCManager.configureMCMC(numCalibrations=9,calibrationLength=500, thinning=1, maxSteps=50000)
    cambMCMCManager.configureSaving(stepsPerSave=5000)
    args = sys.argv
    walk = None
    #walk = 5
    outputToFile = False
    if len(args) > 1:
        try:
            walk = int(args[1])
            outputToFile = False
        except Exception:
            print("Argument %s is not a number" % args[1])
    
    if walk is not None:
        for i in range(walk):
            cambMCMCManager.doWalk(i, outputToFile=outputToFile)
    else:
        cambMCMCManager.consolidateData()
        cambMCMCManager.getParameterBounds()
        #cambMCMCManager.testConvergence()
        #cambMCMCManager.plotResults(filename=text)   6
        cambMCMCManager.plotResults()
        #cambMCMCManager.plotResults(filename=text+'gkde', useGaussianKernelDensityEstimation=True)   
        #cambMCMC.getHistogram("omch2")
        #cambMCMC.getHistogram("sigmav")
        #cambMCMC.getHistogram("alpha")
        #cambMCMC.getHistogram("b2")
        #cambMCMC.getHistogram("beta")
        #cambMCMC.getHistogram("lorentzian")
        cambMCMCManager.plotWalk("sub", "pow", final=True)
        #cambMCMCManager.plotWalk("x4", "x3", final=True)
        #cambMCMCManager.plotWalk("x2", "x1", final=True)
        #cambMCMCManager.plotWalk("sub", "pow", final=False)
        #cambMCMCManager.plotWalk("x4", "x3", final=False)
        #cambMCMCManager.plotWalk("x2", "x1", final=False)
        #cambMCMC.plotWalk("omch2", "alpha", final=True)
        #cambMCMC.getHistogram("b")
        #cambMCMC.getHistogram("c")
        #cambMCMC.getHistogram("d")
        #cambMCMC.plotWalk("a", "b", final=True)
        #cambMCMC.plotWalk("om", "sigvs", final=False)
        #cambMCMC.plotWalk("b2s", "betas", final=False)