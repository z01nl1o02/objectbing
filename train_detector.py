import os,sys
import pickle
import pdb
import numpy as np
from sklearn import svm
import toolkit as tk

def normsamples(samples, minv, maxv):
    m0 = np.tile(minv, (samples.shape[0],1))
    m1 = np.tile(maxv, (samples.shape[0],1))
    samples = (samples - m0) / (m1 - m0)
    idx = samples > 1
    samples[idx] = 1
    idx = samples < 0
    samples[idx] = 0
    return samples
     
def loadsamples(indir):
    posfiles = tk.scanfor(indir, '.pf')
    negfiles = tk.scanfor(indir, '.nf')
    samples = []
    labels = []
    posnum = 0
    total = len(posfiles)
    for sname,fname in posfiles:
        posnum += 1
        with open(fname, 'r') as fin:
            feat = pickle.load(fin)
        feat = feat[0]
        if posnum == 1:
            samples = feat
        else:
            samples = np.vstack((samples, feat))
        if posnum % 10 == 0:
            print '.',
        if posnum % 500 == 0:
            print 'pos '+str(posnum)+'/'+str(total)
    print ' '

    total = len(negfiles)
    negnum = 0
    for sname, fname in negfiles:
        negnum += 1
        with open(fname, 'r') as fin:
            feat = pickle.load(fin)
        feat = feat[0]
        samples = np.vstack((samples, feat))
        if negnum % 10 == 0:
            print '.',
        if 0 == negnum % 500:
            print 'neg '+str(negnum)+'/'+str(total)
    print ' '

    print 'pos : neg = ' + str(posnum) + ' : ' + str(negnum)
    pl = [1 for k in range(posnum)]
    nl = [-1 for k in range(negnum)]
    labels = pl + nl
    labels = np.array(labels)

    print 'samples ' + str(samples.shape[0]) + 'x'+ str(samples.shape[1])
    return labels,samples

def traindetector(labels, samples):
    detector = svm.LinearSVC(C=1.0,verbose=1)
    minv = np.min(samples,0)
    maxv = np.max(samples,0)
    samples = normsamples(samples,minv, maxv)
    detector.fit(samples,labels)
    return minv,maxv,detector

def testdetector(minv,maxv,detector,labels, samples):
    samples = normsamples(samples,minv,maxv)
    prd = detector.predict(samples)
    hitnum = np.sum(prd == labels)
    print 'test: ' + str(hitnum) + '/' + str(labels.shape[0]) + ' ' + str(hitnum * 1.0 / labels.shape[0]) 

if __name__=="__main__":
    featdir = 'f1/'
    detectorpath = 'detector.txt'
    labels, samples = loadsamples(featdir)
    minv,maxv,detector = traindetector(labels, samples)
    testdetector(minv,maxv, detector, labels, samples)
    with open(detectorpath, 'w') as fout:
        pickle.dump((minv,maxv,detector), fout)

