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
    for sname,fname in posfiles:
        with open(fname, 'r') as fin:
            feats = pickle.load(fin)
        for feat in feats: 
            posnum += 1
            if posnum == 1:
                samples = feat
            else:
                samples = np.vstack((samples, feat))
            if posnum % 100 == 0:
                print '.',
            if posnum % 5000 == 0:
                print 'pos '+str(posnum)
    print ' '

    skip = 0
    negnum = 0
    for sname, fname in negfiles:
        with open(fname, 'r') as fin:
            feats = pickle.load(fin)
        for feat in feats:
            skip += 1
       #     if 0 != skip % 5:
       #         continue
            negnum+=1
            samples = np.vstack((samples, feat))
            if negnum % 100 == 0:
                print '.',
            if 0 == negnum % 5000:
                print 'neg '+str(negnum)
    print ' '

    print 'pos : neg = ' + str(posnum) + ' : ' + str(negnum)
    pl = [1 for k in range(posnum)]
    nl = [-1 for k in range(negnum)]
    labels = pl + nl
    labels = np.array(labels)

    print 'samples ' + str(samples.shape[0]) + 'x'+ str(samples.shape[1])
    return labels,samples

def traindetector(labels, samples):
    #detector = svm.SVC(C=1000.0,verbose=1,max_iter=500000,kernel='linear')
    with open('samples.dump', 'w') as fout:
        pickle.dump((labels,samples),fout)
    detector = svm.LinearSVC(C=1000,verbose=1,max_iter=300000,dual=False,tol=1e-6)
    minv = np.min(samples,0)
    maxv = np.max(samples,0)
    samples = normsamples(samples,minv, maxv)
    detector.fit(samples,labels)
    return minv,maxv,detector

def testdetector(minv,maxv,detector,labels, samples):
    samples = normsamples(samples,minv,maxv)
    prd = detector.predict(samples)
    hitnum = np.sum(prd == labels)
    poshit = 0
    for k in range(len(labels)):
        if prd[k] == labels[k] and labels[k] > 0:
            poshit += 1
    print 'test: ' + str(hitnum) + '/' + str(labels.shape[0]) + ' ' + str(hitnum * 1.0 / labels.shape[0]) + ' ' + str(poshit)

if __name__=="__main__":
    featdir = 'f1/'
    detectorpath = 'detector.txt'
    if 0:
        with open('samples.dump','r') as fin:
            labels, samples = pickle.load(fin)
    else:
        labels, samples = loadsamples(featdir)
    minv,maxv,detector = traindetector(labels, samples)
    testdetector(minv,maxv, detector, labels, samples)
    with open(detectorpath, 'w') as fout:
        pickle.dump((minv,maxv,detector), fout)

