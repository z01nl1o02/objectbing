import os,sys
import pickle
import pdb
import numpy as np
from sklearn import svm
import toolkit as tk
import cv2
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
    samples = []
    szdict = {}
    for sname,fname in posfiles:
        with open(fname, 'r') as fin:
            feats,sizes = pickle.load(fin)
        for k in range(len(feats)):
            sz = tuple(sizes[k])
            feat = feats[k]
            posnum += 1
            samples.append(feat)
            if sz in szdict:
                szdict[sz] += 1
            else:
                szdict[sz] = 1
            if posnum % 100 == 0:
                print '.',
            if posnum % 5000 == 0:
                print 'pos '+str(posnum)
    print ' '

    negnum = 0
    for sname, fname in negfiles:
        with open(fname, 'r') as fin:
            feats = pickle.load(fin)
        for feat in feats:
            negnum+=1
            samples.append(feat)
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

    spl = np.zeros((len(samples), samples[0].shape[1]))
    for k in range(spl.shape[0]):
        spl[k,:] = samples[k]

    for key in szdict.keys():
        print key, szdict[key]    
    print 'samples ' + str(spl.shape[0]) + 'x'+ str(spl.shape[1])
    return labels,spl,szdict,

def traindetector(labels, samples):
    #detector = svm.SVC(C=1000.0,verbose=1,max_iter=500000,kernel='linear')
    with open('samples.dump', 'w') as fout:
        pickle.dump((labels,samples),fout)
    detector = svm.LinearSVC(C=10.0,verbose=1,max_iter=5000,dual=False,tol=1e-6,penalty='l1') #1R_2LOSS
    minv = np.min(samples,0)
    maxv = np.max(samples,0)
#    samples = normsamples(samples,minv, maxv)
    detector.fit(samples,labels)
    return minv,maxv,detector

def testdetector(minv,maxv,detector,labels, samples):
#    samples = normsamples(samples,minv,maxv)
    prd = detector.predict(samples)
    hitnum = np.sum(prd == labels)
    poshit = 0
    neghit = 0
    for k in range(len(labels)):
        if prd[k] == labels[k] and labels[k] > 0:
            poshit += 1
        elif prd[k] == labels[k] and labels[k] < 0:
            neghit += 1
    print 'test: ' + str(hitnum) + '/' + str(labels.shape[0]) + ' ' + str(hitnum * 1.0 / labels.shape[0]) + ' ' + str(poshit) + '_'+str(neghit)

def train_stage_1():
    featdir = 'f1/'
    detectorpath = 'detector.txt'
    if 0:
        with open('samples.dump','r') as fin:
            labels, samples = pickle.load(fin)
        if 0:
            for k in range(0,samples.shape[0],1000):
                feat = samples[k,:]
                feat = np.reshape(feat,(8,8))
                outfile = 'dbg/'+str(k) + '_'+str(labels[k])+'.txt'
                with open(outfile, 'w') as f:
                    for y in range(8):
                        line = ""
                        for x in range(8):
                            z = int(feat[y,x])
                            z = '%3d '%z
                            line = line + z
                        line = line + '\n'
                        f.write(line)
    else:
        labels, samples,szdict = loadsamples(featdir)
    minv,maxv,detector = traindetector(labels, samples)
    testdetector(minv,maxv, detector, labels, samples)
    with open(detectorpath, 'w') as fout:
        pickle.dump((minv,maxv,szdict,detector), fout)

def train_stage_2():
    if 0: #collect samples for fast train
        resultpath = 'f2collect.txt'
        featdir = 'f2/'
        poss = []
        negs = []
        names = tk.scanfor(featdir, '.f2')
        for sname, fname in names:
            with open(fname, 'r') as f:
                cands = pickle.load(f)
            if len(cands) > 0:
                for cand in cands:
                    if cand[5] > 0.5:
                        poss.append(cand)
                    else:
                        negs.append(cand)
            print sname + ' ' + str(len(poss)) + ':' + str(len(negs))
        with open(resultpath, 'w') as f:
            pickle.dump((poss,negs), f)
    else:
        featpath = 'f2collect.txt'
        detectorname = 'detector_stage2'
        with open(featpath, 'r') as f:
            poss,negs = pickle.load(f)
        poss_sz = {}
        negs_sz = {}
        for item in poss:
            sz = (item[2],item[3])
            if sz in poss_sz:
                poss_sz[sz].append(item)
            else:
                poss_sz[sz] = [item]

        for item in negs:
            sz = (item[2],item[3])
            if sz in negs_sz:
                negs_sz[sz].append(item)
            else:
                negs_sz[sz] = [item]

        detectors = {}
        for key in poss_sz.keys():
            if key in negs_sz.keys():
                print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
                samples = None
                labels = []
                for item in poss_sz[key]:
                    labels.append(1)
                    t = np.array(item[4])
                    t = np.reshape(t, (1,1))
                    if samples is None:
                        samples = t
                    else:
                        samples = np.vstack((samples, t))
                numneg = 0
                numpos = samples.shape[0] 
                for item in negs_sz[key]:
                    labels.append(-1)
                    t = np.array(item[4])
                    t = np.reshape(t, (1,1))
                    samples = np.vstack((samples, t))
                    numneg += 1
#using all negative samples !!!!!

                labels = np.array(labels)
                print 'train : ', key, ' ', numpos, 'x', numneg
                minv,maxv, detector = traindetector(labels,samples)
                print 'verify...'
                testdetector(minv, maxv, detector, labels,samples)
                detectors[key] = detector
        with open(detectorname+'.txt', 'w') as f:
            pickle.dump(detectors, f)

if __name__ == "__main__":
    if 0 == cmp('stage1', sys.argv[1]):
        train_stage_1()
    elif 0 == cmp('stage2', sys.argv[2]):
        train_stage_2()
    else:
        print 'stage1 or stage2'


