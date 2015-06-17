import os,sys,cv2,pickle,pdb
from sklearn import svm
from toolkit import scanfor
import numpy as np

def cmp_2nd_item(a, b):
    if a[1] < b[1]:
        return 1
    elif a[1] > b[1]:
        return -1
    else:
        return 0



class StageIIClass:
    def __init__(self,vocdir,verbose = True):
        self.vocdir = vocdir
        self.verbose = True
        
    def do_train(self, featdir, svmpath):
        if self.verbose == True:
            print 'train in stage II start'

        filenames = scanfor(featdir, '.siifeat')
        szdict = {} #sz list is determined by samples for stage II
        num = 0
        for sname, fname in filenames:
            if self.verbose == True:
                num += 1
                if 0 == num%10:
                    print '.',
                if 0 == num%500:
                    print num, '/', len(filenames)

            with open(fname, 'r') as f:
                poss, negs = pickle.load(f)
            
            for item in poss:
                 #[rect, score]
                rect = item[0]
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                sz = (w,h)
                s = item[1]
                if sz in szdict:
                    szdict[sz][1].append(s)
                else:
                    szdict[sz] = [ [], [s]] #[neg, pos]
            
            for  item in negs:
                rect = item[0]
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                sz = (w,h)
                s = item[1]
                if sz in szdict:
                    szdict[sz][0].append(s)
                else:
                    szdict[sz] = [ [s], []] #[neg, pos]
        
        if self.verbose == True:
            print '' 

        svmdet_sz = {}
        for sz in szdict.keys():
            w,h = sz
            negnum = len(szdict[sz][0])
            posnum = len(szdict[sz][1])

            if negnum < 1 or posnum < 1:
                continue

            samples = np.zeros( (posnum+negnum, 1 ) )
            k = 0
            for s in szdict[sz][1]:
                samples[k,0] = s
                k += 1
            for s in szdict[sz][0]:
                samples[k,0] = s
                k += 1
            poslabel = [1 for k in range(posnum)]
            neglabel = [-1 for k in range(negnum)]
            labels = np.array(poslabel + neglabel)
            if self.verbose == True:
                print 'pos:neg = ', len(poslabel), ':', len(neglabel), ' size = (', str(w) + ',' + str(h) + ')'

            svmdet = svm.LinearSVC(C=100.0,verbose=1,max_iter=5000,dual=False,tol=1e-6,penalty='l1') #1R_2LOSS
            svmdet.fit(samples,labels)
            svmdet_sz[sz] = svmdet

        with open(svmpath,'w') as f:
            pickle.dump(svmdet_sz, f)

    def do_predict(self, sample_sz, svmdet_sz, num_per_sz = 100):
        founds = []
        for sz in sample_sz.keys():
            if sz in svmdet_sz:
                svmdet = svmdet_sz[sz]
                samples = np.zeros( (len(sample_sz[sz]), 1) )
                k = 0
                w = 0
                h = 0
                for item in sample_sz[sz]:
                    rect = item[0]
                    if w < rect[2]:
                        w = rect[2]

                    if h < rect[3]:
                        h = rect[3]

                    s = item[1]
                    samples[k,0] = s
                    k += 1
                scores = svmdet.decision_function(samples)

                #store score into list
                for k in range(len(scores)):
                    sample_sz[sz][k][1] = scores[k]
               
                #Non-Maxima_Suppress 
                sample_sz[sz].sort(cmp_2nd_item)
                flags = np.ones((h,w))
                result = []
                NBR = 2
                num = 0
                for item in sample_sz[sz]:
                    if num >= num_per_sz:
                        break
                    rect = item[0]
                    s = item[1]
                    x = rect[0]
                    y = rect[1]
                    if flags[y,x] < 0.5:
                        continue
                    num += 1
                    result.append( [rect,s] )

                    x0 = np.maximum(0, x - NBR) 
                    x1 = np.minimum(w - 1, x + NBR)
                    y0 = np.maximum(0, y - NBR)
                    y1 = np.minimum(h - 1, y + NBR)
                    flags[y0:y1,x0:x1] = 0
                founds.extend(result)
        return founds
