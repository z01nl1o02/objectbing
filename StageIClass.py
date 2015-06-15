import os,sys,pickle,cv2,math,random,pdb
import numpy as np
from vocxml import VOCObject, VOCAnnotation
from toolkit import get_norm_gradient, inter2union, maximum_inter2union, scanfor
from sklearn import svm

def cmp_3rd_item(a, b):
    if a[2] < b[2]:
        return 1
    elif a[2] > b[2]:
        return -1
    else:
        return 0

class StageIClass:
    def __init__(self, vocdir):
        self.jpgdir = vocdir + 'JPEGImages/'
        self.annotationdir = vocdir+'Annotations/'
        self.trainfilepath = vocdir+'ImageSets/Main/train.txt'

    def load_trainset_list(self):
        filenames = []
        with open(self.trainfilepath, 'r') as f:
            for line in f:
                sname = line.strip()
                jpgname = self.jpgdir + sname + '.jpg'
                xmlname = self.annotationdir + sname + '.xml'
                filenames.append([sname, jpgname, xmlname])
        return filenames

    def generate_positive(self, AnnoParser, img):
        log2 = math.log(2)
        poss = []
        for obj in AnnoParser.objects:
            objrect = [obj.xmin, obj.ymin, obj.xmax, obj.ymax]  
            w = objrect[2] - objrect[0]
            h = objrect[3] - objrect[1]
            w0 = int( math.log(w) / log2 - 0.5)
            h0 = int( math.log(h) / log2 - 0.5)
            w0 = np.maximum(w0, 16)
            h0 = np.maximum(h0, 16)
            ratio = [k for k in [1,2,3,4]]
            for w in ratio * w0:
                for h in ratio * h0:
                    r = [obj.xmin, obj.ymin, obj.xmin + w, obj.ymin + h]
                    if r[2] >= img.shape[1] or r[3] >= img.shape[0]:
                        continue
                    if inter2union(r, objrect) < 0.5:
                        continue
                    rects.append(r)
            for r in rects:
                subimg = img[r[1]:r[3], r[0]:r[2],:]
                feat = get_norm_gradient(subimg)
                poss.append([feat, r])
        return poss

    def generate_negative(self, AnnoParser, img):
        trynum = 100
        negs = []
        objrects = []
        for obj in AnnoParser.objects:
            objrect = [obj.xmin, obj.ymin, obj.xmax, obj.ymax]  
            objrects.append(objrect)

        w = img.shape[1] - 1
        h = img.shape[0] - 1 
        for k in range(trynum):
            x0 = int(random.uniform(0,w))
            x1 = int(random.uniform(0,w))
            y0 = int(random.uniform(0,h))
            y1 = int(random.uniform(0,h))

            if x0 == x1 or y0 == y1:
                continue

            if x0 > x1:
                x0,x1 = x1,x0
            if y0 > y1:
                y0,y1 = y1,y0
                
            if x1 > w or y1 > h:
                continue
            r = [x0,y0,x1,y1]   
            
            if maximum_inter2union(objrects,r) >= 0.5:
                continue

            subimg = img[y0:y1,x0:x1,:]
            feat = get_norm_gradient(subimg)
            negs.append([feat, r])

        return negs  

    def generate_trainset(self,outdir):
        filenames = self.load_trainset_list()
        AnnoParser = VOCAnnotation()
        for sname, jpgname, xmlname in filenames:
            samples = []
            img = cv2.imread(jpgname,1)
            AnnoParser.load(xmlname)
            poss = self.generate_positive(AnnoParser, img)
            negs = self.generate_negative(AnnoParser, img)
            outfilename = outdir + sname + '.sifeat'
            with open(outfilename, 'w') as f:
                pickle.dump((poss, negs), f)
    
    def do_train(self, sampledir, svmpath):
        filenames = scanfor(sampledir, '.sifeat')
        szdict = {}
        featlist = []
        labellist = []
        for sname, fname in filenames:
            with open(fname, 'r') as f:
                poss,negs = pickle.load(f)
            for feat, rect in poss:
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                sz = (w,h)
                if sz in szdict:
                    szdict[sz] += 1
                else
                    szdict[sz] = 1                         
                featlist.append(feat) 
                labellist.append(1)
                       
            for feat,rect in negs:
                featlist.append(feat)
                labellist.append(-1)
        samples = np.zeros( (len(labellist), 64))
        labels = np.array(labellist)

        for k in range(samples.shape[0]):
            samples[k,:] = featlist[k]
        posnum = np.sum(labels == 1)
        negnum = np.sum(labels == -1)
        total = labels.shape[0]
        print 'StateI: train with pos#', posnum, 'neg#', negnum, ' ', total
        svmdet = svm.LinearSVC(C=10.0,verbose=1,max_iter=5000,dual=False,tol=1e-6,penalty='l1') #1R_2LOSS
        svmdet.fit(samples,labels)
        with open(svmpath,'w') as f:
            pickle.dump((szdict, svmdet), f)

    def do_predict(self, img, szdict,svmdet, num_per_sz = 100):
        result = []
        grad = get_norm_gradient(img)
        for key in szdict.keys():
            cands = []
            blkw,blkh = key
            if szdict[key] < 50:
                continue #ignore size with few samples
            dwid = int(8.0 * img.shape[1]/ blkw)
            dhei = int(8.0 * img.shape[0]/ blkh)
            if dwid < 16 or dhei < 16:
                continue
            resized = cv2.resize(grad, (dwid,dhei))
            for y in range(dhei - 8):
                for  x in range(dwei - 8):
                    feat = resized[y:y+8,x:x+8]
                    cands.append([(x,y), feat])
            
            samples = np.zeros( (len(cands), 64))
            for k in range(samples.shape[0]):
                samples[k,:] = cands[k][1] 

            scores = svmdet.decision_function(samples)
            for k in range(len(cands)):
                cands[k].append(scores[k])


            NBS = 3            
            cands.sort(cmp_3rd_item) #sort in decressing order
            flags = np.ones((dhei, dwid))
            num = 0
            for k in range(len(cands)):
                if num >= num_per_sz:
                    break
                x,y = cands[k][0]
                s = cands[k][2]
                if flags[y,x] < 0.5:
                    continue
                x0 = x * blkw / 8.0
                y0 = y * blkh / 8.0
                result.append([[x0,y0,x0+blkw,y0+blkh], s])
                num += 1
                x0 = np.maximum(0, x - NBS)
                y0 = np.maximum(0, y - NBS)
                x1 = np.minimum(dwid-1, x + blkw)
                y1 = np.minimum(dhei-1, y + blkh)
                flags[y0:y1,x0:x1] = 0 #non-maxima-suppress
        return result 
