import os
import sys
import pickle
import cv2
from StageIClass import StageIClass
from StageIIClass import StageIIClass
from vocxml import VOCObject, VOCAnnotation

def trainI(intpath, svmIpath):
    with open('vocpath', 'r') as f:
        for line in f:
            vocpath = line.strip()
            break
    stageI = StageIClass(vocpath)
    stageI.generate_trainset(intpath)
    stageI.do_train(intpath, svmIpath)

def trainII(intpath, svmIpath, svmIIpath):
    num_per_sz = 1000
    with open('vocpath', 'r') as f:
        for line in f:
            vocpath = line.strip()
            break
    stageI = StageIClass(vocpath)
    stageI.generate_trainset_for_stageII(svmIpath, intdir,num_per_sz)
    
    stageII = StageIIClass(vocpath) 
    stageII.do_train(intpath,svmIIpath)        

def check(svmIpath, svmIIpath, outdir):
    stageI = StageIClass(vocpath)
    stageII = StageIIClass(vocpath) 
   
    with open(svmIpath, 'r') as f:
        szdict, svmdetI = pickle.load(f)
    
    with open(svmIIpath, 'r') as f:
        svmdetII_sz = pickle.load(f) 

    annoparser = VOCAnnotation()
    filenames = stageI.load_testset_list()
    for sname, jpgname, xmlname in filenames:
        annoparser.load(xmlname)
        img = cv2.imread(jpgname,1)
        result = stageI.predict(img, szdict, svmdetI, 200)
        sample_sz = {}
        for rect, score in result:
            w = rect[2] - rect[0]
            h = rect[3] - rect[1]
            sz = (w,h)
            if sz in sample_sz:
                sample_sz[sz].append([rect, score])
            else:
                sample_sz[sz] = [[rect, score]]
        result = stateII.predict(sample_sz, svmdetII_sz, 100)

        #evaluate
        objrects = []
        for obj in annoparser.objects:
            r = [obj.xmin, obj.ymin, obj.xmax, obj.ymax]
            objrects.append(r)

        for rect, score in result:
            if maximum_inter2union(rect, objrects) > 0.5:
                cv2.rectangle(img, (rect[0],rect[1]), (rect[2],rect[3]), (0,0,255), 2)
        outpath = outdir + sname + '.jpg'
        cv2.imwrite(outpath, img)

if __name__ == "__main__":

    intpath = 'tmp/'
    svmI = 'svmI.txt'
    svmII = 'svmII.txt'
    resultpath = 'result/'

    if len(sys.argv) == 2:
        if 0 == cmp(sys.argv[1], '-trainI'):
            trainI(intpath, svmI)    
        elif 0 == cmp(sys.argv[1], '-trainII'):
            trainI(intpath, svmI,svmII)    
        elif 0 == cmp(sys.argv[1], '-check'):
            check(svmI,svmII, resultpath)
    else:
        print 'trainI/trainII/check'
