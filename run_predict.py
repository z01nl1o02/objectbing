import os, sys
import cv2
import numpy as np
import pickle
from sklearn import svm
import toolkit as tk
from get_trainset import get_norm_gradient
from train_detector import normsamples
import pdb

def cmp_score(a,b):
    if a[4] < b[4]:
        return 1
    elif a[4] > b[4]:
        return -1
    else:
        return 0

def nms(cands):
    with open('nms.input.dump', 'w') as fin:
        pickle.dump(cands,fin)

    thresh = 0.05
    cands.sort(cmp_score)
    result = []
    for k in range(len(cands)-1,-1,-1):
        if cands[k][4] < thresh:
            continue
        r1 = (cands[k][0], cands[k][1], cands[k][0] + cands[k][2], cands[k][1] + cands[k][3])
        dup = 0
        for j in range(k):
            r2 = (cands[j][0], cands[j][1], cands[j][0] + cands[j][2], cands[j][1] + cands[j][3])
            if tk.inter2union(r1,r2) > 0.4:
                dup = 1
                break
        if dup == 0:
            if len(result) < 1:
                result = [cands[k]]
            else:
                result.append(cands[k])
    return result

def predict_for_single_image(imgpath, minv, maxv, detector):
    img = cv2.imread(imgpath,1)
    grads = get_norm_gradient(img)
    #blksize = (10,20,40,80,160,320)
    blksize = (20,40,80,160,320)
    result = []
    for blkh in blksize:
        for blkw in blksize:
            scalex = 8.0 / blkw
            scaley = 8.0 / blkh
            dsize = (int(img.shape[1] * scalex), int(img.shape[0] * scaley))
            resizeds = []
            for grad in grads:
                grad = cv2.resize(grad, dsize)
                if len(resizeds) < 1:
                    resizeds = [grad]
                else:
                    resizeds.append(grad)
            print str(blkh) + 'x' + str(blkw) + ' ' + str(scaley) + 'x' + str(scalex)
            res = []
            for y in range(resizeds[0].shape[0] - 8):
                for x in range(resizeds[0].shape[1] - 8):
                    for resized in resizeds:
                        feat = resized[y:y+8, x:x+8]
                        feat = np.reshape(feat,(1,64))
                        if len(res) < 1:
                            samples = feat
                            res = [[x / scalex, y / scaley, blkw, blkh]]
                        else:
                            samples = np.vstack((samples, feat))
                            res.append([x / scalex, y / scaley, blkw, blkh])
            if len(res) > 0:
                samples = normsamples(samples, minv,maxv)
                scores = detector.decision_function(samples)
                idx = [k for k,a in enumerate(scores) if a > 0]
                if len(idx) > 0:
                    poswnd = []
                    for k in idx:
                        res[k].append(scores[k])
                        poswnd.append(res[k])
                    if len(result) < 1:
                        result = poswnd
                    else:
                        result.extend(poswnd)
    #result.sort(cmp_score)
    print "# of objects " + str(len(result))
    result = nms(result)
    print "# of objects " + str(len(result)) + ' after nms'
    num = 0
    for x, y, w, h,s in result:
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        num += 1
        cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),1)
    print "# of prd " + str(num)
    cv2.imwrite('x.jpg',img)
    return result                                    

def run_dbg(imgpath):
    with open('nms.input.dump', 'r') as fin:
        result = pickle.load(fin)


    img = cv2.imread(imgpath,1)
    print "# of objects " + str(len(result))
    result = nms(result)
    print "# of objects " + str(len(result)) + ' after nms'
    num = 0
    for x, y, w, h,s in result:
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        num += 1
        cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),1)
    print "# of prd " + str(num)
    cv2.imwrite('x.jpg',img)



if __name__ == "__main__":
    with open('vocpath','r') as fin:
        vocpath = fin.readline().strip()
    imgpath = vocpath + "JPEGImages/000369.jpg"
    imgpath = vocpath + "JPEGImages/000753.jpg"
    if 1:
        with open('detector.txt','r') as fin:
            minv,maxv,detector = pickle.load(fin)
        predict_for_single_image(imgpath, minv, maxv, detector) 
    else:
        run_dbg(imgpath)
