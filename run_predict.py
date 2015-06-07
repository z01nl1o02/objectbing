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

def predict_for_single_image(imgpath, minv, maxv, detector):
    img = cv2.imread(imgpath,1)
    grad = get_norm_gradient(img)
    #blksize = (10,20,40,80,160,320)
    blksize = (20,40,80,160,320)
    result = []
    for blkh in blksize:
        for blkw in blksize:
            scalex = 8.0 / blkw
            scaley = 8.0 / blkh
            dsize = (int(img.shape[1] * scalex), int(img.shape[0] * scaley))
            resized = cv2.resize(grad, dsize)
            print str(blkh) + 'x' + str(blkw) + ' ' + str(scaley) + 'x' + str(scalex)
            res = []
            for y in range(resized.shape[0] - 8):
                for x in range(resized.shape[1] - 8):
                    feat = resized[y:y+8, x:x+8]
                    feat = np.reshape(feat,(1,64))
                    if x == 0 and y == 0:
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
    result.sort(cmp_score)
    print "# of objects " + str(len(result))
    num = 0
    for x, y, w, h,s in result:
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        if s < 0.1:
            continue
        num += 1
        cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),2)
    print "# of prd " + str(num)
    cv2.imwrite('x.jpg',img)
    return result                                    


if __name__ == "__main__":
    with open('vocpath','r') as fin:
        vocpath = fin.readline().strip()
    imgpath = vocpath + "JPEGImages/000369.jpg"
    with open('detector.txt','r') as fin:
        minv,maxv,detector = pickle.load(fin)
    predict_for_single_image(imgpath, minv, maxv, detector) 

