import os
import sys
import cv2
from vocxml import VOCObject, VOCAnnotation
import toolkit
import pickle
import random
import numpy as np
import pdb


def get_norm_gradientK(gray):
    dx = cv2.Sobel(gray,cv2.CV_32F,1,0)
    dy = cv2.Sobel(gray,cv2.CV_32F,0,1)
    dx = np.absolute(dx)
    dy = np.absolute(dy)
    dx = np.minimum(dx,255)
    dy = np.minimum(dy,255)
    grad = np.maximum(dx,dy)
    return grad


def get_norm_gradient(img):
    grad = get_norm_gradientK(img[:,:,0])
    grad = np.maximum(grad,get_norm_gradientK(img[:,:,1]))
    grad = np.maximum(grad,get_norm_gradientK(img[:,:,2]))
#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    grad = np.maximum(grad,get_norm_gradientK(gray))
#    grad = get_norm_gradientK(gray)

    return [grad]


def get_positive(rootdir, trainclasses, outdir):
    xmldir = rootdir+'Annotations/'
    jpgdir = rootdir+'JPEGImages/'
    xmls = toolkit.scanfor(xmldir, '.xml')
    ann = VOCAnnotation()
    total = len(xmls)
    num = 0
    posnum = 0
    for sname,fname in xmls:
        num += 1
        ann.load(fname)
        img = cv2.imread(jpgdir+sname+'.jpg',1)
        grads = get_norm_gradient(img)
        allfeat = []
        for obj in ann.objects:
            if obj.name in trainclasses:
                for grad in grads:
                    
                    objw = obj.xmax - obj.xmin
                    objh = obj.ymax - obj.ymin
                    if objw * objh < 100:
                        continue 


                    w0 = int(objw / 2)
                    w1 = int(objw * 1.5)
                    h0 = int(objh / 2)
                    h1 = int(objh * 1.5)
                    x0 = obj.xmin
                    y0 = obj.ymin
                    r1 = (x0,y0, x0 + objw, y0 + objh)

                    if 1:
                        hs = (h1 - h0) / 4
                        ws = (w1 - w0) / 4
                        if hs < 2:
                            hs = 2
                        if ws < 2:
                            ws = 2
                        for h in range(h0,h1,hs):
                            for w in range(w0,w1,ws):
                               x1 = np.minimum(img.shape[1] - 1, x0 + w)
                               y1 = np.minimum(img.shape[0] - 1, y0 + h)
                               r2 = (x0,y0,x1,y1)
                               if toolkit.inter2union(r1,r2) <= 0.5:
                                   continue
                               feat = grad[y0:y1,x0:x1]
                               feat = cv2.resize(feat,(8,8))
                               feat.shape = (1,64)
                               if len(allfeat) < 1:
                                  allfeat = [feat]
                               else:
                                  allfeat.append(feat)       
                    else:
                        x1 = x0 + objw
                        y1 = y0 + objh
                        feat = grad[y0:y1,x0:x1]
                        feat = cv2.resize(feat,(8,8))
                        feat.shape = (1,64)
                        if len(allfeat) < 1:
                            allfeat = [feat]
                        else:
                            allfeat.append(feat)       


        posnum += len(allfeat)      
        if len(allfeat) > 0:
            with open(outdir+sname+'.pf','w') as fout:
                pickle.dump(allfeat, fout)
        if 0 == num%10:
            print '.',
        if 0 == num%500:
            print 'pos '+str(num) + '/' + str(total) + ' ' + str(posnum)
    print ''


def max_inter2union(r1, rrs):
    ovrs = np.zeros((1,len(rrs)))
    for k in range(len(rrs)):
        r2 = rrs[k]
        ovrs[0,k] = toolkit.inter2union(r1, r2)
    return ovrs.max()

def get_negative(rootdir, outdir):
    neg_per_img = 10
    xmldir = rootdir+'Annotations/'
    jpgdir = rootdir+'JPEGImages/'
    xmls = toolkit.scanfor(xmldir, '.xml')
    ann = VOCAnnotation()
    num = 0
    negnum = 0
    total = len(xmls)
    for sname,fname in xmls:
        num += 1
        ann.load(fname)
        img = cv2.imread(jpgdir+sname+'.jpg',1)
        grads = get_norm_gradient(img)
        allfeat = []
        rrs = []
        for obj in ann.objects:
            r = [obj.xmin,obj.ymin,obj.xmax,obj.ymax]
            if len(rrs) < 1:
                rrs = [r]
            else:
                rrs.append(r)
        for k in range(neg_per_img):
            x0 = random.uniform(0, img.shape[1] - 1)
            x1 = random.uniform(0, img.shape[1] - 1)
            y0 = random.uniform(0, img.shape[0] - 1)
            y1 = random.uniform(0, img.shape[0] - 1)

            x0 = int(x0)
            y0 = int(y0)
            x1 = int(x1)
            y1 = int(y1)
            if x0 > x1:
                x0,x1 = x1,x0
            if y0 > y1:
                y0,y1 = y1,y0

            if y1 - y0 < 10 or x1 - x0 < 10:
                continue

            r = [x0,y0,x1,y1]
            if max_inter2union(r, rrs) >= 0.5:
                continue

            c = random.uniform(0,len(grads))
            c = int(c)
            if c >= len(grads):
                c = len(grads) - 1
            grad = grads[c]
            feat = grad[y0:y1, x0:x1]
            feat = cv2.resize(feat, (8,8))
            feat.shape = (1,64)
            if len(allfeat) < 1:
                allfeat = [feat]
            else:
                allfeat.append(feat)

        negnum += len(allfeat)
        if len(allfeat) > 0:
            with open(outdir+sname+'.nf','w') as fout:
                pickle.dump(allfeat, fout)
        if 0 == num%10:
            print '.',
        if 0 == num%500:
            print 'neg '+str(num) + '/' + str(total) + ' ' + str(negnum)
    print ''

if __name__ == "__main__":
    rootdir = ''
    outdir = 'f1/'
    trainclass = {}
    with open('vocpath', 'r') as f:
        rootdir = f.readline().strip()
    with open('config', 'r') as f:
        tc = f.readline().strip().split(',')
    for k in tc:
        trainclass[k] = 1

    get_positive(rootdir, trainclass, outdir)
    get_negative(rootdir, outdir)
