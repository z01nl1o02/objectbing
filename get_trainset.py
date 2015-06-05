import os
import sys
import cv2
from vocxml import VOCObject, VOCAnnotation
import toolkit
import pickle
import random
import numpy as np
import pdb

def get_feature(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray,cv2.CV_32F,1,0)
    dy = cv2.Sobel(gray,cv2.CV_32F,0,1)
    grad = np.sqrt(dx * dx + dy * dy)
    grad = cv2.resize(grad, (8,8))
    grad.shape = (1,64)
    return grad

def get_positive(rootdir, trainclasses, outdir):
    xmldir = rootdir+'Annotations/'
    jpgdir = rootdir+'JPEGImages/'
    xmls = toolkit.scanfor(xmldir, '.xml')
    ann = VOCAnnotation()
    total = len(xmls)
    num = 0
    for sname,fname in xmls:
        num += 1
        ann.load(fname)
        img = cv2.imread(jpgdir+sname+'.jpg',1)
        allfeat = []
        for obj in ann.objects:
            if obj.name in trainclasses:
                subimg = img[obj.ymin:obj.ymax, obj.xmin:obj.xmax,:]
                feat = get_feature(subimg)
                if len(allfeat) < 1:
                    allfeat = [feat]
                else:
                    allfeat.append(feat)             
        if len(allfeat) > 0:
            with open(outdir+sname+'.pf','w') as fout:
                pickle.dump(allfeat, fout)
        if 0 == num%10:
            print '.',
        if 0 == num%500:
            print 'pos '+str(num) + '/' + str(total)
    print ''


def get_negative(rootdir, outdir):
    xmldir = rootdir+'Annotations/'
    jpgdir = rootdir+'JPEGImages/'
    xmls = toolkit.scanfor(xmldir, '.xml')
    ann = VOCAnnotation()
    num = 0
    total = len(xmls)
    for sname,fname in xmls:
        num += 1
        ann.load(fname)
        img = cv2.imread(jpgdir+sname+'.jpg',1)
        allfeat = []
        for obj in ann.objects:
            w = obj.xmax - obj.xmin
            h = obj.ymax - obj.ymin
            w = w / 2
            h = h / 2
            x0 = random.uniform(obj.xmin, obj.xmax)
            y0 = random.uniform(obj.ymin, obj.ymax)
            x0 = int(x0)
            y0 = int(y0)
            w = int(w)
            h = int(h)
            x1 = x0 + w
            y1 = y0 + h
            subimg = img[y0:y1, x0:x1, :]
            feat = get_feature(subimg)
            if len(allfeat) < 1:
                allfeat = [feat]
            else:
                allfeat.append(feat)
        if len(allfeat) > 0:
            with open(outdir+sname+'.nf','w') as fout:
                pickle.dump(allfeat, fout)
        if 0 == num%10:
            print '.',
        if 0 == num%500:
            print 'neg '+str(num) + '/' + str(total)
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
