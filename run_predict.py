import os, sys
import cv2
import numpy as np
import pickle
from sklearn import svm
import toolkit as tk
from get_trainset import get_norm_gradient, max_inter2union
from train_detector import normsamples
import pdb
import multiprocessing as mp
from vocxml import VOCObject, VOCAnnotation

def cmp_score(a,b):
    if a[4] < b[4]:
        return 1
    elif a[4] > b[4]:
        return -1
    else:
        return 0

def cmp_pts_score(a,b):
    if a[2] < b[2]:
        return 1
    elif a[2] > b[2]:
        return -1
    else:
        return 0

def nms_single_size(candpts, max_num, radius, imgsize):
    #nms for single size

    #get local maxima
    smap = np.zeros(imgsize)
    for k in range(len(candpts)):
        x,y,s = candpts[k]
        smap[y,x] = s
    smap_blur = cv2.blur(smap,(3,3))
    pts = []
    for y in range(smap.shape[0]):
        for x in range(smap.shape[1]):
            if smap[y,x] >= smap_blur[y,x]:
                if len(pts) < 1:
                    pts = [[x,y,smap[y,x]]]
                else:
                    pts.append( [x,y,smap[y,x]] )
    pts.sort(cmp_pts_score)

    #nms
    flagmap = np.ones(imgsize)
    result = []
    for k in range(len(pts)):
        cx,cy,s = pts[k]
        if flagmap[cy,cx] > 0:
            if len(result) < 1:
                result = [ [cx,cy,s] ]
            else:
                result.append( [ cx,cy,s] )
        else:
            continue

        for dy in range(-radius, radius, 1):
            for dx in range(-radius, radius, 1):
                x = cx + dx
                y = cy + dy
                if x < 0 or x >= imgsize[1] or y < 0 or y >= imgsize[0]:
                    continue
                flagmap[y,x] = -1 

    if len(result) > max_num:
        result = result[0:max_num]
    return result

     
def nms(cands, max_num):

#    with open('nms.input.dump', 'w') as fin:
#        pickle.dump(cands,fin)

    cands.sort(cmp_score)
    result = []
    for k in range(len(cands)-1,-1,-1):
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
    result.sort(cmp_score)
    if max_num < len(result):
        result = result[0:num] 
    return result

def predict_for_single_image(imgpath, xmlpath, outpath, minv, maxv, detector, slient_mode=0):
    num_per_sz = 100
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
            if 0 == slient_mode:
                print str(blkh) + 'x' + str(blkw) + ' ' + str(scaley) + 'x' + str(scalex) + ' ',
            resizeds = []
            for grad in grads:
                grad = cv2.resize(grad, dsize)
                if len(resizeds) < 1:
                    resizeds = [grad]
                else:
                    resizeds.append(grad)
            res = []
            for y in range(resizeds[0].shape[0] - 8):
                for x in range(resizeds[0].shape[1] - 8):
                    for resized in resizeds:
                        feat = resized[y:y+8, x:x+8]
                        feat = np.reshape(feat,(1,64))
                        if len(res) < 1:
                            samples = feat
                            #res = [[x / scalex, y / scaley, blkw, blkh]]
                            res = [[x,y,0]]
                        else:
                            samples = np.vstack((samples, feat))
                            #res.append([x / scalex, y / scaley, blkw, blkh])
                            res.append([x,y,0])

            if len(res) > 0:
#                samples = normsamples(samples, minv,maxv)
                scores = detector.decision_function(samples)
                for k in range(len(scores)):
                    res[k][2] = scores[k]
                if 0 == slient_mode:
                    print str(len(res)) + ' ',
                res = nms_single_size(res,num_per_sz,2,(resizeds[0].shape[0], resizeds[0].shape[1]))
                cands = []
                for x,y,s in res:
                    c = [x/scalex, y/scaley, blkw, blkh, s]
                    if len(cands) < 1:
                        cands = [c]
                    else:
                        cands.append(c) 
                if len(result) < 1:
                    result = cands
                else:
                    result.extend(cands)
            if 0 == slient_mode:
                print str(len(cands))
    if 0 == slient_mode:
        print "# of objects " + str(len(result))
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
    else:
        ann = VOCAnnotation()
        ann.load(xmlpath)
        rrs = []
        for obj in ann.objects:
            rrs.append([obj.xmin, obj.ymin, obj.xmax, obj.ymax])
        pos = 0
        with open(outpath, 'w') as f:
            for k in range(len(result)):
                c = result[k]
                r = c[0:4]
                s = c[4]
                ovr = max_inter2union(r, rrs)
                if ovr >= 0.5:
                    l = 1
                else:
                    l = 0
                pos += l
                result[k].append(l)
            pickle.dump(result, f)
        with open(outpath+'['+str(pos)+','+str(len(result) - pos)+'].stat','w') as f:
            f.write(str(pos) + ":" + str(len(result)))

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

def mp_run(imgpath,xmlpath,outpath):
    with open('detector.txt','r') as fin:
        minv,maxv,detector = pickle.load(fin)
    predict_for_single_image(imgpath, xmlpath, outpath,minv, maxv, detector,1) 

def run_with_mp(vocdir,outdir,cpunum):
    pool = mp.Pool(processes=cpunum)
    xmldir = vocdir+'Annotations/'
    jpgdir = vocdir+'JPEGImages/'
    xmls = tk.scanfor(xmldir, '.xml')
    if 0:
        for sname,fname in xmls:
            mp_run(jpgdir+sname+'.jpg', fname, outdir+sname+'.f2')
    else:
        for sname,fname in xmls:
            print sname
            pool.apply_async(mp_run,(jpgdir+sname+'.jpg', fname, outdir+sname+'.f2'))
        pool.close()
        pool.join()

if __name__ == "__main__":
    with open('vocpath','r') as fin:
        vocpath = fin.readline().strip()
    #imgpath = vocpath + "JPEGImages/000369.jpg"
    imgpath = vocpath + "JPEGImages/000753.jpg"
    mode = 2
    if mode == 1:
        with open('detector.txt','r') as fin:
            minv,maxv,detector = pickle.load(fin)
        predict_for_single_image(imgpath, minv, maxv, detector) 
    elif mode == 2:
        run_with_mp(vocpath,'f2/',3)
    else:
        run_dbg(imgpath)
