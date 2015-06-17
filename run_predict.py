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
    pts = candpts
    pts.sort(cmp_pts_score)

    #nms
    flagmap = np.ones(imgsize)
    result = []
    for k in range(len(pts)):
        cx,cy,s = pts[k]
        if flagmap[cy,cx] < 0:
            continue
        result.append( [ cx,cy,s] )

        x0 = np.maximum(0, cx - radius)
        x1 = np.minimum(imgsize[1] - 1, cx + radius)
        y0 = np.maximum(0, cy - radius)
        y1 = np.minimum(imgsize[0] - 1, cy + radius)
        flagmap[y0:y1,x0:x1] = -1

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

def predict_for_single_image(imgpath, xmlpath, outpath, minv, maxv, szdict, detector1, detector2 = None, slient_mode=0,outjpg=None):
    num_per_sz = 1000
    img = cv2.imread(imgpath,1)
    grads = get_norm_gradient(img)
    result = []
    for key in szdict.keys():
        blkw, blkh = key
        if blkw  * blkh < 1000:
            continue
        if szdict[key] < 50:
            continue

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
        samplelist = []
        for y in range(resizeds[0].shape[0] - 8):
            for x in range(resizeds[0].shape[1] - 8):
                for resized in resizeds:
                    feat = resized[y:y+8, x:x+8]
                    feat = np.reshape(feat,(1,64))
                    if len(res) < 1:
                        samplelist = [feat]
                        res = [[x,y,0]]
                    else:
                        samplelist.append(feat)
                        res.append([x,y,0])

        samples = np.zeros((len(samplelist), 64))
        for k in range(samples.shape[0]):
            samples[k,:] = samplelist[k]

        if len(res) > 0:
            scores = detector1.decision_function(samples)
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

    if not (detector2 is None):
        print 'predict2 input: ',str(len(result))
        spls = {}  
        for item in result:
            sz = (item[2],item[3])
            if sz in detector2.keys():
                if sz in spls.keys():
                    spls[sz].append(item)
                else:
                    spls[sz] = [item]
        result = []
        for key in spls.keys():
            spl = np.zeros((len(spls[key]),1))

            for k in range(len(spls[key])):
                spl[k] = spls[key][k][4]

            scores = detector2[key].decision_function(spl)
            k = 0
            for item in spls[key]:
                item[4] = scores[k]
                k += 1
                result.append(item)


        print 'predict2 output: ',str(len(result))

    if 0 == slient_mode:
        print "# of objects " + str(len(result))

        ann = VOCAnnotation()
        ann.load(xmlpath)
        rrs = []
        for obj in ann.objects:
            rrs.append([obj.xmin, obj.ymin, obj.xmax, obj.ymax])


        result.sort(cmp_score)
        num = 0
        hit = 0
        for x, y, w, h,s in result:
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            
            num += 1
            if num > 130:
                break

            r = [x,y,x+w,y+h]
            ovr = max_inter2union(r, rrs)
            if ovr > 0.5:
                hit += 1
                cv2.rectangle(img, (x,y),(x+w,y+h),(0,0,255),2)
        print "# of prd " + str(hit)
        if not (outjpg is None):
            cv2.imwrite(outjpg,img)
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
                    pos += 1
                else:
                    l = -1
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
        minv,maxv,szdict,detector = pickle.load(fin)
    predict_for_single_image(imgpath, xmlpath, outpath,minv, maxv, szdict,detector,None,1) 

def run_with_mp(vocdir,outdir,cpunum):
    pool = mp.Pool(processes=cpunum)
    xmldir = vocdir+'Annotations/'
    jpgdir = vocdir+'JPEGImages/'
    #xmls = tk.scanfor(xmldir, '.xml')
    xmls = []
    trainfile = vocdir+'ImageSets/Main/train.txt'
    with open(trainfile, 'r') as f:
        for line in f:
            sname = line.strip()
            fname = xmldir+sname+'.xml'
            xmls.append([sname,fname])
    if 0:
        for sname,fname in xmls:
            mp_run(jpgdir+sname+'.jpg', fname, outdir+sname+'.f2')
    else:
        for sname,fname in xmls:
            pool.apply_async(mp_run,(jpgdir+sname+'.jpg', fname, outdir+sname+'.f2'))
        print 'all are sent'
        pool.close()
        pool.join()

if __name__ == "__main__":
    with open('vocpath','r') as fin:
        vocpath = fin.readline().strip()
    #imgpath = vocpath + "JPEGImages/000369.jpg"
    imgpath = vocpath + "JPEGImages/000002.jpg"
    xmldir = vocpath+'Annotations/'
    mode = 1
    if mode == 1:
        with open('detector.txt','r') as fin:
            minv,maxv,szdict,detector = pickle.load(fin)
        with open('detector_stage2.txt','r') as f:
            detector2s = pickle.load(f)
        jpgs = tk.scanfor(vocpath+'JPEGImages/', '.jpg')
        for sname, fname in jpgs:
            predict_for_single_image(fname, xmldir+sname+'.xml', None,minv, maxv, szdict,detector,detector2s,outjpg='out/'+sname+'.jpg') 
    elif mode == 2:
        run_with_mp(vocpath,'f2/',3)
    else:
        run_dbg(imgpath)
