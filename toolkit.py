import os,sys,cv2
import numpy as np
def scanfor(indir, obj_ext):
    result = []
    for rdir, pdir, names in os.walk(indir):
        for name in names:
           sname,ext = os.path.splitext(name)
           if 0 == cmp(ext, obj_ext):
               fname = os.path.join(rdir,name)
               if len(result) < 1:
                   result = [[sname,fname]]
               else:
                   result.append([sname,fname])
    return result

def inter2union(r1,r2):
   ii = [0,0,0,0]
   ii[0] = np.maximum(r1[0], r2[0])
   ii[1] = np.maximum(r1[1], r2[1])
   ii[2] = np.minimum(r1[2], r2[2])
   ii[3] = np.minimum(r1[3], r2[3])

   iw = ii[2] - ii[0]
   ih = ii[3] - ii[1]
   ovr = 0
   if iw > 0 and ih > 0:
       total = (r1[2] - r1[0]) * (r1[3] - r1[1]) + (r2[2] - r2[0]) * (r2[3] - r2[1])
       ovr = iw * ih * 1.0/ (total - iw * ih)
   return ovr

def maximum_inter2union(r1, rrs):
    ovrs = np.zeros((1,len(rrs)))
    for k in range(len(rrs)):
        r2 = rrs[k]
        ovrs[0,k] = inter2union(r1, r2)
    return ovrs.max()



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
    return grad



