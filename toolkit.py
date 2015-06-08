import os,sys
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
