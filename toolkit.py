import os,sys

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

