import os
import sys
import xml.etree.ElementTree as ET

class VOCObject:
    def __init__(self, node):
        self.name = ""
        self.pose = ""
        self.truncated = 0
        self.difficult = 0
        self.xmin,self.ymin,self.xmax,self.ymax = (0,0,0,0)

        for item in node:
            if 0 == cmp(item.tag, 'name'):
                self.name = item.text
            elif 0 == cmp(item.tag, 'pose'):
                self.pose = item.text
            elif 0 == cmp(item.tag, 'truncated'):
                self.truncated = int(item.text)
            elif 0 == cmp(item.tag, 'difficult'):
                self.difficult = int(item.text)
            elif 0 == cmp(item.tag, 'bndbox'):
                bndbox = [int(k.text) for k in item]
                self.xmin,self.ymin,self.xmax,self.ymax = bndbox

           

class VOCAnnotation:
    def __init__(self):
        self.filename = ""
        self.wid,self.hei,self.chn = (-1,-1,-1)
        self.segmented = 0
        self.objects = []
    def load(self,xmlpath):
        self.filename = ""
        self.wid,self.hei,self.chn = (-1,-1,-1)
        self.segmented = 0
        self.objects = []

        tree = ET.parse(xmlpath)
        root = tree.getroot()
        for item in root:
            if 0 == cmp(item.tag, 'filename'):
                self.filename = item.text
            elif 0 == cmp(item.tag, 'size'):
                self.wid,self.hei,self.chn = [int(k.text) for k in item]
            elif 0 == cmp(item.tag, 'segmented'):
                self.segmented = int(item.text)
            elif 0 == cmp(item.tag, 'object'):
                obj = VOCObject(item) 
                self.objects.append(obj)
   
if __name__=="__main__":
    ann = VOCAnnotation()
    vocpath = ""
    with open('vocpath', 'r')  as f:
        vocpath = f.readline().strip()
    xmlpath = vocpath + '/Annotations/000005.xml'
    ann.load(xmlpath)
    print ann.filename + ' ' + str(ann.wid) + 'x' + str(ann.hei) + 'x' + str(ann.chn) + ' ' + str(ann.segmented)
    for k in ann.objects:
        print k.name + ' ' +  k.pose + ' [' + str(k.xmin) + ' ' + str(k.ymin) + ' ' + str(k.xmax) + ' ' + str(k.ymax)+']'

