#!/usr/bin/env python3
import itertools, ezdxf
import dxf
from dxfwriter import DxfCanvas
from canvas import CvCanvas
import numpy as np
from collections import defaultdict, Counter
from scipy import stats
from glob import glob
import pickle, webbrowser
import json
#from train_classify import train_model
from train_classify import *
import os

WDAS_NONE = 0
WDAS_WALLS = 1
WDAS_DOORS = 2
WDAS_PILLARS = 3
WDAS_CLASSES = 4
LABEL_LOOKUP = { 'WALL': 0, '墙': 0, '门': 1, 'DOOR':1,'WIND':3,'窗''window': 3, '柱':2, 'PILLAR':2 ,'COL':2}

#Classify_layer 是用来输出每个layer每个class的分数
#create_html 会call process_dxf 用来给每个classify之后的layer画demo并生成html

def exact_match (tmpl,shape):
    hist_shape = []
    for shp in shape:
        #print (e_shape.dxfattribs())
        handler = dxf.SHAPE_HANDLERS.get(type(shp), None)
        if handler is not None:
            hist_shape.append(handler.tostr(shp))
    pass

def process_dxf (model, path, f, html_path):
    print('#####################process_dxf#####################')
    os.system("mkdir -p %s" % os.path.join(html_path,"demo_files"))
    dr = dxf.Drawing(path)
    a = path.split("/")
    layers = prepare_sample(dr)
    labels = {}
    for name, label, ft, shapes in layers:
        X = []
        X.append(ft)
        X = np.array(X, dtype=np.float32)
        label = model.predict(X)   
        label = np.asscalar(label)  
        labels[name] = label
        #print(labels)
        pass
   
    print(path)
    tmpls = [[] for _ in range(WDAS_CLASSES)]
    dicts = { 'WDAS_WALLS': 0,'WDAS_DOORS' : 1, 'WDAS_PILLARS' : 2}
    boxs = [0,0,0]
    pics = [0,0,0,0,0,0]
    for name, shape in dr.layers.items():
        bbox = dxf.bound_shapes(shape)
        if name[:3] != 'WDA':
            label = labels[name]
        if label > 0:
            tmpls[label].append(shape)
        pass
        if name[:3] == 'WDA':
            if name in dicts:
                print(name,bbox)
                boxs[dicts[name]] = bbox
                cvs = CvCanvas(bbox, 1024)
                dxf.render_shapes(cvs, shape)
                ref_out = 'demo_files/{}.png'.format(name +'_'+ str(a[-1][:-4]))
                pics[dicts[name]] = ref_out
                cvs.save(os.path.join(html_path, ref_out))
    for label, name in [(WDAS_WALLS, 'Wall_pred'),
                  (WDAS_DOORS, 'Door_pred'),
                  (WDAS_PILLARS, 'Pillar_pred')]:
        cvs = CvCanvas(boxs[label - 1], 1024)        
        for shape in tmpls[label]:
            dxf.render_shapes(cvs, shape)
            pred_out = 'demo_files/{}.png'.format(name+'_'+ str(a[-1][:-4]))
            pics[label + 2 ] = pred_out
            cvs.save(os.path.join(html_path, pred_out))
        pass
    pass    
    filename = a[-1]
    Wall = pics[0]
    Wall_pred = pics[3]
    Door = pics[1]
    Door_pred = pics[4]
    Pillar = pics[2]
    Pillar_pred = pics[5]

    message = """
    <tr>
    <td>{a}</td>
    <td><img src={b} alt="" height=512 width=512></img></td>
    <td><img src={c} alt="" height=512 width=512></img></td>
    <td><img src={d} alt="" height=512 width=512></img></td>
    <td><img src={e} alt="" height=512 width=512></img></td>
    <td><img src={f} alt="" height=512 width=512></img></td>
    <td><img src={g} alt="" height=512 width=512></img></td>
    </tr>
    """.format(a = filename,b = Wall,c = Wall_pred,d = Door,e = Door_pred,f = Pillar,g = Pillar_pred)
    f.write(message)

def create_html(model,dxf_path, html_path):
    f = open(os.path.join(html_path,'demo.html'),'w')
    style ="""
    <!DOCTYPE html>
    <html>
    <head>
    <meta http-equiv="content-type" content="text/html; charset=UTF-8">
    <style>
    table, th, td { border: 3px solid black;}

    </style>
    </head>
    <body>

    <h2>Result Display</h2>
    <p>Display all layers with predicted result</p>
    <table style="width:200%"> <tbody>"""

    header = """ <tr>
        <th>File</th>
        <th>WDAS_WALL</th> 
        <th>Pred_WALL</th>
        <th>WDAS_DOOR</th> 
        <th>Pred_DOOR</th>
        <th>WDAS_PILLAR</th> 
        <th>Pred_PILLAR</th>
      </tr>"""
    f.write(style)
    f.write(header)

    for path in dxf_path:
        #a = path.split("/")
        process_dxf (model, path, f, html_path)
    end = """
    <tbody>
    </table>
    </body>
    </html>"""
    f.write(end)
    f.close()

def classify_layer(dr,model):
    layers = prepare_sample(dr)
    labels = []
    for name, label, ft, shapes in layers:
        X = []
        X.append(ft)
        X = np.array(X, dtype=np.float32)
        #label = model.predict_proba(X)
        label = model.predict(X)
        label = np.asscalar(label)
        #pass
        labels.append((name, label))
    return labels

if __name__ == '__main__':

    #directory = "../beaver-test/train/*"

    #dxf_path = glob(directory + ".dxf")
    model_path = '../beaver-tools/models/'
    html_path = '../beaver-tools/html/'
    os.system("mkdir -p %s" % model_path)
    os.system("mkdir -p %s" % html_path)

    '''
    model = train_model(dxf_path,model_path)
    print ("XXXXXXXXXX")
    print(model)

    '''

    with open(os.path.join(model_path,'model.pkl'),'rb') as f:
        model = pickle.load(f)
        #print(model)

    #create_html(model,dxf_path,html_path)

    #test_path = "../beaver-test/dxf/test2.dxf"
    for test_path in glob("../beaver-test/dxf/*dxf"):
        print (test_path)
        drawing = dxf.Drawing(test_path)
        output = classify_layer(drawing,model)
        print ("classify_layer:")
        print (output)

