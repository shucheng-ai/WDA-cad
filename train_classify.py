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
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score,confusion_matrix,classification_report
import os
#from imblearn.over_sampling import SMOTE, ADASYN

WDAS_NONE = 0
WDAS_WALLS = 1
WDAS_DOORS = 2
WDAS_PILLARS = 3
WDAS_CLASSES = 4    
LABEL_LOOKUP = { 'WALL': 0,  '墙': 0, '门': 1,  'DOOR':1,'WIND':3,'窗''window': 3, '柱':2, 'PILLAR':2 ,'COL':2} 

def get_description(lst):
    lst_min,lst_max  = 10e8,0
    lst_mode,lst_modect, lst_med = 0,0,0
    if len(lst) != 0:
        lst_min = min(lst)
        lst_max = max(lst)
        tmp = stats.mode(lst)
        lst_mode = tmp[0][0]
        lst_modect = tmp[1][0]
        tmp2 = np.median(lst)
        lst_med = tmp2
    return [lst_min,lst_max,lst_mode,lst_modect, lst_med]

def get_hist(lst):
    hist = [0,0,0,0,0]
    if len(lst) != 0:
        avg  = sum(lst) / len(lst)
        std = statistics.stdev(lst)
        hist_max = avg + 3*std
        hist_min = max(avg - 3*std, 0) 
        dev = (hist_max - hist_min)/5
        bin = [hist_min+ i * dev for i in range(6)]
        hist = list(np.histogram(lst,bins = bin)[0])
    return hist

def extractor_preprocess (dr, name,shapes):
    count = {'LINE': 0, 'ARC': 0, 'INSERT':0,'LWPOLYLINE':0,'CIRCLE': 0,'SHAPE':0}
    dls,arc_len,circle_szs,szs,LW_lines,LW_arcs =[],[],[],[],[],[]
    inserts_x,inserts_y,inserts_xscale,inserts_yscale,inserts_rot = [],[],[],[],[]
    for e in shapes:
        if e.dxftype() == 'LINE':
            count['LINE'] += 1
            dl = np.sqrt((e.dxf.start[0]-e.dxf.end[0])**2 + (e.dxf.start[1]- e.dxf.end[1])**2)
            dls.append(dl)    
        if e.dxftype() == 'ARC':
            count['ARC'] += 1
            angle = (e.dxf.start_angle > e.dxf.end_angle) * 360 + (e.dxf.end_angle -e.dxf.start_angle)
            arc_len.append(angle/180 * e.dxf.radius * 3.14)
        if e.dxftype() == 'CIRCLE':
            count['CIRCLE'] += 1
            szs.append(3.14 * e.dxf.radius * e.dxf.radius)
        if e.dxftype() == 'SHAPE':
            szs.append(e.dxf.size)
            count['SHAPE'] += 1
            sz = e.dxf.size
        if e.dxftype() == 'LWPOLYLINE':
            count['LWPOLYLINE'] += 1
            closed = e.closed
            pts = e.get_points()
            points = []
            points.extend(pts)
            if closed:
                points.append((points[0][0], points[0][1], 0, 0, 0))
            for i in range(0, len(points) - 1):
                x, y, _, _, b = points[i]
                x2, y2 = points[i + 1][:2]
                if b == 0: 
                    LW_lines.append(np.sqrt((x2-x)**2 + (y2-y)**2))
                    continue
                else:
                    start_point = ezdxf.math.Vector((x, y)).vec2 
                    end_point = ezdxf.math.Vector((x2, y2)).vec2 
                    center = ezdxf.math.bulge_center(start_point, end_point, b)
                    radius = ezdxf.math.bulge_radius(start_point, end_point, b)
                    start_angle = dxf.cal_angle(center, start_point)
                    end_angle = dxf.cal_angle(center, end_point)
                    if b > 0 and start_angle > end_angle:
                        start_angle -= 360
                    if b < 0:
                        start_angle, end_angle = end_angle, start_angle
                        if start_angle > end_angle:
                            start_angle -= 360
                    angle = end_angle - start_angle
                    LW_arcs.append(angle/180 * radius * 3.14)
        if e.dxftype() == 'INSERT':
            count['INSERT'] += 1
            inserts_x.append(e.dxf.insert[0])
            inserts_y.append(e.dxf.insert[1])
            inserts_xscale.append(e.dxf.xscale)
            inserts_yscale.append(e.dxf.yscale)
            inserts_rot.append(e.dxf.rotation) 
    #print(count)
    entity_count = [count['LINE'],count['ARC'],count['CIRCLE'],count['LWPOLYLINE'],count['SHAPE'],count['INSERT']]
    lines_stats = list(itertools.chain(dls,LW_lines))
    arc_stats = list(itertools.chain(arc_len,LW_arcs))
    size_stats = list(itertools.chain(circle_szs,szs))
    insert_stats = [inserts_x,inserts_y,inserts_xscale,inserts_yscale,inserts_rot]
    return entity_count,lines_stats,arc_stats,size_stats,insert_stats

def extractor_name (dr, name, shapes):
    name_ft = [0, 0, 0,0]
    for key, value in LABEL_LOOKUP.items():
        if key in name.upper():
            name_ft[LABEL_LOOKUP[key]] += 1
    return name_ft

def extractor_insert (insert_stats):
    insert_par = []
    for i in range(len(insert_stats)):
        insert_par.extend(get_description(insert_stats[i]))
    return insert_par

#Extract features for machine learning
def extract_features (dr,name,shapes):
    entity_count,lines_stats,arc_stats,size_stats,insert_stats = extractor_preprocess (dr,name,shapes)
    name_ft = extractor_name(dr, name, shapes)
    line_par = get_description(lines_stats)
    arc_par = get_description(arc_stats)
    size_par = get_description(size_stats)
    insert_par = extractor_insert (insert_stats)
    features = list(itertools.chain(name_ft,entity_count,line_par,arc_par,size_par,insert_par))
    return list(itertools.chain(name_ft,entity_count,line_par,arc_par,size_par,insert_par))

def get_entity(shape):
    count = {'LINE': 0, 'ARC': 0, 'INSERT':0,'LWPOLYLINE':0,'CIRCLE' : 0}
    hist_shape = []
    for shp in shape:
        if shp.dxftype() in count.keys():
            count[shp.dxftype()] += 1
        handler = dxf.SHAPE_HANDLERS.get(type(shp), None)
        #print(handler)
        if handler is not None:
            #print(handler.tostr(shp))
            hist_shape.append(handler.tostr(shp))
    #print ("count:")
    #print (count)
    #print ("hist shape:")
    #print (hist_shape)
    return hist_shape

def match_layers (enty_tmpl, WDAS, i,name, shape_hist):
    #print('match_layers')
    name_dict = {
    "NONE": ['#$%%^&^&%'],
    "WALLS": ['墙', 'WALL'],
    "DOORS": ['门', 'DOOR','WIND','窗'], 
    "PILLARS": ['柱', 'PILLAR','COL'] }  # 
    
    #Match Name
    score = 0
    lst = name_dict[WDAS[5:]]
    for j in range(len(lst)):
        if str(lst[j]) in name.upper():
            #print('name fit' )
            score += 30
    
    for shp in shape_hist:
        for tpl in enty_tmpl:
            if shp == tpl:
                score += 1
    return score

def prepare_sample (dr):
    print('#####################prepare_sample#####################')
    tmpls = [[] for _ in range(WDAS_CLASSES)]
    ft_tmpl = [[] for _ in range(WDAS_CLASSES)]
    enty_tmpl = [[] for _ in range(WDAS_CLASSES)]
    for name, shapes in dr.layers.items():
        if name == 'WDAS_WALLS':
            print(name)
            tmpls[WDAS_WALLS] = shapes
            ft_tmpl[WDAS_WALLS] = extract_features(dr,name,shapes)
            enty_tmpl[WDAS_WALLS] = get_entity(shapes)

        elif name == 'WDAS_DOORS':
            print(name)
            tmpls[WDAS_DOORS] = shapes
            ft_tmpl[WDAS_DOORS] = extract_features(dr,name,shapes)
            enty_tmpl[WDAS_DOORS] = get_entity(shapes)

        elif name == 'WDAS_PILLARS':
            print(name)
            tmpls[WDAS_PILLARS] = shapes
            ft_tmpl[WDAS_PILLARS] = extract_features(dr,name,shapes)
            enty_tmpl[WDAS_PILLARS] = get_entity(shapes)
        pass
    ft_tmpl[WDAS_NONE] = extract_features(dr,name,tmpls[WDAS_NONE])

    layers = []
    for name, shapes in dr.layers.items():

        if name[:3] == 'WDA':   
            continu = WDAS_NONE
        
        ft = extract_features(dr,name,shapes)
        WDAS = ['WDAS_NONE','WDAS_WALLS', 'WDAS_DOORS', 'WDAS_PILLARS']
        score = [0,0,0,0]
        shape_hist = get_entity(shapes)
        for i, tmpl in enumerate(tmpls):
            #if match_layers(tmpl,i,name,shapes):
            #pass
            score[i] = match_layers(enty_tmpl[i],WDAS[i],i,name,shape_hist)
        print('score = ' + str(score))        
        label = score.index(max(score))
        print(name + ' has score : ' + str(score))
        #ft = extract_features(dr,name,shapes)
        layer = [name, label, ft, shapes]
        layers.append(layer)
        pass
    #print(ft_tmpl)
    return layers


def train_model(paths, model_path):
    print('#####################train_model######################')
    X = []
    Y = []
    names = []
    for path in paths:
        print(path)
        dr = dxf.Drawing(path)
        layers = prepare_sample(dr)
        for name, label, ft, _ in layers:
            X.append(ft)
            Y.append(label)
            names.append((name,label))
            pass
        pass
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int)
    print ("###SHAPE###")
    print (X.shape)
    print (Y.shape)

    #X_resampled, y_resampled = SMOTE().fit_resample(X, Y)
    #print(sorted(Counter(y_resampled).items()))
    X_resampled, y_resampled = X, Y


    loo = LeaveOneOut()
    loo.get_n_splits(X_resampled)
    accuracy = []
    y_pred = np.zeros((len(Y),1)).flatten()
    y_pred_resampled = np.zeros((len(y_resampled),1)).flatten()

    for train_index, test_index in loo.split(X_resampled):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_resampled[train_index], X_resampled[test_index]
        y_train, y_test = y_resampled[train_index], y_resampled[test_index]
        #print(X_train, X_test, y_train, y_test)
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        clf.fit(X_train,y_train)
        y_pred_resampled[test_index] =  clf.predict(X_test)

    print ("*"*30)
    print(names)
    #print(Y)    
    #print(y_pred)

    target_names = ['None', 'Walls', 'Doors','Pillars']
    print('report for resampled data')
    #print('confusion matrix')
    print(confusion_matrix(y_resampled, y_pred_resampled))
    print(classification_report(y_resampled, y_pred_resampled, target_names=target_names))
    
    #############################################################################
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X)

    count,cor = 0,0
    for i in range(len(Y)):
        if (Y[i] != 0):
            count += 1
            cor += 1* (y_pred[i] == Y[i])
    print(cor/ count)
    acc = np.mean(np.asarray(Y)== y_pred)
    recall = recall_score(Y, y_pred,average='macro')
    print('accuracy = ' + str(acc) + ', recall = ' + str(recall))
    #print('confusion matrix')
    
    print('report for actual data')
    print(confusion_matrix(Y, y_pred))
    print(classification_report(Y, y_pred, target_names=target_names))

    #model = None
    with open(os.path.join(model_path,'model.pkl'),'wb') as f:
        pickle.dump(model,f)
    return model
