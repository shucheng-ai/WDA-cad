#!/usr/bin/env python3
import sys
import os
import json
import glob
import shutil
import pickle
import numpy as np
import dxf
from canvas import DxfLoadingCanvas, CvCanvas, DxfCacheCanvas
from cad import Box, Point, dump_room, CadProcessor
import time 
import base64
import shapely.geometry
from shapely.ops import unary_union
import classify
import convert


import math

HEURISTIC_NAMES = {
        # 如果没有标注，先用图层名识别
        # 如果图层名不能识别再用机器学习模型
        'WALL': 'WALLS',
        'WALLS': 'WALLS',
        'WINDOW': 'WALLS',
        'WINDOWS': 'WALLS',
        'PILLAR': 'PILLARS',
        'PILLARS': 'PILLARS',
        'COLUMN': 'PILLARS',
        'COLUMNS': 'PILLARS',
        'DOOR': 'DOORS',
        'DOORS': 'DOORS',
        'EXIT': 'DOORS',
        'EXITS': 'DOORS',
        'OBSTACLE': 'OBSTACLE',
}

CLASSES = ['', 'WALLS', 'DOORS', 'PILLARS', 'OBSTACLE']

def analyze_components (cache):
    TH = 20000
    CC_RELAX = 8000
    proc = CadProcessor()
    cvs = DxfLoadingCanvas(proc)
    cache.render(cvs)
    ccs = []
    count = 0 
    bbox = None
    for i, box in enumerate(proc.components(CC_RELAX)):
        if bbox is None:
            bbox = Box(box)
        else:
            bbox.expand(Box(box))

        x1, y1, x2, y2 = box
        ## remove small rooms
        if x2 - x1 < TH or y2 - y1 < TH:
            continue
        display = (x2 - x1 >= TH) and (y2 - y1 >= TH)
        if not display:
            continue
        cc = {
            'name': 'Component-%d' % count,
            'bbox': box,
            'display': display
        }
        ccs.append(cc)
        count += 1 
        pass
    ###添加全局component###
    ccs.append({
            'name': 'ALL',
            'bbox': bbox.unpack(),
            'display': True
        })
    return ccs, bbox

class LayerClassifier:
    def __init__ (self):
        p = os.path.abspath(os.path.dirname(__file__))
        model_file = os.path.abspath(os.path.join(p, "models","model.pkl"))
        assert os.path.exists(model_file)
        with open(model_file,'rb') as f:
            self.model = pickle.load(f)
            pass
        pass

    def classify (self, dr, name, shapes):
        ft = classify.extract_features(dr, name, shapes)
        X = []
        X.append(ft)
        X = np.array(X, dtype=np.float32)
        label = self.model.predict(X)
        return label[0]
    pass

def analyze_layers (dr, cache, has_annotation, bbox):
    layers = []
    annos = []

    cls = LayerClassifier()

    for name, shapes in dr.layers.items():
        #cvs = canvas.JsonCanvas() 
        #dxf.render_shapes(cvs, shapes)
        #cache.render_layer(cvs, name)
        layer = {
            'name': name,
            'bbox': bbox.unpack(),
            #'top_view': cvs.dump(),
            }
        anno = {
            'name':name,
            'class': None
            }

        if has_annotation:
            if name[:3] == 'WDA':
                try:
                    label = CLASSES.index(name[5:])
                    anno['class'] = CLASSES[label]
                except:
                    pass
        else:
            heuristic = HEURISTIC_NAMES.get(name.upper(), None)
            if not heuristic is None:
                anno['class'] = heuristic
            else:
                label = cls.classify(dr, name, shapes)
                if label != 0:
                    anno['class'] = CLASSES[label]
        layers.append(layer)
        annos.append(anno)
        pass
    return layers, annos

def create_cache (dr):
    cache = DxfCacheCanvas(dr)
    for name, shapes in dr.layers.items():
        cache.addLayer(name)
        dxf.render_shapes(cache, shapes)
        pass
    return cache

def generate_full (bbox, cache):
    full = canvas.CvCanvas(bbox, 256, 50)
    #dr.render(full)
    cache.render(full)
    return full

def analyze_dxf (dxf_path, analyze_path, annotation_path):
    assert analyze_path is None     # not supported for now
    dr = dxf.Drawing(dxf_path)

    cache = create_cache(dr)

    has_annotation = False
    for name, shapes in dr.layers.items():
        if 'WDA' == name[:3]:
            has_annotation = True
            break
        pass

    boxes, bbox = analyze_components(cache)
    layers, annos = analyze_layers(dr, cache, has_annotation, bbox)

    full = {
            'navigation': boxes,
            'layers': layers,
            'classes': CLASSES,
            'has_annotation': has_annotation
            }

    annotation = {
            "layers":annos,
            "annotation_types":{
                'GUARD':[
                    "GUARD_OBSTACLE",
                    "GUARD_PASSAGE",
                    "GUARD_MINIROOM",
                    "GUARD_FORKLIFT",
                    "GUARD_WORKBENCH",
                    "GUARD_CONVEYOR",
                    "GUARD_AVG",
                    "GUARD_MANUP_TRUCK",
                    "GUARD_PPERATING_PLATFORM"
                ],
                'HINT': [
                    "HINT_ROI",
                    "HINT_CLEAR",
                    "HINT_CONNECT",
                    "HINT_DROP_ROOM"
                ]
            },
            "annotations": []
            }

    with open(annotation_path, 'w') as f:
        f.write(json.dumps(annotation))
        pass
    pass

def test_analyze (root):
    with open(os.path.join(root, 'wda.analyze') , 'r') as f:
        full = json.loads(f.read())
        pass
    for i, cc in enumerate(full['components']):
        if not cc['display']:
            continue
        box = Box(*cc['bbox'])
        cvs = canvas.CvCanvas(box, 1024, 20)
        for layer in full['layers']:
            for path in layer['top_view']['paths']:
                pts = path['points']
                color = path['color']
                with cvs.style(lineColor=color):
                    cvs.path(pts)
                    pass
                pass
            pass
        cvs.save(os.path.join(root, 'test-%d.png' % i))
        pass
    pass

#EXIT_RELAX = 1000
#FIRE_RELAX = 2500
#ROOM_RELAX = 1000
#DOCK_RELAX = 500

fixtures_type = {
'obstacles' : [5, 6, 9, 12], #column, misc column, obstacle, fire_hydrant
'guards' : [8, 10, 11], #guard, guard_2m, acc_guard
'isolates' : [5, 6], #column, misc column
'evitables' : [], #
}

color_list = {
    1:'wall',
    2:'dock',
    3:'door',
    4:'exit',
    5:'column',
    6:'misc_column',
    7:'safety_door',
    8:'guard',
    9:'obstacle',
    10:'guard_2m',
    11:'acc_guard',
    12:'fire_hydrant',
    13:'dock_in',
    14:'dock_out',
    15:'guard_passage',
    16:'customize',
    17:'forklift',
}
def get_bbox (room):

    box = Box()
    for obs in room:
        if obs['type'] == 7 or obs['type'] == 9:    # INVISIBLE
            continue
        for x, y in obs['polygon']:
            x = int(round(x))
            y = int(round(y))
            box.expand(Point(x, y))
            pass
        pass
    return box

def merge_blocks (connections):

    merged = []
    horizontal, vertical = [], []

    for item in connections:
        p = item['box']
        polygon = shapely.geometry.Polygon([(p[0][0], p[0][1]), (p[0][0], p[1][1]), (p[1][0], p[1][1]), (p[1][0], p[0][1])])
        if item['direction'] == [1, 0]:
            horizontal.append(polygon)
        else:
            vertical.append(polygon)
            pass
   
    vertical = shapely.ops.unary_union(vertical)
    horizontal = shapely.ops.unary_union(horizontal)
    
    multipoly = False
    

    if vertical.__class__.__name__ == 'MultiPolygon':
        for item in vertical:
            bd = item.bounds
            path = [[int(bd[0]), int(bd[1])],[int(bd[2]),int(bd[3])]]
            merged.append({'box':path, 'direction':[0, 1]})

    elif vertical.__class__.__name__ == 'Polygon':
        bd = vertical.bounds
        path = [[int(bd[0]), int(bd[1])],[int(bd[2]),int(bd[3])]]
        merged.append({'box':path, 'direction':[0, 1]})

    if horizontal.__class__.__name__ == 'MultiPolygon':
        for item in horizontal:
            bd = item.bounds
            path = [[int(bd[0]), int(bd[1])],[int(bd[2]),int(bd[3])]]
            merged.append({'box':path, 'direction':[1, 0]})

    elif horizontal.__class__.__name__ == 'Polygon':
        bd = horizontal.bounds
        path = [[int(bd[0]), int(bd[1])],[int(bd[2]),int(bd[3])]]
        merged.append({'box':path, 'direction':[1, 0]})     

    return merged

def distance (room_info, column, params):
 
    attached = False
    verticle_block = True
    horizontal_block = True
    
    obstacles = room_info['obstacles']
    blocks = room_info['connection_item']['blocks']
    bbox = room_info['bbox']
    walls = room_info['walls']

    bx0, by0, bx1, by1 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
    cx0, cy0, cx1, cy1 = column[0][0], column[0][1], column[1][0], column[1][1]
    p = [ (column[1][0] + column[0][0]) / 2, (column[1][1] + column[0][1]) / 2 ]

    #判断柱子是否在仓间内    
    point = shapely.geometry.Point(p[0], p[1])
    polygon = []

    #将墙面替换成使用shapely的polygon格式
    for item in walls:
        polygon.append((item[0],item[1]))

    #若柱子不在仓间内则不加入blocks
    if not shapely.geometry.Polygon(polygon).contains(point):
        return

    for id in range(len(walls)):
        if id == len(walls)-1:
            x0, y0, x1, y1 = walls[id][0], walls[id][1], walls[0][0], walls[0][1]
        else:
            x0, y0, x1, y1 = walls[id][0], walls[id][1], walls[id+1][0], walls[id+1][1]

        '''
        wall_poly = shapely.geometry.Polygon(([x0, y0], [x1, y0], [x1, y1], [x0, y1]))
        #计算柱子到墙面的距离
        dis = wall_poly.distance(point)    
        '''    
        
        cross = (x1 - x0) * (p[0] - x0) + (y1- y0) * (p[1] - y0)
        if cross <= 0:
            dis = math.sqrt((p[0] - x0) ** 2 + (p[1] - y0) ** 2)
            projection = [x0, y0]

        else:
            d2 = (x1 - x0) ** 2 + (y1 - y0) ** 2
            if cross > d2:
                dis = math.sqrt((p[0] - x1) ** 2 + (p[1] - y1) **2)
                projection = [x1, y1]

            else:
                r = cross / d2
                px = x0 + (x1 - x0) * r
                py = y0 + (y1 - y0) * r
                dis = math.sqrt((p[0] - px) ** 2 + (py - p[1]) ** 2)
                projection = [px, py]

        #小于params范围值则不加入blocks
        if dis < params:
            attached = True
            
            a0, b0, a1, b1 = cx0, cy0, cx1, cy1
            if projection[0] < p[0]: #以下注释掉的均为左下右上坐标结构
                horizontal_block = False

            elif projection[0] > p[0]:
                horizontal_block = False

            if projection[1] < p[1]:
                verticle_block = False

            elif projection[1] > p[1]:
                verticle_block = False
            #obstacles.append([[a0, b0], [a0, b1], [a1, b1], [a1, b0]])
            
    
    if attached == False:
        path1 = [[cx0, by0], [cx1, by1]]
        path2 = [[bx0, cy0], [bx1, cy1]]
        blocks.append({'box':path1, 'direction':[0, 1]})
        blocks.append({'box':path2, 'direction':[1, 0]})
    
    elif horizontal_block == True:
        path = [[bx0, cy0], [bx1, cy1]]
        blocks.append({'box':path, 'direction':[1, 0]})
   

    
    elif verticle_block == True:
        path = [[cx0, by0], [cx1, by1]]
        blocks.append({'box':path, 'direction':[0, 1]})
    return


def compute_connected_info ( room_info, docks, columns, params ):

    bbox = room_info['bbox']
    bx0, by0, bx1, by1 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1] 
    soft_moving_paths = []
    blocks = []
    for dock in docks: #由dock确定moving path
        x0, y0, x1, y1 = dock[0][0], dock[0][1], dock[1][0], dock[1][1]
        
        if max( abs(x1-x0), abs(y1-y0) ) < 2000:
            continue
        if abs(x1-x0) > abs(y1-y0) : #判断dock的方向
            path = [[x0, by0], [x1, by1]]
            #path = [[x0, by0], [x0, by1], [x1, by1], [x1, by0]]
            direction = [0, 1]

        else:
            path = [[bx0, y0], [bx1, y1]]
            #path = [[bx0, y0], [bx0, y1], [bx1, y1], [bx1, y0]]
            direction = [1, 0]

        soft_moving_paths.append({'box': path, 'direction': direction})
    
    room_info['connection_item'] = {'soft_moving_paths': soft_moving_paths, 'moving_paths':[], 'blocks':blocks}
    
    for i in columns: # column to point
        distance( room_info, i, params)

    room_info['connection_item']['blocks'] = merge_blocks(room_info['connection_item']['blocks'])
    room_info['connection_item']['soft_moving_paths'] = merge_blocks(room_info['connection_item']['soft_moving_paths'])
    #room_info['connection_item']['soft_moving_paths'] = []
   
    return 

def generate_room_info (fixtures, params):
    
    room_info = {}
    box = get_bbox(fixtures).unpack()
    room_info['bbox'] = [[box[0], box[1]], [box[2], box[3]]]
    room_info['obstacles'], room_info['guards'], room_info['isolates'], room_info['evitables'], room_info['walls'] = [], [], [], [], []
    docks = []
    columns = []

    for obs in fixtures:

        p = obs['polygon']
        pts = [[p[0][0], p[0][1]], [p[2][0], p[2][1]]]

        for item in fixtures_type:
            if obs['type'] in fixtures_type[item]:
                #room_info[item].append(pts)
                room_info[item].append(obs['polygon'][:-1])

        if obs['type'] == 1:
            room_info['walls'] += obs['polygon'][:-1]
        elif obs['type'] == 3:
            docks.append(pts)
        elif obs['type'] == 5:
            columns.append(pts)
            
    compute_connected_info(room_info, docks, columns, params)
    return room_info

def generate_scene_info (rooms_info, path):

    scene = {} #前端呈现部分
    bbox = Box()#全仓库bbox
    scene['navigation'] = []
    scene['top_view'] = []

    for room_id, room in enumerate(rooms_info):
        navigation = {}
        fixtures = room['fixtures']
        box = get_bbox(fixtures).unpack()
        bbox.expand(Point(box[0], box[1]))
        bbox.expand(Point(box[2], box[3]))

        navigation['bbox'] = [[box[0], box[1]], [box[2], box[3]]]
        navigation['room'] = 'room' + str(room_id+1)

        room_dir = os.path.join(path, str(room_id))
        render_room(fixtures, os.path.join(room_dir, 'room.png'))
        #render_room_svg(fixtures, os.path.join(room_dir, 'room.svg'))

        with open(os.path.join(room_dir, 'room.png'), 'rb') as f:
            buf = f.read()
            navigation['image'] = 'data:image/png;base64,' + base64.b64encode(buf).decode('ascii')
        scene['navigation'].append(navigation)

        for obs in fixtures:
            if obs['type'] == 8 or obs['type'] == 10 or obs['type'] == 11:
                continue
            top_view = {}
            top_view['points'] = obs['polygon']
            top_view['points'].append(top_view['points'][0])
            top_view['color'] = color_list[obs['type']]
            top_view['color_code'] = []
            scene['top_view'].append(top_view)


    box = bbox.unpack()
    scene['top_view_bbox'] = [[box[0], box[1]], [box[2], box[3]]]

    return scene

def render_room (room, path):
 
    box = get_bbox (room)
    cvs = CvCanvas(box, 1024, 50)

    for obs in room:
        if obs['type'] == 7:    # INVISIBLE
            continue


        add = 0
        if not obs['effective']:    # 如果不是effective的，改变颜色
            add += 3
        with cvs.style(lineColor=obs['type']+add):
            cvs.path(obs['polygon'], closed=True)
            pass
        pass
    cvs.save(path)

def render_room_svg (room, path):
    from svg_canvas import SvgCanvas
    box = get_bbox (room)
    cvs = SvgCanvas(box)
    for obs in room:
        if obs['type'] == 7:    # INVISIBLE
            continue
        add = 0
        if not obs['effective']:    # 如果不是effective的，改变颜色
            add += 3
        with cvs.style(lineColor=obs['type']+add):
            cvs.path(obs['polygon'], closed=True)
            pass
        pass
    cvs.save(path)
    pass

def render_room_dxf (room, path):
    import dxfwriter
    cvs = dxfwriter.DxfCanvas()
    for obs in room:
        if obs['type'] == 7:    # INVISIBLE
            continue
        add = 0
        if not obs['effective']:    # 如果不是effective的，改变颜色
            add += 3
        with cvs.style(lineColor=obs['type']+add):
            cvs.path(obs['polygon'], closed=True)
            pass
        pass
    cvs.save(path)
    pass

def extract_error_file (dr, scene, output_root):

    scene['navigation'] = []
    scene['top_view'] = []
    box = dr.bbox()
    navigation = {}

    scene['top_view_bbox'] = [[box[0][0],box[0][1]],[box[1][0],box[1][1]]]
    navigation['bbox'] = scene['top_view_bbox']
    navigation['room'] = 'room1'
    navigation['image'] = ''
    scene['navigation'].append(navigation)

    for entity in dr.dxf.modelspace():
        item = {}
        p = []
        dtype = entity.dxftype() 

        if dtype == 'LWPOLYLINE' :
            pts = entity.get_points()
            
            for i in pts:
                #p.append([i[0])
                p.append([int(i[0]), int(i[1])])
            p.append(p[0])

        elif dtype == 'LINE' :
            s = entity.dxf.start
            e = entity.dxf.end
            p = [[int(s[0]), int(s[1])], [int(e[0]), int(e[1])]]

        item['points'] = p
        item['color'] = ''
        item['color_code'] = entity.dxf.color


        scene['top_view'].append(item)
    
    room_dir = os.path.join(output_root, '0')
    '''
    img_dir = os.path.join(room_dir, 'room.png')
    canvas = CvCanvas(box, 1024, 50)
    dr.render(canvas)
    canvas.save(img_dir)
    '''

def process_dxf (dxf_path, annotation_path, output_root, params):
    # 注意,annotation_path是Layout第一版的标注文件地址, 
    # 在项目外跑可以用None.

    print(dxf_path, output_root)
    # 删除现有的结果
    found = True
    for existing_room in glob.glob(os.path.join(output_root, '*')):
        print("FOUND EXISTING ROOM DIR", existing_room)
        found = True
        pass
    if found:
        shutil.rmtree(output_root)
        os.mkdir(output_root)

    dr = dxf.Drawing(dxf_path)


    hint_roi = []
    hint_clear = []
    hint_dock = []
    hint_dock_in = []
    hint_dock_out = []
    add_guard = []
    add_passage = []

    layer_anno = {}

    ui_anno_mapping = {
            # 之所以用这种mapping，有一个好处是'WALLS'这种常量之出现一次，避免typo。
            # 下面的dxf_anno_mapping也是
            # 如果哪天protocol改了，‘WALLS’需要统一改成'WALL'，也只需要改一个地方
            'WALLS': 'WDAS_WALLS',
            'PILLARS': 'WDAS_PILLARS',
            'DOORS': 'WDAS_DOORS',
            'OBSTACLE': 'WDAS_OBSTACLE'
    }

    proc = CadProcessor()

    if annotation_path is None or not os.path.exists(annotation_path):
        print("WARNING: cannot find annotation path", annotation_path)
        pass
    else:
        with open(annotation_path, 'r') as f:
            f = json.loads(f.read())

            for l in f['layers']:
                anno = ui_anno_mapping.get(l['class'], None)
                if not anno is None:
                    layer_anno[l['name']] = anno
                pass

            for a in f['annotations']:
                x1,y1,x2,y2 = a["bbox"]
                anno_type = a["type"]
                #anno_layer = a["layer"]
                #if anno_type == 'CAD_ADD':
                #    assert False, "TODO"
                #    continue;
                #    #msp.add_lwpolyline(points,dxfattribs={'layer':anno_layer, })
                x1 = round(x1)
                y1 = round(y1)
                x2 = round(x2)
                y2 = round(y2)
                proc.annotate(anno_type, x1, y1, x2, y2)
        pass

    cvs = DxfLoadingCanvas(proc)

    for name, shapes in dr.layers.items():
        anno = layer_anno.get(name, name)
        if not proc.select(anno):
            continue
        dxf.render_shapes(cvs, shapes)
        pass

    rooms = {}
    rooms_info, ccs = proc.extract(output_root)
    fixtures = {}
    scene = {}

    if rooms_info == []:
        extract_error_file(dr, scene, output_root)
    else:
        scene = generate_scene_info(rooms_info, output_root)

    for room_id, room in enumerate(rooms_info):
        fixtures = room['fixtures']
        room_info = generate_room_info(fixtures, params )
        rooms['room'+str(room_id+1)] = room_info
        
    '''
    with open('output.json', 'w')as f:
        json.dump(scene, f)
    '''
    return (rooms, scene)  

def decode_dxf (dir, params = 1000):

    assert os.path.exists(dir)

    #output_dir = os.path.abspath(os.path.join(dir, "project", "input", "dxf" ))
    dxf_path = os.path.abspath(os.path.join(dir, "wda.dxf"))
    dwg_path = os.path.abspath(os.path.join(dir, "wda.dwg"))
    annotation_path = os.path.abspath(os.path.join(dir, "wda.annotation"))

    if os.path.exists(dxf_path):
        pass
    elif os.path.exists(dwg_path):
        # convert
        convert.dwg2dxf(dwg_path, dxf_path)
        pass

    output_path = os.path.abspath(os.path.join(dir, "output"))
    analyze_dxf(dxf_path, None, annotation_path)

    assert not output_path is None
    os.makedirs(output_path, exist_ok=True)
    rooms, scene = process_dxf(dxf_path, annotation_path, output_path, params)
    return (rooms,  scene)  
    

if __name__ == '__main__':
    # 可以直接用命令行调用
    # render.py --path xx/path_to_dxf_file
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='.', help='dxf path to be added to')
    args = parser.parse_args()
    decode_dxf(args.path)

    pass

