#!/usr/bin/env python3
from abc import ABC, abstractmethod
from contextlib import contextmanager
import math
import ezdxf
from cad import Box
import numpy as np
import cv2
import os

def angle_cad_to_cv (angle, start_angle, end_angle, flip=True):
    angle = round(angle)
    start_angle = round(start_angle)
    end_angle = round(end_angle)

    if start_angle == 0 and end_angle == 360:
        return angle, 0, 360

    if flip:
        angle, start_angle, end_angle = -angle, -end_angle, -start_angle

    while end_angle < start_angle:
        end_angle += 360

    if end_angle - start_angle > 180:
        end_angle -= 360
        start_angle, end_angle = end_angle, start_angle
    return angle, start_angle, end_angle


CC_RELAX = 5000 

class Style:
    def __init__(self, lineColor=0, fillColor=None):
        self.lineColor = lineColor
        self.fillColor = fillColor
        pass

    def copy(self):
        s = Style()
        s.lineColor = self.lineColor
        s.fillColor = self.fillColor
        return s

    def __str__(self):
        return 'LC: %s FC: %s' % (self.lineColor, self.fillColor)
    pass


class Canvas:

    def __init__(self):
        self.styles = [Style()]  # style栈，可以用with canvas.style不断往里压
        pass

    @contextmanager
    def style(self, **kwargs):
        try:
            s = self.styles[-1].copy()
            for k, v in kwargs.items():
                s.__setattr__(k, v)
                pass
            self.styles.append(s)
            yield None
        finally:
            self.styles.pop()
            pass
        pass

    def line(self, v1, v2):
      self.path([v1, v2])
      pass

    def hatch (self, points):
        """
        图案填充
        :param points: 边界点
        :return:
        """
        lc = self.styles[-1].lineColor
        fc = self.styles[-1].fillColor
        if fc is None:
            fc = lc
        with self.style(lineColor=lc, fillColor=fc):
            self.path(points, closed=True)
        pass
    pass


# dongwei: 注意不能有颜色[0,0,0] -- 会导致检测失效
TABLEAU20 = [[255, 255, 255],[127, 127, 225],[220, 10, 10],[230, 220, 10],[20, 200, 10],[170, 20, 220],[200, 200, 200],[0, 230, 230],[100, 100, 100],[180, 119, 31],[232, 199, 174],[14, 127, 255],[120, 187, 255],[44, 160, 44],[138, 223, 152],[40, 39, 214],[150, 152, 255],[189, 103, 148],[213, 176, 197],[75, 86, 140],[148, 156, 196],[194, 119, 227],[210, 182, 247],[127, 127, 127],[199, 199, 199],[34, 189, 188],[141, 219, 219],[207, 190, 23],[229, 218, 158],
        [240, 240, 240],[127, 127, 225],[220, 10, 10],[230, 220, 10],[20, 200, 10],[170, 20, 220],[200, 200, 200],[0, 230, 230],[100, 100, 100],[180, 119, 31],[232, 199, 174],[14, 127, 255],[120, 187, 255],[44, 160, 44],[138, 223, 152],[40, 39, 214],[150, 152, 255],[189, 103, 148],[213, 176, 197],[75, 86, 140],[148, 156, 196],[194, 119, 227],[210, 182, 247],[127, 127, 127],[199, 199, 199],[34, 189, 188],[141, 219, 219],[207, 190, 23],[229, 218, 158]]

class RasterCanvas(Canvas):

    def __init__ (self, bbox, size, padding=0):
        '''
        bbox: 被画对象的bounding box
        size: canvas较长边的大小
        '''
        super().__init__()
        self.padding = padding
        self.styles = [Style()]
        self.bbox = bbox
        self.palette = TABLEAU20
        x0, y0, x1, y1 = bbox.unpack()
        w = x1 - x0
        h = y1 - y0
        assert w > 0 and h > 0
        l = max(w, h)
        self.scale_num = size - 1 - padding * 2
        self.scale_denom = l
        self.size = ((h * self.scale_num + self.scale_denom - 1) // self.scale_denom + 1 + padding * 2), \
                    ((w * self.scale_num + self.scale_denom - 1) // self.scale_denom + 1 + padding * 2)
        pass

    def scale (self, l):
        return l * self.scale_num / self.scale_denom;

    def unscale (self, l):
        return l * self.scale_denom / self.scale_num;

    def map (self, vector):
        '''坐标转换, ezdxf.math.vector转成整数(x,y)'''
        x = round((vector[0] - self.bbox[0][0]) * self.scale_num / self.scale_denom)
        y = round((vector[1] - self.bbox[0][1]) * self.scale_num / self.scale_denom)
        return (x + self.padding, self.size[0] - y - self.padding)

    def unmap (self, pt):
        '''坐标逆转换，返回的是浮点数'''
        x, y = pt
        x -= self.padding
        y = self.size[0] - y - self.padding
        x = x * self.scale_denom / self.scale_num + self.bbox[0][0]
        y = y * self.scale_denom / self.scale_num + self.bbox[0][1]
        return (x, y)

    def scale (self, r):
        ''' 半径转换为整数(四舍五入)'''
        return round(r * self.scale_num / self.scale_denom)


class CvCanvas(RasterCanvas):

    def __init__ (self, box, size, padding=0):
        super().__init__(box, size, padding)
        self.image = np.zeros(self.size + (3,), dtype=np.uint8)
        pass

    def gray (self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def lineColor (self):
        '''获取当前应该用的颜色, [b,g,r]'''
        return self.palette[self.styles[-1].lineColor]

    def fillColor (self):
        '''获取当前应该用的颜色, [b,g,r]'''
        if self.styles[-1].fillColor is None:
            return None
        return self.palette[self.styles[-1].fillColor]

    def copycvs (self, target, resolution):
        self.image = cv2.resize( target.image, (resolution, resolution), interpolation=cv2.INTER_CUBIC )
    
    def path (self,points,closed = False):
        """
        多个点构成的折线
        :param points: 多个点  [(x1,y1),(x2,y2)]
        :param closed: 图形是否闭合
        """
        if len(points) == 0:
            return
        pts = []
        for p in points:
            pts.append(self.map(p))

        if closed and not self.fillColor() is None:
            #实现hatch
            cv2.fillPoly(self.image, [np.round(np.array(pts)).astype(np.int32)], self.fillColor())
            return
        cv2.polylines(self.image, [np.round(np.array(pts)).astype(np.int32)], closed, self.lineColor())
        pass

    def arc (self,center,radius, angle, start_angle, end_angle):
        """
        圆弧（可实现 圆 、 椭圆 、 圆弧等）
        :param center: 中心
        :param radius: 半径 格式为（r1,r2),r1为半长轴，r2为半短轴。若需绘制图形为圆，则r1=r2
        :param angle: 旋转的角度 顺时针
        :param start_angle: 开始角度
        :param end_angle: 结束角度
        :param shift: 线宽 -1填充图形 默认0
        """
        angle, start_angle, end_angle = angle_cad_to_cv(angle, start_angle, end_angle)

        cv2.ellipse(self.image, self.map(center), 
                (self.scale(radius[0]), self.scale(radius[1])),
                angle, start_angle, end_angle, self.lineColor())
        pass


    #def MText(self, string, center,angle, scale=1.2):
    #    #TODO: 旋转文字
    #    font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
    #    cv2.putText(self.image,string,self.map(center),font,scale,self.lineColor(),1)
    #def Text(self,string ,center,angle, scale=1.2):
    #    # TODO: 旋转文字
    #    font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
    #    cv2.putText(self.image, string, self.map(center), font, scale, self.lineColor(), 1)

    def save(self, path):
        cv2.imwrite(path, self.image)
        pass

    def save_alpha(self, path):
        alpha = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        alpha = (alpha > 0) * 255
        image = np.dstack([self.image, alpha])
        cv2.imwrite(path, image)
        pass

    pass


def expandFloat(box, vec):
    box.expandFloat((vec[0], vec[1]))
    pass


def round_point(v):
    return [round(v[0]), round(v[1])]


class ShapeStatCanvas (Canvas):
    def __init__(self, box, size, padding=0):
        super().__init__(box, size, padding)
        self.path_num = 0
        self.arc_num = 0
        pass

    def path (self,points,closed = False):
        """
        多个点构成的折线

        """
        self.path_num += 1
        print("总线段数",self.path_num)

    def arc (self,center,radius, angle,start_angle, end_angle):
        """
        圆弧（可实现 圆 、 椭圆 、 圆弧等）

        """
        self.arc_num += 1
        print("总线段数", self.arc_num)
        pass

class JsonCanvas(Canvas):
    '''把画的内容存成我们内部格式的json'''

    def __init__(self):
        super().__init__()
        self.shapes = []
        self.bbox = Box()
        self.label = ''
        self.offset = [0,0]
        pass

    def path (self,points,closed = False):
        """
        多个点构成的折线
        """
        if len(points) == 0:
            return
        for v in points:
            expandFloat(self.bbox, v)
        points = [round_point(v) for v in points]
        if closed:
            points.append(points[0])
        points = [[p[0]+self.offset[0],p[1]-self.offset[1]] for p in points]
        self.shapes.append({
            'points': points, 'color': self.styles[-1].lineColor
        })
        pass

    def arc (self,center,radius, angle, start_angle, end_angle):
        angle, start_angle, end_angle = angle_cad_to_cv(angle, start_angle, end_angle)
        center = round(center[0]), round(center[1])
        radius = (round(radius[0]), round(radius[1]))
        pts = cv2.ellipse2Poly(center, radius, angle, start_angle, end_angle, 20)
        ll = []
        for i in range(pts.shape[0]):
            x, y = pts[i]
            ll.append([int(x), int(y)])
            pass
        self.path(ll)
        pass

    def dump(self):
        return {'paths': self.shapes}

    def update(self, second_canvas):
        self.shapes.append(second_canvas.shapes)
    pass


class CompactingCanvas (CvCanvas):
    def __init__ (self, boxes, size, padding = 0):

        self.boxes = boxes
        self.size = size
        self.padding = padding
        self.mapped_boxes, self.vects = compact_boxes (self.boxes)
        self.bbox = bound_boxes(self.mapped_boxes)

        super().__init__(self.bbox, self.size, self.padding)

    def path (self,points,closed = False):
        """
        多个点构成的折线
        :param points: 多个点  [(x1,y1),(x2,y2)]
        :param closed: 图形是否闭合
        """
        pts = []
        for p in points:
            conv_p = convert_point(p, self.boxes, self.vects)
            if conv_p is not None:
                pts.append(self.map(conv_p))
                pass
            pass
        if len(pts) == 0:
            return

        if closed and not self.fillColor() is None:
            #实现hatch
            cv2.fillPoly(self.image, [np.round(np.array(pts)).astype(np.int32)], self.fillColor())
            return
        cv2.polylines(self.image, [np.round(np.array(pts)).astype(np.int32)], closed, self.lineColor())
        pass

    def arc (self,center,radius, angle,start_angle, end_angle):
        """
        圆弧（可实现 圆 、 椭圆 、 圆弧等）
        :param center: 中心
        :param radius: 半径 格式为（r1,r2),r1为半长轴，r2为半短轴。若需绘制图形为圆，则r1=r2
        :param angle: 旋转的角度 顺时针
        :param start_angle: 开始角度
        :param end_angle: 结束角度
        :param shift: 线宽 -1填充图形 默认0
        """
        if center is not None:
            if len(center) > 2:
                center = center[:2]
            conv_center = convert_point(center, self.boxes, self.vects)
            if conv_center is not None:
                angle, start_angle, end_angle = angle_cad_to_cv(angle, start_angle, end_angle)
                cv2.ellipse(self.image, self.map(conv_center),
                            (self.scale(radius[0]), self.scale(radius[1])),
                            angle, start_angle, end_angle, self.lineColor())
        pass


def compact_boxes(boxes, dist=5000):
    i = 0 
    mapped_boxes = []
    vects = []
    xo = 0  #compact图的左上角
    yo = 0
    wide = 0
    height = 0
    tmp_x = 0 #左上角 
    for box in boxes:
        x0, y0, x1, y1 = box.unpack()
        # 第一个box
        if i == 0:
            mapped_boxes.append(box)
            vects.append([0,0])
            xo = x0
            yo = y1
            tmp_x = x0
            
        else:
            tmp_x += (wide + dist) #更新左上角 
            mapped_boxes.append(Box(tmp_x, yo+y0-y1, tmp_x+x1-x0, yo))
            vects.append([tmp_x-x0,yo-y1])

        wide = x1-x0
        height = y1-y0
        
        i += 1
        
    return mapped_boxes, vects

def bound_boxes(mapped_boxes):
    if len(mapped_boxes) < 1:
        return None
    min_x = min_y = max_x = max_y = None
    for i in range(len(mapped_boxes)):
        box = mapped_boxes[i]
        if i == 0:
            min_x, min_y, max_x, max_y = box.unpack()
        else:
            x0, y0, x1, y1 = box.unpack()
            if x1> max_x:
                max_x = x1
            if y0 < min_y:
                min_y = y0
    return Box(min_x, min_y, max_x, max_y)

def convert_point(p, boxes, vects):
    for i in range(len(boxes)):
        x0, y0, x1, y1 = boxes[i].unpack()
        xp, yp = p
        if x0 <= xp <= x1 and y0 <= yp <= y1:
            #return Point(xp+vects[i][0],yp+vects[i][1])
            return (xp+vects[i][0],yp+vects[i][1])

class CCXtorCanvas(Canvas):
    # 提取CAD中的连续模块
    # connected component extractor
    # extracts bounding boxes of connected components

    def __init__ (self, relax=CC_RELAX):
        super().__init__()
        self.boxes = []
        self.cnt = 0
        self.relax = relax
        pass

    def update (self, box):
        for bb in self.boxes:
            if overlap(bb, box, self.relax):
                bb.expand(box)
                box = None
                break
            pass
        if not box is None:
            self.boxes.append(box)
            pass
        self.cnt += 1

        if self.cnt % 1000 == 0:
            self.merge()
        pass
    
    def merge (self):
        while True:
            merged = 0
            N = len(self.boxes)
            for i in range(N):
                if self.boxes[i] is None:
                    continue
                for j in range(i+1, N):
                    if self.boxes[j] is None:
                        continue
                    if overlap(self.boxes[i], self.boxes[j], self.relax):
                        self.boxes[i].expand(self.boxes[j])
                        self.boxes[j] = None
                        merged += 1
                        pass
                    pass
                pass
            if merged == 0:
                break
            self.boxes = [b for b in self.boxes if not b is None]
            pass

    def path (self, points, closed = False):
        box = bound_points(points)
        self.update(box)
        pass

    def arc (self, center, radius, angle, start_angle, end_angle):
        pass

    def extract (self):
        self.merge()
        return self.boxes
    pass

def overlap (box1, box2, relax = 0):
    for i in range(2):
        lb = max(box1[0][i] - relax, box2[0][i])
        ub = min(box1[1][i] + relax, box2[1][i])
        if lb > ub:
            return False
        pass
    return True

def bound_points (points):
    box = Box()
    for x, y in points:
        box.expandFloat((x, y))
        pass
    return box


# 我们用CvCanvs搞pillars误差太大
# 所以用CvCanvas确定pillars的untight bounding box以后
# 需要用原始的向量图数据来细化pillars的准确边界
class PillarRefiningCanvas (Canvas):
    def __init__ (self, pillars):
        super().__init__()
        self.boxes = [(pillar, Box()) for pillar in pillars]
        pass

    def path (self, points, closed = False):
        box = bound_points(points)
        for p in self.boxes:
            if overlap(p[0], box):
                p[1].expand(box)
        pass

    def arc (self, center, radius, angle, start_angle, end_angle):
        # TODO: dongwei: 这个写得比较糙,先用着
        r = max(radius[0], radius[1])
        p0 = (center[0]-r, center[1]-r)
        p1 = (center[0]+r, center[1]+r)
        self.path([p0, p1])
        pass

    def extract (self):
        for p in self.boxes:
            x1, y1, x2, y2 = p[0].unpack()
            s1 = (x2 - x1) * (y2 - y1)
            x1, y1, x2, y2 = p[1].unpack()
            s2 = (x2 - x1) * (y2 - y1)
            assert s2 > 0
            #print("PILLAR REDUCTION ", s2/s1)
            pass
        return [p[1] for p in self.boxes]
    pass

class DxfLoadingCanvas (Canvas):
    def __init__ (self, proc):
        super().__init__()
        self.proc = proc
        pass

    def path (self, points, closed = False):
        for i in range(1, len(points)):
            p0 = points[i-1]
            p1 = points[i]
            self.proc.add(p0[0], p0[1], p1[0], p1[1])
            pass
        if closed and len(points) > 1:
            p0 = points[-1]
            p1 = points[0]
            self.proc.add(p0[0], p0[1], p1[0], p1[1])
        pass

    def arc(self, center, radius, angle, start_angle, end_angle):
        cx, cy = round(center[0]), round(center[1])
        radius = (round(radius[0]), round(radius[1]))
        angle, start_angle, end_angle = angle_cad_to_cv(angle, start_angle, end_angle, False)
        pts = cv2.ellipse2Poly((cx, cy), radius, angle, start_angle, end_angle, 10)
        self.path(pts)
        pass
    pass

class DxfCacheCanvas (Canvas):

    def __init__ (self, dr):
        super().__init__()
        self.layers = {}
        self.paths = []
        self.arcs = []
        pass


    def addLayer (self, layer_name):
        paths = []
        arcs = []
        self.layers[layer_name] = (paths, arcs)
        self.paths = paths
        self.arcs = arcs
        pass

    def path (self, points, closed = False):
        self.paths.append((points, closed))
        pass

    def arc(self, center, radius, angle, start_angle, end_angle):
        self.arcs.append((center, radius, angle, start_angle, end_angle))
        pass

    def render_one (self, cvs, one):
        paths, arcs = one
        for path in paths:
            cvs.path(*path)
            pass
        for arc in arcs:
            cvs.arc(*arc)
            pass
        pass

    def render (self, cvs):
        for k, v in self.layers.items():
            self.render_one(cvs, v)
            pass
        pass

    def render_layer (self, cvs, layer_name):
        self.render_one(cvs, self.layers[layer_name])
        pass
    pass


