#!/usr/bin/env python3
import os
import sys
import math
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
import ezdxf
from canvas import CvCanvas
from cad import Point, Box

MAX_COLORS = 20

class ShapeHandler(ABC):
    '''Handle DXF shapes
    '''

    def color(shape):
        # 获取形状颜色代码
        tc = None
        if shape.dxf.color == 0:
            #TODO: 取block颜色, 需要实现
            #print("TODO: COLOR BY BLOCK")

            '''
            for i in shape.drawing.blocks:
                if i.__contains__(shape):
                    print(i.get_attdef_text('color'))
            '''
            # tc = shape.drawing.blocks.get(shape.dxf.block).dxf.rgb
            pass
        elif shape.dxf.color == 256:
            # 取layer颜色
            #print(shape.graphic_properties() )
            #tc = shape.drawing.layers.get(shape.dxf.layer).dxf.color
            tc = shape.dxf.color
            pass

        elif shape.dxf.color == 257:
            #取obj颜色
            #print("TODO: COLOR BY OBJ")
            pass
        else:
            tc = shape.dxf.color

        if tc is None or tc < 0 or tc >= MAX_COLORS:
            tc = 0
        return tc

    @staticmethod
    @abstractmethod
    def bound(box, shape):
        '''更新bounding box box'''
        pass

    @staticmethod
    @abstractmethod
    def render(canvas, shape):
        '''Draw shape on canvas'''
        pass

    @staticmethod
    @abstractmethod
    def tostr(shape):
        '''convert information to string'''
        pass
    pass


# map from ezdxf.entities.XXX to ShapeMethods
SHAPE_HANDLERS = {}
MISSING_HANDLERS = set()


def bound_shapes(shapes):
    # 返回shapes的bounding box
    box = Box()
    for shape in shapes:
        handler = SHAPE_HANDLERS.get(type(shape), None)
        if handler is None:
            MISSING_HANDLERS.add(type(shape))
            continue
        handler.bound(box, shape)
        pass
    return box


def render_shapes(canvas, shapes):
    for shape in shapes:
        handler = SHAPE_HANDLERS.get(type(shape), None)
        if handler is None:
            MISSING_HANDLERS.add(type(shape))
            continue
        handler.render(canvas, shape)
        pass


def test_overlap(box, shape):
    # 测试shape是否在box里面
    handler = SHAPE_HANDLERS.get(type(shape), None)
    if handler is None:
        MISSING_HANDLERS.add(type(shape))
        return False
    box2 = Box()
    handler.bound(box2, shape)
    for d in [0, 1]:
        m = max(box[0][d], box2[0][d])
        M = min(box[1][d], box2[1][d])
        if m > M:
            return False
        pass
    return True


class Drawing:
    '''The content of a CAD file, or DXF file for now.
    '''

    def __init__(self, dxf_path, skip_hidden=True):
        self.dxf = ezdxf.readfile(dxf_path)
        self.layers = defaultdict(lambda: [])
        for entity in self.dxf.modelspace():
            layer = getattr(entity.dxf, 'layer', None)
            if not layer is None:
                self.layers[layer].append(entity)
                pass
            pass
        if skip_hidden:
            for layer in self.dxf.layers:
                name = layer.dxf.name
                if layer.is_off()  or layer.is_frozen() and name in self.layers:
                    del self.layers[name]
        pass

    def bbox(self):
        # 返回全图的bounding box
        return bound_shapes(self.dxf.modelspace())

    def render(self, canvas):
        # 把cad画到canvas上。canvas是提前用全图的bbox创建的
        render_shapes(canvas, self.dxf.modelspace())
        pass

    def segment(self, threshold=0.1, MAX=2048):
        # 用图像处理的方式，把CAD分成几个区域
        # 返回世界坐标系的boxes
        import cv2
        import scipy.ndimage
        from skimage import measure

        cvs = CvCanvas(self.bbox(), MAX)
        self.render(cvs)

        image = cvs.image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = gray > 0
        mask = scipy.ndimage.morphology.binary_fill_holes(mask).astype(np.uint8)
        labels = measure.label(mask > 0, background=0)

        regions = measure.regionprops(labels)
        # for box in measure.regionprops(labels):
        #    y0, x0, y1, x1 = box.bbox
        regions.sort(key=lambda x: -x.area)
        if len(regions) == 0:
            return []

        ath = regions[0].area * threshold
        boxes = []
        for region in regions:
            if region.area < ath:
                break
            y0, x0, y1, x1 = region.bbox
            box = Box()
            box.expandFloat(cvs.unmap((x0, y0)))
            box.expandFloat(cvs.unmap((x1, y1)))
            boxes.append(box)
            pass
        return boxes

    pass


def expandFloat(box, vec):
    box.expandFloat((vec.x, vec.y))
    pass

def radian2angle(radian):
    """ 弧度 转化为 角度 """
    angle = (radian / math.pi) * 180
    return angle


def angle2radian(angle):
    """ 角度转化为弧度 """
    radian = (angle / 180) * math.pi
    return radian


def cal_angle(center, point):
    """
    利用向量点乘 ,计算center为顶点，center->point 与 x 轴 的夹角
    :param center: 顶点 tuple
    :param point:  x轴外边上一点 tuple
    :return:angle 角度
    """
    center = center[:2]
    point = point[:2]
    vec1 = (point[0] - center[0], point[1] - center[1])
    dis_of_vec1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)  # 向量vec1长度
    # vec1·vec2 = |vec1|*|vec2|*cos(θ)
    vec2 = (1,0)  #x轴方向
    # 由于将(1,0)作为vec2，计算时 vec1·vec2 = vec1[0] ,|vec2| = 1
    # TODO: consider math.atan2 so we don't need to worry about division by 0
    cos_Theta = vec1[0] / max(dis_of_vec1, 0.00001)

    cos_Theta = math.acos(cos_Theta)  # 弧度
    # Theta = math.atan2(vec1[1], vec1[0])
    angle = radian2angle(cos_Theta)  # 转换为角度
    if vec1[1] < 0:
        # 如果 vec1的y轴方向向下，角在第三、第四象限，故 360-angle
        angle = 360 - angle
    return angle

# 以这个为范例
class LineHandler(ShapeHandler):
    @staticmethod
    def bound(box, shape):
        # shape.dxf.start的类型是ezdxf.math.vector
        # 这个类型是本质上就是(x, y, z)
        # 其.vec2属性对应的是(x,y)

        # Box本身只存储整数值
        # Box.expandFloat(p)相当于
        # Box.expand(p往上取整) +  Box.expand(p往下取整)
        if shape.dxf.start == shape.dxf.end:
            return
        expandFloat(box, shape.dxf.start.vec2)
        expandFloat(box, shape.dxf.end.vec2)
        pass

    @staticmethod
    def render(canvas, shape):
        # 用canvas的API画图
        if shape.dxf.start == shape.dxf.end:
            return
        with canvas.style(lineColor=ShapeHandler.color(shape)):
            # 颜色scope
            points = [shape.dxf.start.vec2, shape.dxf.end.vec2]
            canvas.path(points)
        pass

    @staticmethod
    def tostr(shape):
        return 'Line from %s to %s' %(shape.dxf.start.vec2,shape.dxf.end.vec2)
        pass

    pass


class CircleHandler(ShapeHandler):
    @staticmethod
    def bound(box, shape):
        center = shape.dxf.center.vec2  # 等同于 .vec2属性
        r = shape.dxf.radius

        vec = ezdxf.math.Vector(r, r)
        expandFloat(box, center - vec)
        expandFloat(box, center + vec)
        pass

    @staticmethod
    def render(canvas, shape):
        # 用canvas的API画图
        with canvas.style(lineColor=ShapeHandler.color(shape)):
            # 颜色scope
            canvas.arc(shape.dxf.center.vec2, (shape.dxf.radius, shape.dxf.radius), 0, 0, 360)
            pass
        pass

    def tostr(shape):
        return 'Circle at %s has radius %s' %(shape.dxf.center.vec2,shape.dxf.radius)
        pass
    pass

class EllipseHandler(ShapeHandler):
    def convert (shape):
        x, y = shape.dxf.center.vec2
        l1 = shape.dxf.major_axis.magnitude
        angle = shape.dxf.major_axis.angle
        l2 = l1 * shape.dxf.ratio

        angle = radian2angle(angle)
        start = radian2angle(shape.dxf.start_param)
        end = radian2angle(shape.dxf.end_param)
        return (x, y), (l1, l2), angle, start, end

    @staticmethod
    def bound(box, shape):
        #center = shape.dxf.center.vec2  # 获得中心点坐标
        #r1 = shape.dxf.major_axis.vec2  # 半长轴
        #tmp = shape.dxf.ratio * r1  # 半短轴
        #r2 = ezdxf.math.Vector((-tmp.y, tmp.x))  # 转换为与半长轴垂直的角度
        #expandFloat(box, center - r1 - r2)
        #expandFloat(box, center + r1 + r2)

        # TODO
        pass

    @staticmethod
    def render(canvas, shape):
        # 用canvas的API画图
        with canvas.style(lineColor=ShapeHandler.color(shape)):
            canvas.arc(*EllipseHandler.convert(shape)) 
            pass

    def tostr(shape):
        return 'Ellipse at %s has major axis %s with ratio %s' %(shape.dxf.center.vec2,shape.dxf.major_axis.vec2,shape.dxf.ratio)
        pass
    pass


class ArcHandler(ShapeHandler):
    @staticmethod
    def bound(box, shape):
        center = shape.dxf.center.vec2
        r = shape.dxf.radius  # 半径
        vec = ezdxf.math.Vector(r, r)
        expandFloat(box, center - vec)
        expandFloat(box, center + vec)
        pass

    @staticmethod
    def render(canvas, shape):
        with canvas.style(lineColor=ShapeHandler.color(shape)):
            center = shape.dxf.center.vec2
            r = shape.dxf.radius
            start_angle = radian2angle(shape.dxf.start_angle)
            end_angle = radian2angle(shape.dxf.end_angle)
            #if end_angle < start_angle:
            #    end_angle += 360
            canvas.arc(center, (r, r), 0, start_angle, end_angle)
            pass

    def tostr(shape):
        return 'Arc at %s has radius %s with start angle %s and end angle %s' %(shape.dxf.center.vec2,shape.dxf.radius,shape.dxf.start_angle,shape.dxf.end_angle)
        pass
    pass


class LWPolylineHandler(ShapeHandler):

    @staticmethod
    def bound(box, block):
        for shape in block.virtual_entities():
            handler = SHAPE_HANDLERS.get(type(shape), None)
            assert not handler is None
            handler.bound(box, shape)
        pass

    @staticmethod
    def render(canvas, block):
        with canvas.style(lineColor=ShapeHandler.color(block)):
            for shape in block.virtual_entities():
                handler = SHAPE_HANDLERS.get(type(shape), None)
                assert not handler is None
                handler.render(canvas, shape)
                pass
        pass

    def tostr(shape):
        return 'LWPolyline has %s points with axis %s' %(shape.dxf.count, shape.get_points())
        pass


class SolidHandler(ShapeHandler):
    @staticmethod
    def bound(box, shape):
        '''
        expandFloat(shape.dxf.vtx0)
        expandFloat(shape.dxf.vtx1)
        expandFloat(shape.dxf.vtx2)
        expandFloat(shape.dxf.vtx3)
        '''
        ######LINGRUI#######
        expandFloat(box, shape.dxf.vtx0)
        expandFloat(box, shape.dxf.vtx1)
        expandFloat(box, shape.dxf.vtx2)
        expandFloat(box, shape.dxf.vtx3)

        pass

    @staticmethod
    def render(canvas, shape):
        with canvas.style(lineColor=ShapeHandler.color(shape), fillColor=ShapeHandler.color(shape)):
            points = [shape.dxf.vtx0[:2], shape.dxf.vtx1[:2], shape.dxf.vtx3[:2], shape.dxf.vtx2[:2]]
            canvas.path(points, closed=True)
            pass
        pass


class HatchHandler(ShapeHandler):
    @staticmethod
    def bound(box, shape):
        boundaryPaths = shape.paths
        boundary_points = []

        for i in boundaryPaths.paths:
            if type(i) == ezdxf.entities.hatch.PolylinePath:
                pass
            if type(i) == ezdxf.entities.hatch.EdgePath:
                for edge in i.edges:
                    if type(edge) == ezdxf.entities.hatch.LineEdge:
                        boundary_points.append((edge.start[0], edge.start[1]))
                ###LINGRUI###
                if len(boundary_points) > 0:
                    boundary_points.append(boundary_points[0])
        for i in boundary_points:
            expandFloat(box,ezdxf.math.Vector(i))
        pass

    @staticmethod
    def render(canvas, shape):

        with canvas.style(lineColor=ShapeHandler.color(shape)):

            boundaryPaths = shape.paths
            boundary_points = []

            for i in boundaryPaths.paths:
                if type(i) == ezdxf.entities.hatch.PolylinePath:
                    pass
                if type(i) == ezdxf.entities.hatch.EdgePath:
                    for edge in i.edges:
                        if type(edge) == ezdxf.entities.hatch.LineEdge:
                            boundary_points.append((edge.start[0], edge.start[1]))
                    #####LINGRUI####
                    if len(boundary_points) > 0:
                        boundary_points.append(boundary_points[0])
            if boundary_points != []:
                canvas.hatch(boundary_points)
        pass


class InsertHandler(ShapeHandler):

    @staticmethod
    def bound(box, block):
        try:
            for shape in block.virtual_entities(True):
                handler = SHAPE_HANDLERS.get(type(shape), None)
                if handler is None:
                    MISSING_HANDLERS.add(type(shape))
                    continue
                handler.bound(box, shape)
        except:
            traceback.print_exc()
            pass
        pass

    @staticmethod
    def render(canvas, block):
        with canvas.style(lineColor=ShapeHandler.color(block)):
            try:
                #for shape in block.virtual_entities(True):
                for shape in block.virtual_entities(True):
                    handler = SHAPE_HANDLERS.get(type(shape), None)
                    if handler is None:
                        MISSING_HANDLERS.add(type(shape))
                        continue
                    #assert not handler is InsertHandler
                    handler.render(canvas, shape)
                    pass
            except:
                traceback.print_exc()
                pass
        pass

    def tostr(shape):
        insert_block = shape.drawing.blocks.get(shape.dxf.name)  # 插入的实体类型
        block_point =  insert_block.block.dxf.base_point # block 基点
        #return 'Insert has insert point %s with angle and scaled at %s %s with blocks %s starting from %s ' %(shape.dxf.insert.vec2,shape.dxf.rotation,shape.dxf.xscale, insert_block, block_point)
        return 'Insert has insert point %s with angle and scaled at %s %s' %(shape.dxf.insert.vec2,shape.dxf.rotation,shape.dxf.xscale)
        pass

'''
TODO: 需要支持如下形状
MISSING HANDLER <class 'ezdxf.entities.circle.Circle'> √
MISSING HANDLER <class 'ezdxf.entities.ellipse.Ellipse'>√
MISSING HANDLER <class 'ezdxf.entities.arc.Arc'>√
MISSING HANDLER <class 'ezdxf.entities.lwpolyline.LWPolyline'>√
MISSING HANDLER <class 'ezdxf.entities.insert.Insert'> 注意Insert需要递归
下面这些暂时可以不考虑
MISSING HANDLER <class 'ezdxf.entities.mtext.MText'>
MISSING HANDLER <class 'ezdxf.entities.solid.Solid'>
MISSING HANDLER <class 'ezdxf.entities.text.Text'>
MISSING HANDLER <class 'ezdxf.entities.hatch.Hatch'>
MISSING HANDLER <class 'ezdxf.entities.dimension.Dimension'>
'''

SHAPE_HANDLERS[ezdxf.entities.Line] = LineHandler
SHAPE_HANDLERS[ezdxf.entities.circle.Circle] = CircleHandler
SHAPE_HANDLERS[ezdxf.entities.ellipse.Ellipse] = EllipseHandler
SHAPE_HANDLERS[ezdxf.entities.arc.Arc] = ArcHandler
SHAPE_HANDLERS[ezdxf.entities.lwpolyline.LWPolyline] = LWPolylineHandler
SHAPE_HANDLERS[ezdxf.entities.polyline.Polyline] = LWPolylineHandler
SHAPE_HANDLERS[ezdxf.entities.insert.Insert] = InsertHandler
SHAPE_HANDLERS[ezdxf.entities.solid.Solid] = SolidHandler
#SHAPE_HANDLERS[ezdxf.entities.hatch.Hatch] = HatchHandler


# SHAPE_HANDLERS[ezdxf.entities.mtext.MText] = MTextHandler
# SHAPE_HANDLERS[ezdxf.entities.text.Text] = TextHandler
# SHAPE_HANDLERS[ezdxf.entities.text.Text] = TextHandler
# SHAPE_HANDLERS[ezdxf.entities.multileader.MULTILEADER] = LeaderHandle

if __name__ == '__main__':
    dr = Drawing('test.dxf')
    print(dr.bbox())
    cvs = CvCanvas(dr.bbox(), 1024)
    dr.render(cvs)
    cvs.save('a.png')

    for x in MISSING_HANDLERS:
        print("MISSING HANDLER", x)
        pass

    boxes = dr.segment()
    for box in boxes:
        print(box)
    pass
