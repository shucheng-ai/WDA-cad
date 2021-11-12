#!/usr/bin/env python3
from contextlib import contextmanager
import ezdxf
from canvas import Canvas


class DxfCanvas(Canvas):
    '''把画的内容存成我们内部格式的json'''

    def __init__(self, path=None, version='R2010'):
        super().__init__()
        self.offset = [0,0]
        if path is None:
            self.dxf = ezdxf.new(dxfversion=version)
        else:
            self.dxf = ezdxf.readfile(path)
        self.layers=[]    
        pass

    def addLayer(self, name):
        self.dxf.layers.new(name)  # , dxfattribs={'color': 2})
        pass

    # 用法
    #   with canvas.layer("xxx"):
    #       canvas.line(...)
    #       ...
    #   新建一个layer，并且所有内容都会画至这个layer

    @contextmanager
    def layer(self, name):
        try:
            self.dxf.layers.new(name)
            self.layers.append(name)
            yield None
        finally:
            self.layers.pop()
            pass
        pass
    
    '''
    @contextmanager
    def offset(self,origin):
        try:
            self.offset = origin
        finally:
            self.offset = None
            pass
        pass
    '''

    def dxfattribs(self):
        v = {'color': self.styles[-1].lineColor}
        if len(self.layers) > 0:
            v['layer'] = self.layers[-1]
            pass
        return v
    
    def textattribs(self):
        msg = {'layer': 'TEXTLAYER','height': 60}
        pass
        return msg

    def path (self,points,closed = False):
        if closed:
            points = points + [points[0]]
        points = [[p[0]+self.offset[0],p[1]-self.offset[1]] for p in points]
        self.dxf.modelspace().add_lwpolyline(points, dxfattribs=self.dxfattribs())
        pass

    def arc(self, center, r, start_angle, end_angle):
        # cv2 arc
        self.dxf.modelspace().add_arc(center,r,start_angle,end_angle, dxfattribs=self.dxfattribs())

    def save(self, path):
        self.dxf.set_modelspace_vport(height=100000, center=(0, 0))
        self.dxf.saveas(path)

    def line(self, v1, v2):
        self.dxf.modelspace().add_line(v1, v2, dxfattribs=self.dxfattribs())
        pass

    def circle(self, v1, r):
        self.dxf.modelspace().add_circle(v1, r, dxfattribs=self.dxfattribs())

    def ellipse(self, center, major_axis, ratio, start_param, end_param):
        self.dxf.modelspace().add_ellipse(center, major_axis, ratio, start_param,
                                          end_param, dxfattribs=self.dxfattribs())

    def insert(self,insert):
        self.dxf.modelspace().add_blockref(insert.dxf.name,insert.dxf.insert.vec2,dxfattribs={
            'xscale': insert.dxf.xscale,
            'yscale': insert.dxf.yscale,
            'rotation': insert.dxf.rotation
        })
    def text(self,text,points):
        points[0]+=self.offset[0]
        points[1]-=self.offset[1]-45
        self.dxf.modelspace().add_text(text, dxfattribs=self.textattribs()).set_pos(points, align='MIDDLE_LEFT')

    def leader(self, points):
        points = [[p[0]+self.offset[0],p[1]-self.offset[1]] for p in points]
        self.dxf.modelspace().add_leader(vertices=points)
    
    def hatch(self, points):
        points = [[p[0]+self.offset[0],p[1]-self.offset[1]] for p in points]
        fillColor = self.styles[-1].fillColor
        if fillColor==None:
            fillColor = self.styles[-1].lineColor
        fill = self.dxf.modelspace().add_hatch(color=fillColor)
        fill.paths.add_polyline_path(points)

    pass
