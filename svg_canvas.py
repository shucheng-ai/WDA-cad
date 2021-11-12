import svgwrite
#from svgwrite import mm
from canvas import Canvas, TABLEAU20

class SvgCanvas(Canvas):

    def __init__ (self, bbox):
        super().__init__()
        x0, y0, x1, y1 = bbox.unpack()
        x0 -= 2000
        y0 -= 2000
        x1 += 2000
        y1 += 2000
        self.scale = 100
        self.bbox= (x0, y0, x1, y1)
        ss = self.scale
        self.dwg = svgwrite.Drawing(size=(int(x1-x0)//ss, int(y1-y0)//ss))
        self.palette = TABLEAU20
        pass

    def lineColor (self):
        '''获取当前应该用的颜色, [b,g,r]'''
        b, g, r = self.palette[self.styles[-1].lineColor]
        return 'rgb(%d,%d,%d)' % (r, g, b)

    def fillColor (self):
        '''获取当前应该用的颜色, [b,g,r]'''
        if self.styles[-1].fillColor is None:
            return None
        return self.palette[self.styles[-1].fillColor]

    def path (self,points,closed = False):
        """
        多个点构成的折线
        :param points: 多个点  [(x1,y1),(x2,y2)]
        :param closed: 图形是否闭合
        """
        if len(points) == 0:
            return
        x0, y0, _, _ = self.bbox
        pts = []
        ss = self.scale
        for x, y in points:
            pts.append((int(x-x0)//ss, int(y-y0)//ss))
            pass
        if closed:
            pts.append(pts[0])

        self.dwg.add(svgwrite.shapes.Polyline(pts, stroke=self.lineColor(), fill='none'))
        pass

    def arc (self,center,radius, angle,start_angle, end_angle,shift=0):
        """
        圆弧（可实现 圆 、 椭圆 、 圆弧等）
        :param center: 中心
        :param radius: 半径 格式为（r1,r2),r1为半长轴，r2为半短轴。若需绘制图形为圆，则r1=r2
        :param angle: 旋转的角度 顺时针
        :param start_angle: 开始角度
        :param end_angle: 结束角度
        :param shift: 线宽 -1填充图形 默认0
        """
        pass

    def save (self, path):
        self.dwg.saveas(path)

    pass
