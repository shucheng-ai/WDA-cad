#pragma once
#include <array>
#include <vector>
#include <fstream>
#include <sstream>
#include <list>
#include <queue>
#include <stack>
#include <map>
#include <limits>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <algorithm>
#include <sys/stat.h>
#include <glog/logging.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace beaver {

    namespace py = pybind11;
    using std::string;
    using std::array;
    using std::vector;
    using std::deque;
    using std::queue;
    using std::list;
    using std::map;
    using std::function;
    using std::unique_ptr;
    using std::numeric_limits;
    using std::tuple;

    // D维点
    template <int D> using Size_ = array<int, D>;
    template <int D> using Point_ = array<int, D>;

    // return a - b
    template <int D> static inline
        Size_<D> sub (Point_<D> const &a, Point_<D> const &b) {
            Size_<D> sz;
            for (int i = 0; i < D; ++i) {
                sz[i] = a[i] - b[i];
            }
            return sz;
        }

    // return a + b
    template <int D> static inline
        Point_<D> add (Point_<D> const &a, Size_<D> const &sz) {
            Point_<D> b;
            for (int i = 0; i < D; ++i) {
                b[i] = a[i] + sz[i];
            }
            return b;
        }

    template <int D> static inline
    Size_<D> abs (Size_<D> const &a){
        Size_<D> b;
        for (int i = 0; i < D; ++i){
            b[i] = std::abs(a[i]);
        }
        return b;
    }

    typedef Point_<2> Point;
    typedef Size_<2> Size;

    typedef Point_<3> Point3D;
    typedef Size_<3> Size3D;

    typedef vector<Point> Polygon;

    // scale是线性尺度，毫米转米，scale = 1.0/1000
    double polygon_area (Polygon const &, double scale);

    template <int D>
        struct Box_: public array<Point_<D>, 2> {
            // Box[0]: min/下界
            // Box[1]: max/上界
            static const int DIM = D;

            Size_<D> size () const {
                return sub<D>(this->at(1), this->at(0));
            }

            // 返回python列表表示的坐标数组
            py::list unpack () const {
                py::list r;
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < D; ++j) {
                        r.append(this->at(i)[j]);
                    }
                }
                return r;
            }

            bool empty () const {
                for (int i = 0; i < D; ++i) {
                    if (this->at(0)[i] >= this->at(1)[i]) return true;
                }
                return false;
            }

            bool overlap (Box_<D> const &box) const {
                for (int i = 0; i < D; ++i) {
                    int lb = std::max((*this)[0][i], box[0][i]);
                    int ub = std::min((*this)[1][i], box[1][i]);
                    if (lb > ub) return false;
                }
                return true;
            }

            bool intersect (Box_<D> const &box, Box_<D> *out) const {
                // 返回值同overlap
                // 交叉部分返回到out
                bool r = true;
                for (int i = 0; i < D; ++i) {
                    int lb = out->at(0)[i] = std::max((*this)[0][i], box[0][i]);
                    int ub = out->at(1)[i] = std::min((*this)[1][i], box[1][i]);
                    if (lb > ub) r = false;
                }
                return r;
            }

            void expand (int r) {
                for (int i = 0; i < D; ++i) {
                    this->at(0)[i] -= r;
                    this->at(1)[i] += r;
                }
            }

            void expand (Point_<D> const &p) {
                for (int i = 0; i < D; ++i) {
                    if (p[i] < this->at(0)[i]) this->at(0)[i] = p[i];
                    if (p[i] > this->at(1)[i]) this->at(1)[i] = p[i];
                }
            }

            void expand (Box_<D> const &p) {
                expand(p[0]);
                expand(p[1]);
            }

            void shrink (Box_<D> const &p){
                for (int i = 0; i < D; ++i) {
                    if (p[0][i] > this->at(0)[i]) this->at(0)[i] = p[0][i];
                    if (p[1][i] < this->at(1)[i]) this->at(1)[i] = p[1][i];
                }
            }

            Box_ () {
                std::fill(this->at(0).begin(), this->at(0).end(), numeric_limits<int>::max());
                std::fill(this->at(1).begin(), this->at(1).end(), numeric_limits<int>::min());
            }

            Box_ (Point_<D> const &p1, Point_<D> const &p2): Box_() {
                expand(p1);
                expand(p2);
            }

            template <int D2>
                 Box_ (Box_<D2> const &from) {
                    for (int p = 0; p < 2; ++p) {
                        for (int i = 0; i < std::min(D, D2); ++i) {
                            this->at(p)[i] = from.at(p)[i];
                        }
                    }
                    if (D > D2) {
                        std::fill(this->at(0).begin() + D2, this->at(0).end(), numeric_limits<int>::max());
                        std::fill(this->at(1).begin() + D2, this->at(1).end(), numeric_limits<int>::min());
                    }
                }

            // 在最后增加一个维度
            Box_<D+1> lift (int zmin, int zmax) const {
                Box_<D+1> box(*this);
                box[0][D] = zmin;
                box[1][D] = zmax;
                return box;
            }

            // 去掉最后一个维度(z)
            Box_<D-1> drop () const {
                return Box_<D-1>(*this);
            }

            // 去掉维度d
            Box_<D-1> drop (int d) const {
                Box_<D-1> b;
                for (int i = 0, j = 0; i < D; ++i) {
                    if (i != d) {
                        b[0][j] = (*this)[0][i];
                        b[1][j] = (*this)[1][i];
                        ++j;
                    }
                }
                return b;
            }

            Point_<2> bl () const {
                return Point_<2>{this->at(0)[0], this->at(0)[1]};
            }

            Point_<2> br () const {
                return Point_<2>{this->at(1)[0], this->at(0)[1]};
            }

            Point_<2> tl () const {
                return Point_<2>{this->at(0)[0], this->at(1)[1]};
            }

            Point_<2> tr () const {
                return Point_<2>{this->at(1)[0], this->at(1)[1]};
            }


            Point norm (Point const &p, Size sz) const {
                return Point{(p[0] - this->at(0)[0]) * sz[0] / (this->at(1)[0] - this->at(0)[0]),
                    (p[1] - this->at(0)[1]) * sz[1] / (this->at(1)[1] - this->at(0)[1])};
            }

            py::object python () const {
                return py::cast(new Box_<D>(*this));
            }
        };

    typedef Box_<0> Box0D;
    typedef Box_<2> Box;
    typedef Box_<3> Box3D;

    inline Box make_box (int x1, int y1, int x2, int y2) {
        return Box{{x1, y1}, {x2, y2}};
    }

    inline float area_m2 (Size sz) {
        return (sz[0] / 1000.0) * (sz[1] / 1000.0);
    }

    inline float area_m2 (Box box) {
        return area_m2(box.size());
    }

    inline int64_t area (Box box) {
        Size sz = box.size();
        return int64_t(sz[0]) * sz[1];
    }

    /*
    inline double horizontal_col(Point const &x, Point const &y, int pos){
        double per = double(pos - x[0]) / (y[0] - x[0]);
        return x[1] * (1 - per) + y[1] * per;
    }
    */

    void exportGeoTypes (py::module &module);

    typedef array<cv::Point, 2> Line;

    // CAD图层中的矢量图
    // 所有矢量图全都转换成线段导入C++进行识别
    // 即是是圆，也会先转换成多边形，再打散成线段
    // 这是为了简化C++处理
    struct Layer {
        vector<Line> lines;
        //vector<cv::Rect> hints;
    };

    // python传入的CAD对象
    // 包括：
    // - 一组图层。python只会传入C++识别的图层
    // - 一组标注。
    struct CAD {
        enum {
            DEFAULT = 0,
            WDAS_WALLS,
            WDAS_PILLARS,
            WDAS_DOORS,
            WDAX_EXITS,             // WDAX_EXITS, WDAX_DOCK(_IN/OUT)
            WDAX_DOCK,              // 从CAD上看都属于WDAS_DOORS图层
            WDAX_DOCK_IN,           // 识别完以后再分出来
            WDAX_DOCK_OUT,
            WDAX_FORKLIFT,              // 叉车

            WDAS_SAFETY_DOORS,
            WDAS_FIRE_HYDRANT,
            WDAS_OBSTACLE,
            WDAS_GUARD,

            WDAX_GUARD_2M,
            WDAX_ACC_GUARD,

            WDAF_PASSAGE,
            WDAF_DOCK,
            WDAF_DOCK_IN,
            WDAF_DOCK_OUT,
            WDAF_MINIROOM,
            NUM_LAYERS
        };
        enum {
            ANNO_HINT_ROI = 0,
            ANNO_HINT_CLEAR,
            ANNO_HINT_DOCK,
            ANNO_HINT_DOCK_IN,
            ANNO_HINT_DOCK_OUT,
            ANNO_HINT_CONNECT,
            ANNO_HINT_DROP_ROOM,
            ANNO_GUARD_OBSTACLE,
            ANNO_GUARD_PASSAGE,
            ANNO_GUARD_MINIROOM,
            ANNO_GUARD_FORKLIFT,        // 叉车
            NUM_ANNO
        };
        array<Layer, NUM_LAYERS> layers; // 图层，每个Layout是一组线段
        array<vector<cv::Rect>, NUM_ANNO> annotations;
                                         // 每个标注是一个矩形
    };

    struct Fixture {
        // 仓间内的各种固定物，主要是障碍物等。
        enum {
            // 这个顺序不能动，如果要加的话加到最后
            // 类型号码和Python接口用到
            WALL = 1,
            DOCK = 2,      // U型的，in/out在一块
            DOOR = 3,       // 走货的门
            EXIT = 4,       // 走人的门
            COLUMN = 5,
            MISC_COLUMN = 6,// 不规则的column, 不能影响layout // wdong:已经不再识别MISC_COLUMN
            SAFETY_DOOR = 7,
            GUARD = 8,      // INVISIBLE obstacles
            OBSTACLE = 9,   // other VISIBLE obstacles
            GUARD_2M = 10,
            ACC_GUARD = 11,
            FIRE_HYDRANT = 12,
            DOCK_IN = 13,
            DOCK_OUT = 14,
            GUARD_PASSAGE = 15,
            CUSTOMIZE = 16,
            FORKLIFT = 17,          // 叉车
        };
    };
}

