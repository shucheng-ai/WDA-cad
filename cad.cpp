#include <map>
#include <tuple>
#include <boost/format.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/algorithms/intersection.hpp>
#include "cad.h"

namespace beaver {
    using std::map;
    using boost::format;
    using std::pair;
    using std::tuple;
    using std::get;
    using std::make_pair;
    using std::make_tuple;

    static const int CC_RELAX = 5000;
    static const int OBJECT_RELAX = 10; // 1cm
    static const int OBJECT_ROOM_TEST_RELAX = 500;
    static const int ROOM_SIZE_MAX_THRESHOLD = 20000;
    static const int ROOM_SIZE_MIN_THRESHOLD = 15000;
    static const int CONNECT_TH = 8000; // 2米
    static const int CONNECT_TH_TINY = 500; // 2米
    static const int CONNECT_TH_SMALL = 2000; // 2米
    static const int CONNECT_TH_MEDIUM = 8000; // 8米   sample-5xy
    static const int CONNECT_TH_LONG = 10000; // 8米
    static const int MERGE_GAP_TH = 50;
    static const float PARALLEL_TH = 0.5;
    static const int OPEN_KERNEL = 5;
    static const int RANK_GROUP_TH = 10000;
    static const int DOOR_EXIT_THRESHOLD = 2800;
    static const int EXTEND_LINE_THRESHOLD = 1000;
    static const float EXTEND_RATIO_TH = 0.3;
    static const int EXTEND_LINE = 2000;
    static const int FIRE_HYDRANT_PILLAR_GAP = 500;
    static const int FIRE_ACC_GUARD_SIZE = 1500;
    static const int FIRE_GUARD_2M_SIZE = 1000;
    static const int ATTRACT_ASPECT_RATIO = 20;
    static const int ATTRACT_TOLERATE = 5;
    static const int MAX_NON_REAL_EXTEND = 1;
    static const int MIN_DUAL_EXTEND = 2000;
    static const float POPULAR_VALUE_TH = 0.3;
    static const int POPULAR_TH = 3;
    static const int ALIGN_TO_ROOM_RELAX = 1500;
    static const int WALL_GUARDS_DIST = 500;
    static const int WALL_GUARDS_TRIANGULATE = 200;

    typedef vector<cv::Point> Contour;

    struct Object {
        cv::Rect bbox;
        Contour contour;

        cv::Point center () const {
            return cv::Point(bbox.x + bbox.width/2, bbox.y + bbox.height/2);
        }
    };


    typedef vector<Contour> Contours;
    typedef vector<Object> Objects;
    typedef array<Objects, CAD::NUM_LAYERS> ObjectLayers;

    class ImageLog {
        string root;
    public:
        ImageLog (string const &r = ""): root(r) {
            if (root.size()) {
                system((format{"mkdir -p \"%1%\""} % root).str().c_str());
            }
        }

        bool enabled () const {
            return root.size() > 0;
        }

        void log (cv::Mat image, string const &path) const {
            if (root.size()) {
                cv::imwrite((format{"%s/%s"} % root % path).str(), image);
            }
        }

        void log (cv::Mat image, boost::basic_format<char> const &path) const {
            if (root.size()) {
                cv::imwrite((format{"%s/%s"} % root % path.str()).str(), image);
            }
        }
    } ilog("ilog");

    class RasterCanvas {
        cv::Rect bb;
        int scale_num;
        int scale_denom;
        int pad;
    public:
        cv::Mat image;

        RasterCanvas (cv::Rect bb_, int size, int pad_ = 0, int ctype = CV_8U): bb(bb_), pad(pad_) {
            scale_num = size - 1 - 2 * pad;
            scale_denom = std::max(bb.width, bb.height);
            int rows = (int64_t(bb.height) * scale_num + scale_denom - 1) / scale_denom + 1 + pad * 2;
            int cols = (int64_t(bb.width) * scale_num + scale_denom - 1) / scale_denom + 1 + pad * 2;
            image = cv::Mat(rows, cols, ctype, cv::Scalar::all(0));
        }

        int scale (int v) const {
            return int64_t(v) * scale_num / scale_denom;
        }

        int unscale (int v) const {
            return int64_t(v) * scale_denom / scale_num;
        }

        void map (cv::Point *pt) const {
            pt->x = scale(pt->x - bb.x) + pad;
            pt->y = image.rows - pad - scale(pt->y - bb.y);
        }

        void map (cv::Rect *rect) const {
            cv::Point tl = rect->tl(), br = rect->br();
            map(&tl); map(&br);
            std::swap(tl.y, br.y);
            *rect = cv::Rect(tl, br);
        }

        void unmap (cv::Point *pt) const {
            pt->x = bb.x + unscale(pt->x - pad);
            pt->y = bb.y + unscale(image.rows - pad - pt->y);

        }

        void draw (cv::Rect const &r, cv::Scalar const &color = cv::Scalar(1), int thickness= 1) {
            cv::Rect r2 = r;
            map(&r2);
            cv::rectangle(image, r2, color, thickness);
        }

        void draw (cv::Point const &pt, int r, cv::Scalar const &color = cv::Scalar(1), int thickness= 1) {
            cv::Point p = pt;
            map(&p);
            cv::circle(image, p, r, color);
        }

        void draw (Line const &l, cv::Scalar const &color = cv::Scalar(1), int thickness = 1) {
            cv::Point p1 = l[0];
            cv::Point p2 = l[1];
            map(&p1); map(&p2);
            cv::line(image, p1, p2, color, thickness);
        }

        void draw (Layer const &layer, cv::Scalar const &color = cv::Scalar(1), int thickness=1) {
            for (auto const &l: layer.lines) {
                draw(l, color, thickness);
            }
        }

        void draw (CAD const &cad, cv::Scalar const &color = cv::Scalar(1), int thickness = 1) {
            for (auto const &l: cad.layers) {
                draw(l, color, thickness);
            }
        }
    };

    // 之所以存在，是为了方便计算两个不同类型的Point的距离
    template <typename T1, typename T2>
    double norm (T1 const &p1, T2 const &p2) {
        cv::Point2d d(double(p1.x)-p2.x, double(p1.y)-p2.y);
        return std::sqrt(d.x * d.x + d.y * d.y);
    }

    template <typename T1, typename T2>
    double dot (T1 const &p1, T2 const &p2) {
        return double(p1.x) * p2.x + double(p1.y) * p2.y;
    }

    // 故意用整数端点
    // 两个端点必须不一样, 构建对象前需要验证
    // 0度，90度是严格计算的(通过判断x或者y是否相等, 不会有舍入误差
    // 别的情况不一定
    // 如果直线是水平，则必须是0度，不会是180度
    // angle in [0, 180)

    struct XLine: Line {
        Line core;
        bool real;    // 点是在实际结构上的
        float length;
        float angle;  // digree, so we can do exact match
                      // 0 - 179
        bool update1 = false;
        bool update2 = false;
        XLine () {}

        XLine (cv::Point p1, cv::Point p2, bool r = true): real(r) {
            cv::Point d = p2 - p1;          // angle: [-pi : pi]
            if (d.y == 0) {
                CHECK(p2.x != p1.x);        // p1 and p2 cannot be equal
                if (p1.x > p2.x) {          // p1 -> p2
                                            // p2.x is bigger
                    std::swap(p1, p2);
                }
                angle = 0;
            }
            else {
                if (d.y < 0) {  
                    std::swap(p1, p2);
                    d = p2 - p1;
                }                               // angle: [0: pi]
                // p2 on top of p1
                if (d.x == 0) {
                    angle = 90;
                }
                else {
                    double r = std::atan2(d.y, d.x);
                    angle = 180 * r / M_PI;
                    CHECK(angle >= 0);
                    CHECK(angle < 180);
                }
            }
            at(0) = p1;
            at(1) = p2;
            core[0] = p1;
            core[1] = p2;
            update_length();
        }

        explicit XLine (Line const &ln, bool r = true): XLine(ln[0], ln[1], r) {}

        void update_length () {
            length = beaver::norm(at(0), at(1));
        }

        // 长度
        double norm () const {
            return length;
        }

        float extend_th (float th) const {
            return std::min(th, length * EXTEND_RATIO_TH);
        }
    };

    cv::Point2d direction (cv::Point const &p1, cv::Point const &p2) {
        cv::Point2d d(p2-p1);
        d /= cv::norm(d);
        return d;
    }

    bool parallel (XLine const &l1, XLine const &l2) {
        float a = l1.angle;
        float b = l2.angle;
        if (a > b) std::swap(a, b);
        // a <= b
        float d = std::min(b - a, a+180-b);
        return d <= PARALLEL_TH;
    }

    cv::Rect relax_box (cv::Point p1, cv::Point p2, int w) {
        std::pair<int,int> x = std::minmax(p1.x, p2.x);
        std::pair<int,int> y = std::minmax(p1.y, p2.y);
        x.first -= w; x.second += w;
        y.first -= w; y.second += w;
        return cv::Rect(x.first, y.first, x.second - x.first, y.second - y.first);
    }

    cv::Rect relax_box (cv::Point p, int w) {
        return cv::Rect(p.x - w, p.y - w, 2*w+1, 2*w+1);
    }

    cv::Rect relax_box (cv::Rect p, int w) {
        return cv::Rect(p.x-w, p.y-w, p.width+2*w, p.height+2*w);
    }

    void box2contour (cv::Rect const &r, Contour *c) {
        c->clear();
        c->push_back(cv::Point(r.x, r.y));
        c->push_back(cv::Point(r.x + r.width, r.y));
        c->push_back(cv::Point(r.x + r.width, r.y + r.height));
        c->push_back(cv::Point(r.x, r.y + r.height));
    }

    // 提取连续部分
    void extract_cc (CAD const &cad, vector<cv::Rect> *rects_, int cc_relax, int pick_layer = -1) {
        // if pick_layers >= 0, only use this layer
        vector<cv::Rect> rects;
        for (int i = 0; i < cad.layers.size(); ++i) {
            if ((pick_layer >= 0) && (i != pick_layer)) continue;
            auto const &layer = cad.layers[i];
            for (auto const &line: layer.lines) {
                cv::Rect r = relax_box(line[0], cc_relax) | relax_box(line[1], cc_relax);
                int j = 0;
                while (j < rects.size()) {
                    cv::Rect one = rects[j];
                    if ((one & r).width > 0) {     // overlap
                        r |= one;
                        rects[j] = rects.back();
                        rects.pop_back();
                    }
                    else {
                        ++j;
                    }
                }
                rects.push_back(r);
            }
        }
        for (;;) {
            int i = 0;
            int cc = 0;
            while (i < rects.size()) {
                cv::Rect &r = rects[i];
                int j = i+1;
                while (j < rects.size()) {
                    cv::Rect one = rects[j];
                    if ((one & r).width > 0) {     // overlap
                        r |= one;
                        rects[j] = rects.back();
                        rects.pop_back();
                        ++cc;
                    }
                    else {
                        ++j;
                    }
                }
                ++i;
            }
            if (cc == 0) break;
        }
        rects_->swap(rects);
    }

#if 0
namespace method1 {
    typedef array<int, 3> SpecialLine;
    // 三个数字分别是
    // 垂直线段 x, y1, y2
    // 水平线段 y, x1, x2
    //
    static void split_groups (vector<SpecialLine> const &ll,
                vector<vector<SpecialLine>> *rr, int th = 1) {
        rr->clear();
        int begin = 0;
        while (begin < ll.size()) {
            int end = begin + 1;
            rr->emplace_back();
            while ((end < ll.size()) && (ll[end][0] - ll[begin][0] < th)) {
                ++end;
            }
            for (int i = begin; i < end; ++i) {
                rr->back().push_back(ll[i]);
            }
            begin = end;
        }
    }

    // python-style sort
    template <typename A, typename B>
    void sort (A *v, B key) {
        std::sort(v->begin(), v->end(),
                [&key](typename A::value_type const &a,
                       typename A::value_type const &b) {
                    return key(a) < key(b);
                });
    }

    Line extend (Line const &l) {
        cv::Point d = l[1] - l[0];
        int r = std::sqrt(d.x * d.x + d.y * d.y);
        if (r < EXTEND_LINE_THRESHOLD) return l;
        d.x = d.x * EXTEND_LINE / r;
        d.y = d.y * EXTEND_LINE / r;
        return Line{l[0] - d, l[1] + d};
    }

    int merge_lines (vector<SpecialLine> &lines) {
        // 按第一个下标排序
        // SpecialLine三个下标分别为 x, y1, y2; y1 < y2  (假设直线竖直，水平的话就是xy互换)
        // 所有x都一样
        int cnt = 0;
        CHECK(lines.size());
        sort(&lines, [](SpecialLine const &l) {return l[1];});
        int o = 0;  // 当前线段，用于输出
        for (int i = 1; i < lines.size(); ++i) {
            if (lines[i][1] - lines[o][2] < CONNECT_TH) {
                // 连接，扫描下一个
                lines[o][2] = std::max(lines[o][2], lines[i][2]);
                ++cnt;
            }
            else {
                // 无法连接
                ++o;
                lines[o] = lines[i];
            }
        }
        lines.resize(o+1);
        return cnt;
    }

    int calculate_cross (int r, int r1, int r2, int t1, int t2) {
        // r1 <= r <= r2
        // r, t按比例，计算t应该在的位置
        if (r2 <= r1) return (t1 + t2)/2;
        CHECK(r >= r1);
        CHECK(r2 >= r);
        return t1 + (t2-t1) * (r - r1) / (r2 - r1);
    }

    // TODO
    void connect_lines (vector<Line> const &lines,
                        ObjectLayers const &layers,
                        vector<Line> *out) {
        out->clear();
        // 先做一个特殊的方法，只处理0度和90度的
        vector<SpecialLine> dirs[2]; // tuple<offset, index>
        for (int i = 0; i < lines.size(); ++i) {
            Line const &l = lines[i];
            int cc = 0;
            if (l[0].y == l[1].y) {
                ++cc;
                dirs[0].emplace_back(SpecialLine{l[0].y, std::min(l[0].x, l[1].x), std::max(l[0].x, l[1].x)});
            }
            if (l[0].x == l[1].x) {
                ++cc;
                dirs[1].emplace_back(SpecialLine{l[0].x, std::min(l[0].y, l[1].y), std::max(l[0].y, l[1].y)}); 
            }
            if (cc == 0) {
                // 如果是斜边，需要适当延长两边后加入
                out->push_back(extend(l)); // 否则不予连接直接输出
            }
        }
        sort(&dirs[0], [](SpecialLine const &l) { return l[0];});
        sort(&dirs[1], [](SpecialLine const &l) { return l[0];});
        vector<vector<SpecialLine>> groups[2];
        split_groups(dirs[0], &groups[0]);
        split_groups(dirs[1], &groups[1]);

        for (int i = 0; i < 2; ++i) {
            int j = 1-i;
            // i是一个方向，j是另一个方向
            for (auto const &ll: groups[i]) {
                // 对于一个方向里的每个条线段l
                for (auto const &l: ll) {
                    // 如果线段其实是一个点则略过
                    if (l[1] >= l[2]) continue;
                    // 对于另一方向里的每一组g
                    for (auto &g: groups[j]) {
                        int m = g[0][0];
                        // 如果线段跨越g2的位置
                        if ((m >= l[1] - EXTEND_LINE) && (m <= l[2] + EXTEND_LINE)) {
                            g.push_back(SpecialLine{m, l[0], l[0]});
                        }
                    }
                }
            }
        }

        for (Line const &l: *out) { // 斜线也需要添加点，不然有可能有的边界连不上
            Line lx = l;    // 两个点按x从小到大排
            Line ly = l;    //         y

            if (lx[0].x > lx[1].x) std::swap(lx[0], lx[1]);
            if (ly[0].y > ly[1].y) std::swap(ly[0], ly[1]);

            for (int i = 0; i < 2; ++i) {
                // 对于每个方向的每个group
                for (auto &g: groups[i]) {
                    int m = g[0][0];
                    if (i == 0) {
                        // g中所有元素y一样，x不同, m是y
                        if (ly[0].y <= m && ly[1].y >= m) {
                            // 加入x位置
                            int cross = calculate_cross(m, ly[0].y, ly[1].y, ly[0].x, ly[1].x);
                            g.push_back(SpecialLine{m, cross, cross});
                        }
                    }
                    else {
                        // g中所有元素x一样，y不同, m是x
                        if (lx[0].x <= m && lx[1].x >= m) {
                            int cross = calculate_cross(m, lx[0].x, lx[1].x, lx[0].y, lx[1].y);
                            g.push_back(SpecialLine{m, cross, cross});
                        }
                    }
                }
            }
        }


        for (auto &ll: groups[0]) merge_lines(ll);
        for (auto &ll: groups[1]) merge_lines(ll);

        /*
        for (int it = 0;; ++it) {    // 循环三次
            int cc = 0;
            for (auto &ll: groups[0]) cc += merge_lines(ll);
            for (auto &ll: groups[1]) cc += merge_lines(ll);
            if (it >= 2) break;
            // 交叉填充
        }
        */
        for (auto const &ll: groups[0]) {   // y 一样
            for (auto const &l: ll) {
                if (l[1] >= l[2]) continue;
                out->emplace_back(Line{cv::Point(l[1], l[0]), cv::Point(l[2], l[0])});
            }
        }
        for (auto const &ll: groups[1]) {   // x 一样
            for (auto const &l: ll) {
                if (l[1] >= l[2]) continue;
                out->emplace_back(Line{cv::Point(l[0], l[1]), cv::Point(l[0], l[2])});
            }
        }
    }

    void merge_layers (Layer *to, Layer const &from) {
        to->lines.insert(to->lines.end(),
                         from.lines.begin(), from.lines.end());
    }
}
#endif

    namespace method2 { // preferrable

        /*
        void collect_aux_lines (ObjectLayers const &layers, vector<XLine> *aux) {
            aux->clear();
            for (int l: vector<int>{CAD::WDAS_DOORS,
                                    CAD::WDAS_SAFETY_DOORS} ) {
                for (auto const &obj: layers[l]) {
                    append_qlines(aux, l, obj.bbox, CONNECT_TH1);
                }
            }
        }

        double determant (Line const &l1, Line const &l2) {
            Point2d a(l1[1] - l1[0]);
            Point2d b(l2[1] - l2[0]);
            return a.x * b.y - a.y * b.x;
        }

        bool parallel (Line const &l1, Line const &l2) {

        }
        */

        // 点到线段所在直线的距离
        double distance (XLine const &l, cv::Point p) {
            cv::Point2d dir(direction(l[0], l[1]));
            cv::Point2d x(p - l[0]);
            double a = cv::norm(x);
            double b = dot(dir, x);
            return std::sqrt(a * a - b * b);
        }

        bool connect_parallel (XLine l1, XLine l2, int th, XLine *conn) {
            if (!parallel(l1, l2)) return false;
            // try connect
            double n1 = l1.norm();
            double n2 = l2.norm();
            if (n1 < n2) {
                std::swap(l1, l2);
                std::swap(n1, n2);
            }
            // l1 is longer than (or equal to) l2
            double best = -1;
            int best1 = -1;
            int best2 = -1;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    double n = norm(l1[i], l2[j]);
                    if (n > best) {
                        best = n;
                        best1 = i;
                        best2 = j;
                    }
                }
            }
            CHECK(best1 >= 0 && best2 >= 0);
            if (best <= n1) {   // l1 covers l2, succeeed
                if (!(distance(l1, l2[0]) < MERGE_GAP_TH)) return false;
                if (!(distance(l1, l2[1]) < MERGE_GAP_TH)) return false;
                *conn = l1;
                return true;
            }
            if (best < 1) return false; // connected line is too short
            // now the outer points are l1[0] and l2[1]
            Line c{l1[best1], l2[best2]};
            XLine cx(c); // direction of cx might change, that's why we need c
            // the connected line must be parallel to both components
            if (!parallel(cx, l1)) return false;
            if (!parallel(cx, l2)) return false;

            if (!(distance(cx, l1[0]) < MERGE_GAP_TH)) return false;
            if (!(distance(cx, l1[1]) < MERGE_GAP_TH)) return false;
            if (!(distance(cx, l2[0]) < MERGE_GAP_TH)) return false;
            if (!(distance(cx, l2[1]) < MERGE_GAP_TH)) return false;

            // inner points are l1[1] and l2[0]
            cv::Point2d dir(direction(c[0], c[1]));
            double o1 = dot(dir, l1[1-best1] - c[0]); // , n1, l1.norm());
            double o2 = dot(dir, l2[1-best2] - c[0]);
            if (std::abs(o2 - o1) > th) return false;
            *conn = cx;
            return true;
        }

        bool connect_strict_parallel (XLine l1, XLine l2, int th, XLine *conn) {
            cv::Point2d v1 = l1[1] - l1[0];
            cv::Point2d v2 = l2[1] - l2[0];
            double cross = v1.x * v2.y - v2.x * v1.y;
            if (cross != 0) return false;
            return connect_parallel(l1, l2, th, conn);
        }


        cv::Point2d intersect (XLine const &a, XLine const &b) {
            cv::Point2d da = a[1] - a[0];
            cv::Point2d db = b[1] - b[0];
            double u = a[0].x * da.y - a[0].y * da.x;
            double v = b[0].x * db.y - b[0].y * db.x;
            double bottom = da.y * db.x - da.x * db.y;
            CHECK(std::abs(bottom) > 1);
            double x = (u * db.x - v * da.x) / bottom;
            double y = (u * db.y - v * da.y) / bottom;
            return cv::Point2d(x, y);
        }

        struct Extend {
            // 在线段所在的直线上找一个点
            // 不管这个点在哪儿，都是可以试图延长这个线段直到这个点的

            // hit表明点的距离是否足够近
            bool hit = false;   
            // side表明点是否在延长线上，而不在线段内部
            int side = -1;
                        // -1 not hit
                        // 0 extend point 0
                        // 1 extend point 1
                        // 2 inside, do not extend
            cv::Point pt;
            bool real = true;

            // 如果pt在线段之外则硬延长返回true
            // 否则返回false, 和hit无关
            bool apply (XLine *line) const {
                //if (!hit) return false;
                if (side == 0) {
                    line->at(0) = pt;
                    if (real) line->core[0] = pt;
                    line->update_length();
                    return true;
                }
                else if (side == 1) {
                    line->at(1) = pt;
                    if (real) line->core[1] = pt;
                    line->update_length();
                    return true;
                }
                return false;
            }
        };

        Extend try_extend (XLine const &line, cv::Point2d const &pt, bool real, int th, int tip = 3) {
            // real 表示点pt是在实际结构上的
            cv::Point2d dir(line.core[1]-line.core[0]);
            double l = norm(dir);
            dir /= l;

            double lb = dot(dir, cv::Point2d(line[0]) - cv::Point2d(line.core[0]));
            double rb = dot(dir, cv::Point2d(line[1]) - cv::Point2d(line.core[0]));

            double x = dot(dir, cv::Point2d(pt - cv::Point2d(line.core[0])));

            Extend ext;
            ext.real = real;
            if (x < lb) {
                if (x >= -th) {
                    ext.hit = true;
                }
                x -= tip;
                ext.side = 0;
                cv::Point2d delta = x * dir;
                ext.pt = cv::Point(round(line.core[0].x + delta.x),
                                   round(line.core[0].y + delta.y));
            }
            else if (x > rb) {
                if (x <= l + th) {
                    ext.hit = true;
                }
                x += tip;
                ext.side = 1;
                x -= l;
                cv::Point2d delta = x * dir;
                ext.pt = cv::Point(round(line.core[1].x + delta.x),
                                   round(line.core[1].y + delta.y));
            }
            else if ((x >= lb) && (x <= rb)) {
                ext.hit = true;
                ext.side = 2;
            }
            return ext;
        }


        // 用四条线段表出的矩形，不能旋转
        class QLines: public array<XLine, 2>  {
        public:
            QLines (cv::Point const &p1, cv::Point const &p2,
                    cv::Point const &p3, cv::Point const &p4,
                    bool real) {
                at(0) = XLine(p1, p2, real);
                at(1) = XLine(p3, p4, real);
            }

            bool extend (XLine *line, int th) const {
                bool cross = false;
                vector<Extend> es;
                for (auto const &a: *this) {
                    if (parallel(*line, a)) continue;
                    cv::Point2d c = intersect(*line, a);
                    if (try_extend(a, c, true, 0).hit) {  // 必须和框框相交
                        Extend e = try_extend(*line, c, a.real, th);
                        es.push_back(e);
                        cross = cross || e.hit;
                    }
                }
                bool u = false;
                if (cross) {
                    for (auto const &e: es) {
                        u |= e.apply(line);
                    }
                }
                return u;
            }
        };

        void add_qlines (vector<QLines> *qlines, cv::Rect const &r, bool real, int relax) {
                int x1 = r.x;
                int y1 = r.y;
                int x2 = r.x + r.width;
                int y2 = r.y + r.height;    //      p4  p3
                cv::Point p1(x1, y1);       //      
                cv::Point p2(x2, y1);       //      p1  p2
                cv::Point p3(x2, y2);
                cv::Point p4(x1, y2);
                cv::Point dx(relax, 0);
                cv::Point dy(0, relax);
                qlines->emplace_back(p1-dx, p2+dx, p4-dx, p3+dx, real);
                qlines->emplace_back(p1-dy, p4+dy, p2-dy, p3+dy, real);
        }


        void connect_walls (cv::Rect bb, vector<Line> const &lines_in,
                            ObjectLayers const &layers, vector<cv::Rect> const &hint_connect,
                            vector<Line> *out) {

            vector<XLine> lines;
            vector<QLines> auxes; 
            for (auto const &r: hint_connect) {
                add_qlines(&auxes, r, false, 0);
            }
            for (auto const &l: lines_in) {
                if (norm(l[0],l[1]) < 100) {
                    add_qlines(&auxes, relax_box(l[0], l[1], 0), true, CONNECT_TH_TINY);
                }
                else {
                    lines.emplace_back(l);
                }
            }
            for (int l: vector<int>{CAD::WDAS_DOORS,
                                    CAD::WDAS_SAFETY_DOORS} ) {
                for (auto const &obj: layers[l]) {
                    //auxes.emplace_back(relax_box(obj.bbox, CONNECT_TH_SMALL));
                    add_qlines(&auxes, obj.bbox, true, CONNECT_TH_TINY);
                }
            }

            for (auto &line: lines) {
                line.update1 = true;
                line.update2 = false;
            }

            // update1: 当前这次extend是否要考虑延长这根线
            // update2: 新的update1的值，下一轮用; 如果已经测试了无法延长，update2就要变成false
            //          只有延长了的下一次还需要接着测试

            for (;;) {
                int s0 = lines.size();
                for (auto &line: lines) {
                    if (!line.update1) continue;
                    for (auto const &aux: auxes) {
                        //CHECK(aux.size() == 4);
                        if (aux.extend(&line, line.extend_th(CONNECT_TH_MEDIUM))) {
                            line.update2 = true;
                        }
                    }
                }

                // 先链接strictly parallel的
                int strict_parallel_connected = 0;
#if 0
                // 以下代码是正确的，但是bug通过别的途径解决了，不需要这段检查
                int i = 0;
                while (i < lines.size()) {
                    int j = i + 1;
                    while (j < lines.size()) {
                        if (!(lines[i].update1 || lines[j].update1)) {
                            ++j;
                            continue;
                        }
                        // merge
                        XLine c;
                        if (connect_strict_parallel(lines[i], lines[j], CONNECT_TH_MEDIUM, &c)) {
                            CHECK(c.norm() >= lines[i].norm());
                            CHECK(c.norm() >= lines[j].norm());
                            lines[i] = c;
                            lines[i].update1 = lines[i].update2 = true;
                            std::swap(lines[j], lines.back());
                            lines.pop_back();
                            ++strict_parallel_connected;
                            continue;
                        }
                        ++j;
                    }
                    ++i;
                }
#endif

                // 如果有严格平行的被连了，则不要再做更加宽松的测试，
                if (strict_parallel_connected == 0) {
                    int i = 0;
                    while (i < lines.size()) {
                        int j = i + 1;
                        while (j < lines.size()) {
                            if (!(lines[i].update1 || lines[j].update1)) {
                                ++j;
                                continue;
                            }
                            if (parallel(lines[i], lines[j])) {
                                // merge
                                XLine c;
                                if (connect_parallel(lines[i], lines[j], CONNECT_TH_MEDIUM, &c)) {
                                    CHECK(c.norm() >= lines[i].norm());
                                    CHECK(c.norm() >= lines[j].norm());
                                    lines[i] = c;
                                    lines[i].update1 = lines[i].update2 = true;
                                    std::swap(lines[j], lines.back());
                                    lines.pop_back();
                                    continue;
                                }
                            }
                            else if (lines[i].norm() >= MIN_DUAL_EXTEND
                                    && lines[j].norm() >= MIN_DUAL_EXTEND) {
                                cv::Point2d c = intersect(lines[i], lines[j]);
                                Extend e1 = try_extend(lines[i], c, false, lines[i].extend_th(CONNECT_TH_SMALL));
                                Extend e2 = try_extend(lines[j], c, false, lines[j].extend_th(CONNECT_TH_SMALL));
                                if (e1.hit && e2.hit) {
                                    if (e1.apply(&lines[i])) {
                                        lines[i].update2 = true;
                                    }
                                    if (e2.apply(&lines[j])) {
                                        lines[j].update2 = true;
                                    }
                                }
                            }
                            ++j;
                        }
                        ++i;
                    }
                }
                LOG(INFO) << "REDUCE " << s0 << " => " << lines.size();
                int update = 0;
                for (auto &line: lines) {
                    if (line.update2) ++update;
                    line.update1 = line.update2;
                    line.update2 = false;
                }
                if (update == 0) break;
            }
            out->clear();
            for (auto const &l: lines) {
                out->push_back(l);
            }
        }
    }

    cv::Rect bound (Contour const &pts) {
        cv::Rect r(pts[0].x, pts[0].y, 1, 1);
        for (auto const &p: pts) {
            r |= cv::Rect(p.x, p.y, 1, 1);
        }
        return r;
    }

    // counterclockwise, open(first != last)
    // OpenCV's polygon is counterclock-wise outer and clockwise for inner holes
    typedef boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double>, false, false> BGPolygon;


    // scale是线性尺度，平方毫米转平方米，输入scale = 1.0/1000
    double polygon_area (Polygon const &input_poly, double scale) {
        BGPolygon po;
        for (auto const &p: input_poly) {
            boost::geometry::append(po.outer(), boost::geometry::model::d2::point_xy<double>(p[0] * scale, p[1] * scale));
        }
        return std::abs(boost::geometry::area(po));
    }

    void contour2polygon (Contour const &c, BGPolygon *po) {
        for (auto const &p: c) {
            boost::geometry::append(po->outer(), boost::geometry::model::d2::point_xy<double>(p.x, p.y));
        }
    }

    void polygon2contour (BGPolygon const &po, Contour *c) {
        c->clear();
        for (auto const &p: po.outer()) {
            c->emplace_back(int(std::round(p.x())), int(std::round(p.y())));
        }
    }

    void contour_intersect (Object const &o1, Object const &o2, Object *out) {
        BGPolygon p1; contour2polygon(o1.contour, &p1);
        BGPolygon p2; contour2polygon(o2.contour, &p2);
        vector<BGPolygon> outs;
        boost::geometry::intersection(p1, p2, outs);
        BGPolygon best;
        double best_area = 0;   // 找最大的交
        for (auto &p: outs) {
            double a = boost::geometry::area(p);
            if (a > best_area) {
                best_area = a;
                std::swap(best, p);
            }
        }
        out->bbox = cv::Rect();
        out->contour.clear();
        if (best_area > 0) {
            polygon2contour(best, &out->contour);
            out->bbox = bound(out->contour);
        }
    }

    void round_to_group_lb (vector<int> &v) {
        /*
    # 先对输入的数值进行分组
    # 分组方法为：所有数值排序，然后当相邻两个数字之间大于GROUPING_TH是切断
    # 按切断方法对原数组进行分组
    # 然后把原数组所有的数值替换为其所在组最小的值

    # 目的是让一群大致有序的数字，其中差不多的数值变成一样
        */
        vector<pair<int, int>> o;
        for (int i = 0; i < v.size(); ++i) {
            o.push_back(make_pair(v[i], i));
        }
        sort(o.begin(), o.end());
        int begin = 0;
        while (begin < o.size()) {
            int end = begin + 1;
            while (end < o.size() && (o[end].first - o[end-1].first) < RANK_GROUP_TH) {
                ++end;
            }
            for (int i = begin; i < end; ++i) {
                v[o[i].second] = o[begin].first;
            }
            begin = end;
        }
    }

    void sort_rooms (vector<Object> const &rooms, vector<int> *order) {
        vector<int> xs;
        vector<int> ys;
        for (auto const &r: rooms) {
            xs.push_back(r.bbox.x + r.bbox.width/2);
            ys.push_back(r.bbox.y + r.bbox.height/2);
        }
        round_to_group_lb(xs);
        round_to_group_lb(ys);
        vector<tuple<int, int, int>> rank;
        for (int i = 0; i < rooms.size(); ++i) {
            rank.push_back(make_tuple(-ys[i], xs[i], i));
        }
        sort(rank.begin(), rank.end());
        order->clear();
        for (auto t: rank) {
            order->push_back(get<2>(t));
        }
    }

    template <int SIDE>
    int find_popular_value (vector<int> const &a, int min, int max, int th) {
        std::unordered_map<int, int> cnt;
        int cc = 0;
        for (int v: a) {
            int hit = 0;
            for (int i = 0; i < th; ++i) {
                int v2 = v + SIDE * i;
                if (v2 < min) continue;
                if (v2 > max) continue;
                hit += 1;
                cnt[v2] += 1;
            }
            if (hit > 0) ++cc;
        }
        std::pair<int, int> best(-1, -1);
        for (auto p: cnt) {
            if (p.second > best.second) {
                best = p;
            }
        }
        LOG(INFO) << "POPULARITY: " << (1.0 * best.second / cc);
        if (cc == 0) return -1;
        if (cc * POPULAR_VALUE_TH > best.second) return -1;
        return best.first;
    }


    template <int D>
    void check_switch_index (int i, int j, int *pi, int *pj) {
        if (D == 0) {
            *pi = i;
            *pj = j;
        }
        else {
            *pi = j;
            *pj = i;
        }
    }

    template <int D>
    void squeeze (cv::Mat room, vector<int> const &lbs,
                                vector<int> const &ubs,
                                int popular_l, int popular_u, int th) {
        int n = (D == 0) ? room.rows : room.cols;
        CHECK(n == lbs.size() && n == ubs.size());
        int a, b;
        for (int i = 0; i < n; ++i) {
            int lb = lbs[i], ub = ubs[i];
            //   [lb, ub] = 1,   outside are 0
            if ((popular_l >= 0) && (lb > popular_l - th)) {
                for (int j = lb; j < popular_l; ++j) {
                    check_switch_index<D>(i, j, &a, &b);
                    room.at<uint8_t>(a, b) = 0;
                }
            }
            if ((popular_u >= 0) && (ub < popular_u + th)) {
                for (int j = popular_u + 1; j <= ub; ++j) {
                    check_switch_index<D>(i, j, &a, &b);
                    room.at<uint8_t>(a, b) = 0;
                }
            }
        }
    }

    void regularize_room (cv::Mat room, int pixels = 3) {
        CHECK(room.type() == CV_8U);
        vector<int> left(room.rows, room.cols); // x: left -> right
        vector<int> right(room.rows, -1);
        vector<int> top(room.cols, room.rows);  // y: top -> bottom
        vector<int> bottom(room.cols, -1);
        for (int i = 0; i < room.rows; ++i) {
            uint8_t const *row = room.ptr<uint8_t const>(i);
            for (int j = 0; j < room.cols; ++j) {
                if (row[j] == 0) continue;
                if (j < left[i]) left[i] = j;
                if (j > right[i]) right[i] = j;
                if (i < top[j]) top[j] = i;
                if (i > bottom[j]) bottom[j] = i;
            }
        }
        int th = POPULAR_TH;
        int l, u;
        l = find_popular_value<1>(left, 0, room.cols-1, th);
        u = find_popular_value<-1>(right, 0, room.cols-1, th);
        squeeze<0>(room, left, right, l, u, th);
        l = find_popular_value<1>(top, 0, room.rows-1, th);
        u = find_popular_value<-1>(bottom, 0, room.rows-1, th);
        squeeze<1>(room, top, bottom, l, u, th);
    }

    bool test_intersect (Contour const &c1, Object const &obj, int relax) {
        // TODO: 注意：如果c2完全包含c1，则会返回false BUG?
        int x1 = obj.bbox.x - relax;
        int y1 = obj.bbox.y - relax;
        int x2 = obj.bbox.x + obj.bbox.width + relax;
        int y2 = obj.bbox.y + obj.bbox.height + relax;
        if (cv::pointPolygonTest(c1, cv::Point(x1, y1), true) >= 0) return true;
        if (cv::pointPolygonTest(c1, cv::Point(x1, y2), true) >= 0) return true;
        if (cv::pointPolygonTest(c1, cv::Point(x2, y2), true) >= 0) return true;
        if (cv::pointPolygonTest(c1, cv::Point(x2, y1), true) >= 0) return true;
        return false;
    }


    void extract_rooms (CAD const &cad, ObjectLayers const &layers, cv::Rect cc_bb, int cc_id,
                        vector<Object> *results,
                        vector<Contour> *borders,
                        vector<Object> *obstacles) {
        Layer connected;
        // extract rooms
        RasterCanvas cvs(cc_bb, 8192, 20);
        //connect_lines(cad.layers[CAD::WDAS_WALLS].lines, layers, &connected.lines);
        //connected.lines = cad.layers[CAD::WDAS_WALLS].lines;
        method2::connect_walls(cc_bb, cad.layers[CAD::WDAS_WALLS].lines, layers, cad.annotations[CAD::ANNO_HINT_CONNECT], &connected.lines);
        cvs.draw(connected, cv::Scalar(1));

        if (ilog.enabled()) {
            RasterCanvas vis111(cc_bb, 1024, 20, CV_8UC3);
            vis111.draw(connected, cv::Scalar(0, 255, 0));
            ilog.log(vis111.image, format("walls_%sc.png") % cc_id);
            vis111.draw(cad.layers[CAD::WDAS_WALLS], cv::Scalar(0, 0, 255));
            vis111.draw(cad.layers[CAD::WDAS_DOORS], cv::Scalar(255, 0, 0));
            vis111.draw(cad.layers[CAD::WDAS_SAFETY_DOORS], cv::Scalar(255, 0, 0));
            ilog.log(vis111.image, format("walls_%sd.png") % cc_id);
        }

        /*
        if (cad.annotations[CAD::ANNO_HINT_ROI].size()) {
            // apply annotation
            cv::Mat mask(cvs.image.rows, cvs.image.cols, CV_8U, cv::Scalar(0));
            for (cv::Rect roi: cad.annotations[CAD::ANNO_HINT_ROI]) {
                cvs.map(&roi);
                cv::rectangle(mask, roi, cv::Scalar(1), -1);
            }
            cvs.image &= mask;
        }
        */

        for (cv::Rect roi: cad.annotations[CAD::ANNO_HINT_CLEAR]) {
            cvs.map(&roi);
            cv::rectangle(cvs.image, roi, cv::Scalar(0), -1);
        }

        // 外边界填充2
        cv::floodFill(cvs.image, cv::Point(0,0), cv::Scalar(2));
        // cvs: 外侧2, 边界和物体1, 空的地方0
        {
            cv::Mat border;
            cv::threshold(cvs.image, border, 1, 1, cv::THRESH_BINARY_INV);
            // border: 外侧0, 其余1
            ilog.log(border * 255, format("border_%s.png") % cc_id);
            borders->clear();
            cv::findContours(border, *borders, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS);
            // findContours modifies border, so discard border here by scope
        }

        // cvs: 外侧1, 边界和物体1, 空的地方0
        cv::threshold(cvs.image, cvs.image, 1, 1, cv::THRESH_TRUNC);
        ilog.log(cvs.image * 255, format("trunc_%s.png") % cc_id);

        cv::Mat labels, stats, centroids;
        // holes: 外侧0, 边界和物体0, 空的地方1
        cv::Mat holes = cv::Scalar::all(1) - cvs.image;
        ilog.log(holes * 255, format{"cc_%d.png"} % cc_id);
        int n = cv::connectedComponentsWithStats(holes, labels, stats, centroids, 4, CV_32S);
        vector<Object> rooms;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(OPEN_KERNEL, OPEN_KERNEL));

        obstacles->clear();


        for (int i = 1; i < n; ++i) {
            std::pair<int,int> minmax = std::minmax(cvs.unscale(stats.at<int>(i, cv::CC_STAT_WIDTH)),
                                        cvs.unscale(stats.at<int>(i, cv::CC_STAT_HEIGHT)));
            if (minmax.first < ROOM_SIZE_MIN_THRESHOLD) continue;
            if (minmax.second < ROOM_SIZE_MAX_THRESHOLD) continue;
            // refine room contour

            cv::Mat room = (labels == i);   // 0 or 255
            CHECK(room.type() == CV_8U); 
            cv::threshold(room, room, 1, 1, cv::THRESH_BINARY); // -> 0/1

            {
                bool drop = false;
                for (cv::Rect roi: cad.annotations[CAD::ANNO_HINT_DROP_ROOM]) {
                    roi &= cc_bb;
                    if (roi.width <= 0) continue;
                    if (roi.height <= 0) continue;
                    cvs.map(&roi);
                    cv::Mat sub(room, roi);
                    double min, max;
                    cv::minMaxLoc(sub, &min, &max);
                    if (max > 0) {
                        drop = true;
                        break;
                    }
                }
                // 放弃仓间
                if (drop) continue;
            }


            
            // cv::drawContours(room, contours, i, cv::Scalar(1), -1);
            cv::morphologyEx(room, room, cv::MORPH_CLOSE, kernel);
            // CLOSE可能会close出窟窿来，需要填上
            // room: 房间内可用的地方是1, 其余地方是0
            cv::Mat closed_room = room.clone();
            cv::floodFill(closed_room, cv::Point(0,0), cv::Scalar(2));
            // closed_room: 外侧是2, 可用的地方是1, 内部的洞是0
            cv::threshold(closed_room, closed_room, 1, 1, cv::THRESH_BINARY_INV);
            // closed_room: 外侧是0, 内侧是1
            regularize_room(closed_room);
            Contours contours1;
            cv::findContours(closed_room.clone(), contours1, CV_RETR_LIST, CV_CHAIN_APPROX_TC89_KCOS);
            CHECK(contours1.size() == 1);


            rooms.emplace_back();
            rooms.back().contour.swap(contours1[0]);
            // 找出来内部的窟窿, 作为obstacle
            closed_room -= room;
            // closed_room: 内侧不可用的地方是1, 其余是0 
            cv::floodFill(closed_room, cv::Point(0,0), cv::Scalar(2));
            cv::threshold(closed_room, closed_room, 1, 1, cv::THRESH_BINARY_INV);
            // closed_room: 内侧不可用的地方是1, 其余是0 
            cv::findContours(closed_room, contours1, CV_RETR_LIST, CV_CHAIN_APPROX_TC89_KCOS);
            for (auto &c: contours1) {
                obstacles->emplace_back();
                obstacles->back().contour.swap(c);
            }
        }

        // unmap to world coordinates
        for (auto &room: rooms) {
            for (auto &p: room.contour) {
                cvs.unmap(&p);
            }
            room.bbox = bound(room.contour);
        }
        for (auto &obs: *obstacles) {
            for (auto &p: obs.contour) {
                cvs.unmap(&p);
            }
            obs.bbox = bound(obs.contour);
        }
        for (auto &c: *borders) {
            for (auto &p: c) {
                cvs.unmap(&p);
            }
        }
        results->swap(rooms);
    }

    bool box_contains_line (cv::Rect const &rect, Line const &l) {
        // TODO
        if (rect.contains(l[0])) return true;
        if (rect.contains(l[1])) return true;
        return false;
        //The Liang-Barsky algorithm is a cheap way to find the intersection points between a line segment and an axis-aligned rectangle. 
        // https://gist.github.com/ChickenProp/3194723
#if 0
        int dx = l[1].x - l[0].x;
        int dy = l[1].y - l[0].y;

        int p[] = {-dx, dx, -dy, dy};
        int q[] = {l[0].x - rect.x,                 // > 0:  l[0]是否在rect左侧
                   rect.x + rect.width - l[0].x,    // 
                   l[0].y - rect.y,
                   rect.y + rect.height - l[0].y
                    };
        var u1 = Math.NEGATIVE_INFINITY;
        var u2 = Math.POSITIVE_INFINITY;

        for (int i = 0; i < 4; ++i) {
            if (p[i] == 0) {
                if (q[i] < 0) return false;
            }
            else {
                var t = q[i] / p[i];
                if (p[i] < 0 && u1 < t)
                        u1 = t;
                else if (p[i] > 0 && u2 > t)
                        u2 = t;
            }
        }

        if (u1 > u2 || u1 > 1 || u1 < 0) return false;

        return true;
#endif
    }

    void filter_objects_by_contour(Contour const &contour,
                                   ObjectLayers const &from,
                                   ObjectLayers *to) {
        for (int i = 0; i < from.size(); ++i) {
            //to->at(i).clear();
            for (auto const &obj: from[i]) {
                if (test_intersect(contour, obj, OBJECT_ROOM_TEST_RELAX)) {
                    to->at(i).push_back(obj);
                }
            }
        }
    }

    void filter_objects_by_bbox (cv::Rect const &bbox,
                                   ObjectLayers const &from,
                                   ObjectLayers *to) {
        for (int i = 0; i < from.size(); ++i) {
            //to->at(i).clear();
            for (auto const &obj: from[i]) {
                cv::Rect inter = bbox & obj.bbox;
                if (inter.width > 0 && inter.height > 0) {
                    to->at(i).push_back(obj);
                }
            }
        }
    }

    bool align_object_to_room (Contour const &contour, Object const &obj, Object *tmp) {
        vector<method2::QLines> qlines;
        method2::add_qlines(&qlines, obj.bbox, true, ALIGN_TO_ROOM_RELAX);
        vector<cv::Point> pts;
        for (int i = 0; i < contour.size(); ++i) {
            auto const &p1 = contour[i];
            auto const &p2 = contour[(i+1) % contour.size()];
            if (p1 == p2) continue;
            XLine l(p1, p2);
            for (auto const &q: qlines) {
                for (auto const &l2: q) {
                    if (parallel(l, l2)) continue;
                    cv::Point x = method2::intersect(l, l2);
                    if (method2::try_extend(l2, x, true, 0, 0).hit && method2::try_extend(l, x, true, 0, 0).hit) {
                        pts.push_back(x);
                    }
                }
            }
        }
        if (pts.size() < 2) return false;
        tmp->bbox = bound(pts);
        box2contour(tmp->bbox, &tmp->contour);
        return true;
    }

    // TODO: speedup
    void align_objects_to_room (Contour const &contour,
                                   ObjectLayers const &from,
                                   ObjectLayers *to) {
        for (int i = 0; i < from.size(); ++i) {
            //to->at(i).clear();
            for (auto const &obj: from[i]) {
                Object tmp;
                if (align_object_to_room(contour, obj, &tmp)) {
                    to->at(i).push_back(tmp);
                }
            }
        }
    }

    map<string, int> const WDAX_LOOKUP{
        {"WDA_WALLS", CAD::WDAS_WALLS},
        {"WDA_PILLARS", CAD::WDAS_PILLARS},
        {"WDA_DOORS", CAD::WDAS_DOORS},
        {"WDA_OBSTACLE", CAD::WDAS_OBSTACLE},
        //{"WDA_EXITS", CAD::WDAX_EXITS},   // TODO
        {"WDAS_WALLS", CAD::WDAS_WALLS},
        {"WDAS_PILLARS", CAD::WDAS_PILLARS},
        {"WDAS_DOORS", CAD::WDAS_DOORS},
        {"WDAS_SAFETY_DOORS", CAD::WDAS_SAFETY_DOORS},
        {"WDAS_FIRE_HYDRANT", CAD::WDAS_FIRE_HYDRANT},
        {"WDAS_OBSTACL", CAD::WDAS_OBSTACLE},
        {"WDAS_OBSTACLE", CAD::WDAS_OBSTACLE},
        {"WDAS_GUARD", CAD::WDAS_GUARD},
        {"WDAF_PASSAGE", CAD::WDAF_PASSAGE},
        {"WDAF_DOCK", CAD::WDAF_DOCK},
        {"WDAF_DOCK_IN", CAD::WDAF_DOCK_IN},
        {"WDAF_DOCK_OUT", CAD::WDAF_DOCK_OUT},
        {"WDAF_MINIROOM", CAD::WDAF_MINIROOM}, 
        {"WDAS_MINIROOM", CAD::WDAF_MINIROOM},       // 容错
        {"WDAX_FORKLIFT", CAD::WDAX_FORKLIFT}, 
    };

    map<string, int> const ANNO_LOOKUP{
        {"HINT_ROI", CAD::ANNO_HINT_ROI},
        {"HINT_CLEAR", CAD::ANNO_HINT_CLEAR},
        {"HINT_DOCK", CAD::ANNO_HINT_DOCK},
        {"HINT_DOCK_IN", CAD::ANNO_HINT_DOCK_IN},
        {"HINT_DOCK_OUT", CAD::ANNO_HINT_DOCK_OUT},
        {"HINT_CONNECT", CAD::ANNO_HINT_CONNECT},
        {"HINT_DROP_ROOM", CAD::ANNO_HINT_DROP_ROOM},
        {"GUARD_OBSTACLE", CAD::ANNO_GUARD_OBSTACLE},
        {"GUARD_PASSAGE", CAD::ANNO_GUARD_PASSAGE},
        {"GUARD_MINIROOM", CAD::ANNO_GUARD_MINIROOM},
        {"GUARD_FORKLIFT", CAD::ANNO_GUARD_FORKLIFT}
    };

    vector<int> const DOOR_LIKE_LAYERS{CAD::WDAS_DOORS,
                                       CAD::WDAX_EXITS,
                                       CAD::WDAX_DOCK,
                                       CAD::WDAX_DOCK_IN,
                                       CAD::WDAX_DOCK_OUT,
                                       CAD::WDAS_SAFETY_DOORS};


    map<int, int> const MAP_WDAX_FIXTURE_CODE{
        {CAD::WDAS_WALLS, Fixture::WALL},
        {CAD::WDAS_PILLARS, Fixture::COLUMN},
        {CAD::WDAS_DOORS, Fixture::DOOR},
        {CAD::WDAX_EXITS, Fixture::EXIT},
        {CAD::WDAX_DOCK, Fixture::DOCK},
        {CAD::WDAX_DOCK_IN, Fixture::DOCK_IN},
        {CAD::WDAX_DOCK_OUT, Fixture::DOCK_OUT},
        {CAD::WDAS_SAFETY_DOORS, Fixture::SAFETY_DOOR},
        {CAD::WDAS_FIRE_HYDRANT, Fixture::FIRE_HYDRANT},
        {CAD::WDAS_OBSTACLE, Fixture::OBSTACLE},
        {CAD::WDAS_GUARD, Fixture::GUARD},
        {CAD::WDAF_PASSAGE, Fixture::GUARD_PASSAGE},
        {CAD::WDAF_MINIROOM, Fixture::GUARD},
        {CAD::WDAX_FORKLIFT, Fixture::FORKLIFT},
        {CAD::WDAX_GUARD_2M, Fixture::GUARD_2M},
        {CAD::WDAX_ACC_GUARD, Fixture::ACC_GUARD},
    };

    py::list create_py_contour (Contour const &c) {
        py::list l;
        for (auto const &p: c) {
            py::list pp;
            pp.append(p.x);
            pp.append(p.y);
            l.append(pp);
        }
        return l;
    }

    py::dict create_py_room (int cc, bool miniroom, ObjectLayers const &layers) {
        py::list fixtures;
        array<int, CAD::NUM_LAYERS> indices;
        for (int i = 0; i < indices.size(); ++i) indices[i] = i;
        // 把guard换到前面以免显示的时候覆盖住别的东西
        std::swap(indices[CAD::WDAS_PILLARS], indices[CAD::WDAS_GUARD]);
        for (int i: indices) {
            auto it = MAP_WDAX_FIXTURE_CODE.find(i);
            if (it == MAP_WDAX_FIXTURE_CODE.end()) continue;
            if (layers[i].empty()) continue;
            int code = it->second;
            //LOG(INFO) << "OBJ " << code << ": " << layers[i].size();
            for (auto const &c: layers[i]) {
                py::dict fixture;
                fixture["polygon"] = create_py_contour(c.contour);
                fixture["type"] = code;
                fixture["height"] = -1;
                fixture["effective"] = true;
                if (code == Fixture::WALL) {
                    fixture["miniroom"] = miniroom;
                }
                fixtures.append(fixture);
            }
        }
        py::dict room;
        room["cc"] = cc;
        room["fixtures"] = fixtures;
        return room;
    }

    void merge_boxes_to_objects (vector<cv::Rect> const &boxes, Objects *objs) {
        for (auto const &box: boxes) {
            Object obj;
            obj.bbox = box;
            box2contour(box, &obj.contour);
            objs->push_back(obj);
        }
    }

    void cleanup_object_layers (ObjectLayers &obj_layers, Contours const &borders) {
        obj_layers[CAD::DEFAULT].clear();
        Objects doors;
        doors.swap(obj_layers[CAD::WDAS_DOORS]);

        vector<pair<Objects const*, Objects*>> dock_tests{
            {&obj_layers[CAD::WDAF_DOCK], &obj_layers[CAD::WDAX_DOCK]},
            {&obj_layers[CAD::WDAF_DOCK_IN], &obj_layers[CAD::WDAX_DOCK_IN]},
            {&obj_layers[CAD::WDAF_DOCK_OUT], &obj_layers[CAD::WDAX_DOCK_OUT]}};

        // 把DOORS分类成DOORS, EXITS, DOCK{_IN/OUT}
        for (auto &door: doors) {
            int l = std::max(door.bbox.width, door.bbox.height);
            if (l < DOOR_EXIT_THRESHOLD) {
                obj_layers[CAD::WDAX_EXITS].push_back(std::move(door));
                continue;
            }
            // 测试各种dock, dock_in, dock_out
            bool used = false;
            // TODO: 如果WDAF_DOCK*有重叠，可能会导致一个dock被分入多个目标图层
            for (auto p: dock_tests) {
                Objects const *tests = p.first;
                Objects *targets = p.second;
                for (Object const &bb: *tests) {
                    if (test_intersect(bb.contour, door, 0)) {
                        targets->push_back(door);
                        used = true;
                        break;
                    }
                }
            }
            if (!used) {
                // 测试是否是dock
                Contour pts;
                {
                    int const relax = OBJECT_ROOM_TEST_RELAX;
                    cv::Rect bb = door.bbox;
                    bb.x -= relax;
                    bb.y -= relax;
                    bb.width += 2 * relax;
                    bb.height += 2 * relax;
                    box2contour(bb, &pts);
                }
                int all_in = true; // 所有的点全在墙内
                for (auto const &pt: pts) {
                    int ii = 0;
                    while (ii < borders.size()) {
                        auto const &c = borders[ii];
                        if (cv::pointPolygonTest(c, pt, true) > 0) break;
                        if (cv::pointPolygonTest(c, pt, true) > 0) break;
                        if (cv::pointPolygonTest(c, pt, true) > 0) break;
                        if (cv::pointPolygonTest(c, pt, true) > 0) break;
                        ++ii;
                    }
                    if (ii >= borders.size()) { // 该点不在任何墙内
                        all_in = false;
                        break;
                    }
                }
                if (!all_in) { // 有点在墙外就是dock
                    used = true;
                    obj_layers[CAD::WDAX_DOCK].push_back(std::move(door));
                }
            }
            if (!used) {
                obj_layers[CAD::WDAS_DOORS].push_back(std::move(door));
            }
        }

        // 处理消防栓
        // http://gitlab.shucheng-ai.com:38080/wda/wda-doc/issues/34
        for (auto const &fire: obj_layers[CAD::WDAS_FIRE_HYDRANT]) {
            // 找最近的柱子以及距离
            cv::Point ct = fire.center();
            double min_dist = 0;
            Object const *nearest = nullptr;
            for (auto const &pillar: obj_layers[CAD::WDAS_PILLARS]) {
                double dist = cv::norm(ct - pillar.center());
                if (nearest == nullptr || dist < min_dist) {
                    min_dist = dist;
                    nearest = &pillar;
                }
            }
            if (nearest) { // 确定是否足够近
                cv::Rect inter = fire.bbox & relax_box(nearest->bbox, FIRE_HYDRANT_PILLAR_GAP);
                // 太远 废掉
                if (inter.width <= 0) nearest = nullptr;
            }
            if (nearest) {
                cv::Point dir = ct - nearest->center();
                int dw = (fire.bbox.width + nearest->bbox.width)/2;
                int dh = (fire.bbox.height + nearest->bbox.height)/2;
                // 测试在左右还是在上下
                Object guard;
                if (std::abs(std::abs(dir.x) - dw) < std::abs(std::abs(dir.y) - dh)) {
                    if (dir.x >= 0) {
                        // fire on right of pillar
                        guard.bbox = cv::Rect(ct.x, ct.y - FIRE_ACC_GUARD_SIZE/2,
                                         FIRE_ACC_GUARD_SIZE, FIRE_ACC_GUARD_SIZE);
                    }
                    else {
                        // fire on left of pillar
                        guard.bbox = cv::Rect(ct.x+fire.bbox.width-FIRE_ACC_GUARD_SIZE,
                                         ct.y - FIRE_ACC_GUARD_SIZE/2,
                                         FIRE_ACC_GUARD_SIZE, FIRE_ACC_GUARD_SIZE);
                    }

                }
                else {
                    if (dir.y > 0) {
                        // fire on top of pillar
                        guard.bbox = cv::Rect(ct.x-FIRE_ACC_GUARD_SIZE/2, ct.y,
                                         FIRE_ACC_GUARD_SIZE, FIRE_ACC_GUARD_SIZE);
                    }
                    else {
                        // fire on bottom of pillar
                        guard.bbox = cv::Rect(ct.x-FIRE_ACC_GUARD_SIZE/2,
                                         ct.y+fire.bbox.height-FIRE_ACC_GUARD_SIZE,
                                         FIRE_ACC_GUARD_SIZE, FIRE_ACC_GUARD_SIZE);
                    }
                }
                box2contour(guard.bbox, &guard.contour);
                obj_layers[CAD::WDAX_ACC_GUARD].push_back(guard);
            }
            else {  // 假设在墙上
                Object guard;
                guard.bbox = relax_box(ct, FIRE_GUARD_2M_SIZE);
                box2contour(guard.bbox, &guard.contour);
                obj_layers[CAD::WDAX_GUARD_2M].push_back(guard);
            }
        }

        // miniroom全都变成GUARD
        for (auto &miniroom: obj_layers[CAD::WDAF_DOCK]) {
            obj_layers[CAD::WDAS_OBSTACLE].push_back(miniroom);
        }

        obj_layers[CAD::WDAF_DOCK].clear();
        obj_layers[CAD::WDAF_DOCK].clear();
        obj_layers[CAD::WDAF_DOCK_IN].clear();
        obj_layers[CAD::WDAF_DOCK_OUT].clear();
    }

    void add_guards_one_segment (cv::Point p1, cv::Point p2, int dist, Objects *guards) {
        cv::Point d = p2 - p1;
        cv::Point2f d1(-d.y, d.x);    // 90 degree counter
        {
            float l = cv::norm(d1);
            if (l == 0) return;
            d1 *= dist / l;
        }
        cv::Point2f s1 = p1;
        cv::Point2f s2 = p2;
        cv::Point2f s3 = s2 + d1;
        cv::Point2f s4 = s1 + d1;
        guards->emplace_back();
        Object &obj = guards->back();
        obj.contour.emplace_back(s1);
        obj.contour.emplace_back(s2);
        obj.contour.emplace_back(s3);
        obj.contour.emplace_back(s4);
        obj.bbox = bound(obj.contour);
    }

    void add_wall_guards (Contour const &contour, int dist, Objects *guards) {
        // wall contour is counter-clockwise
        for (int i = 0; i < contour.size(); ++i) {
            cv::Point const &p1 = contour[i];
            cv::Point const &p2 = contour[(i+1) % contour.size()];
            add_guards_one_segment(p1, p2, dist, guards);
        }
    }

    struct RoomInfo {
        int cc = -1;
        Object walls;
        bool miniroom = false;
        cv::Mat thumbnail;
    };

    class CadProcessor {
        CAD cad;
        int current_layer;
    public:
        CadProcessor (): current_layer(CAD::DEFAULT) {
        }

        // select layer to inject
        bool select (string const &layer) {
            auto it = WDAX_LOOKUP.find(layer);
            if (it == WDAX_LOOKUP.end()) return false;
            current_layer = it->second;
            CHECK(current_layer >= 0);
            CHECK(current_layer < cad.layers.size());
            return true;
        }

        void add (float x1f, float y1f, float x2f, float y2f) {
            int x1 = int(round(x1f));
            int x2 = int(round(x2f));
            int y1 = int(round(y1f));
            int y2 = int(round(y2f));
            cad.layers[current_layer].lines.emplace_back(Line{cv::Point(x1, y1), cv::Point(x2, y2)});
        }

        void annotate (string const &annotation, int x1, int y1, int x2, int y2) {
            auto it = ANNO_LOOKUP.find(annotation);
            if (it == ANNO_LOOKUP.end()) {
                LOG(INFO) << "ANNOTATION NOT RECOGNIZED: " << annotation;
                return;
            }
            cad.annotations[it->second].push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
        }

        py::list extract (string const &root) {

            // filter with ROI
            auto const &rois = cad.annotations[CAD::ANNO_HINT_ROI];
            if (rois.size()) {
                for (auto &layer: cad.layers) {
                    vector<Line> lines;
                    for (auto const &l: layer.lines) {
                        for (auto const &roi: rois) {
                            // 如果相交则保留
                            if (box_contains_line(roi, l)) {
                                lines.push_back(l);
                                break;
                            }
                        }
                    }
                    layer.lines.swap(lines);
                }
            }

            vector<cv::Rect> ccs;
            extract_cc(cad, &ccs, CC_RELAX);
            LOG(INFO) << "CCS: " << ccs.size();

            ObjectLayers all_objs;
            //ObjectLayers all_objs_test;

            for (int i = 0; i < CAD::NUM_LAYERS; ++i) {
                if (i == CAD::WDAS_WALLS) continue;
                Contours contours;
                /*
                */
                vector<cv::Rect> bbs;
                extract_cc(cad, &bbs, OBJECT_RELAX, i);

                Objects &objs = all_objs[i];
                //Contours &objs_test = all_objs_test[i];
                for (auto &bb: bbs) { // boxes to contours
                    objs.emplace_back();
                    objs.back().bbox = bb;
                    box2contour(bb, &objs.back().contour);
                    /*
                    bb.x -= OBJECT_ROOM_TEST_RELAX;
                    bb.y -= OBJECT_ROOM_TEST_RELAX;
                    bb.width += 2 * OBJECT_ROOM_TEST_RELAX;
                    bb.height += 2 * OBJECT_ROOM_TEST_RELAX;
                    objs_test.emplace_back();
                    box2contour(bb, &objs_test.back());
                    */
                }
            }
            // move annotations to objects
            merge_boxes_to_objects(cad.annotations[CAD::ANNO_HINT_DOCK], &all_objs[CAD::WDAF_DOCK]);
            merge_boxes_to_objects(cad.annotations[CAD::ANNO_HINT_DOCK_IN], &all_objs[CAD::WDAF_DOCK_IN]);
            merge_boxes_to_objects(cad.annotations[CAD::ANNO_HINT_DOCK_OUT], &all_objs[CAD::WDAF_DOCK_OUT]);
            merge_boxes_to_objects(cad.annotations[CAD::ANNO_GUARD_OBSTACLE], &all_objs[CAD::WDAS_GUARD]);
            merge_boxes_to_objects(cad.annotations[CAD::ANNO_GUARD_PASSAGE], &all_objs[CAD::WDAF_PASSAGE]);
            merge_boxes_to_objects(cad.annotations[CAD::ANNO_GUARD_MINIROOM], &all_objs[CAD::WDAF_MINIROOM]);
            merge_boxes_to_objects(cad.annotations[CAD::ANNO_GUARD_FORKLIFT], &all_objs[CAD::WDAX_FORKLIFT]);

            // 被WDAF_PASSAGE覆盖的pillars都需要干掉
            {
                Objects pillars;
                pillars.swap(all_objs[CAD::WDAS_PILLARS]);
                auto const &passages = all_objs[CAD::ANNO_GUARD_PASSAGE];
                for (auto &pillar: pillars) {
                // TODO: 进行更完整的测试和切分
                // 如果有一半在通道里，应该把不在的部分切出来。
                    bool hit = false;
                    for (auto const &pt: pillar.contour) {
                        for (auto const &passage: passages) {
                            if (passage.bbox.contains(pt)) {
                                hit = true;
                                goto found_hit;
                            }
                        }
                    }
found_hit:          if (!hit) {
                        all_objs[CAD::WDAS_PILLARS].push_back(pillar);
                    }
                }
                
            }



            vector<RoomInfo> rooms;
            Contours borders;   // 所有的仓间连续部分的外围多边形
                                    // 用于判断仓间门 vs dock
            int cc_id = 0;
            py::list pyccs;

            for (auto const &cc: ccs) {
                std::pair<int,int> minmax = std::minmax(cc.width, cc.height);
                if (minmax.first < ROOM_SIZE_MIN_THRESHOLD) continue;
                if (minmax.second < ROOM_SIZE_MAX_THRESHOLD) continue;


                {
                    RasterCanvas vis(cc, 2048, 20, CV_8U);
                    vis.draw(cad, cv::Scalar(255));
                    ilog.log(vis.image, format("ccv_%d.png") % cc_id);
                }

                Objects walls;
                Objects internal_holes;
                Contours bds;
                extract_rooms(cad, all_objs, cc, cc_id, &walls, &bds, &internal_holes);

                LOG(INFO) << "GOT OBSTACLES " << internal_holes.size();
                all_objs[CAD::WDAS_OBSTACLE].insert(all_objs[CAD::WDAS_OBSTACLE].end(),
                                internal_holes.begin(), internal_holes.end());

                if (walls.empty()) continue;

                // bds合并入borders
                for (auto &c: bds) {
                    borders.emplace_back();
                    borders.back().swap(c);
                }

                RasterCanvas thumbnail(cc, 256, 10, CV_8UC3);
                thumbnail.draw(cad, cv::Scalar(255, 255, 255));

                vector<int> order;
                sort_rooms(walls, &order);

                // 把minirooms加入
                int normal_walls = walls.size();
                for (auto const &miniroom: all_objs[CAD::WDAF_MINIROOM]) {
                    for (int xxx = 0; xxx < normal_walls; ++xxx) {
                        cv::Rect sect = miniroom.bbox & walls[xxx].bbox;
                        if (sect.width > 0 && sect.height > 0) {
                            // 碰撞了，加入miniroom
                            Object intersect;
                            contour_intersect(miniroom, walls[xxx], &intersect);
                            if (intersect.contour.size()) {
                                order.push_back(walls.size());
                                walls.push_back(intersect);
                                break;
                            }
                        }
                    }
                }

                for (int room_index: order) {
                    rooms.emplace_back();
                    RoomInfo &room = rooms.back();
                    room.cc = cc_id;

                    room.walls = walls[room_index];
                    room.miniroom = room_index >= normal_walls;

                    room.thumbnail = thumbnail.image.clone();
                    thumbnail.draw(room.walls.bbox, cv::Scalar(0, 255, 0), 2);
                    cv::swap(thumbnail.image, room.thumbnail);
                }
                {
                    py::list l2;
                    l2.append(cc.x);
                    l2.append(cc.y);
                    l2.append(cc.x + cc.width);
                    l2.append(cc.y + cc.height);
                    pyccs.append(l2);
                }
                ++cc_id;
            }

            cleanup_object_layers(all_objs, borders); //, rooms);

            ObjectLayers all_door_like;
            for (int l: DOOR_LIKE_LAYERS) {
                all_door_like[l].swap(all_objs[l]);
            }

            py::list pyrooms;
            for (int room_id = 0; room_id < rooms.size(); ++room_id) {
                auto &room = rooms[room_id];
                if (root.size()) {
                    system((format{"mkdir -p \"%1%/%2%\""} % root % room_id).str().c_str());
                    cv::imwrite((format("%1%/%2%/thumbnail.png") % root % room_id).str(), room.thumbnail);
                }
                ObjectLayers layers;
                //filter_objects_by_contour(room.walls.contour, all_objs, &layers);
                filter_objects_by_bbox(room.walls.bbox, all_objs, &layers);
                if (room.miniroom) { 
                    // 如果仓间本身是miniroom，判断碰撞时会和作为障碍物的自己碰撞
                    // 所以如果仓间本身是miniroom, 则minirooms不再作为障碍物加入
                    layers[CAD::WDAF_MINIROOM].clear();
                }
                align_objects_to_room(room.walls.contour, all_door_like, &layers);
                int wall_guard_dist = room.miniroom ? 0 : WALL_GUARDS_DIST;
                add_wall_guards(room.walls.contour, wall_guard_dist, &layers[CAD::WDAS_GUARD]);
                layers[CAD::WDAS_WALLS].push_back(room.walls);
                pyrooms.append(create_py_room(room.cc, room.miniroom, layers));
            }
            return py::make_tuple(pyrooms, pyccs);
        }

        py::list components (int relax) {
            vector<cv::Rect> ccs;
            extract_cc(cad, &ccs, relax);
            py::list l1;
            for (auto const &b: ccs) {
                py::list l2;
                l2.append(b.x);
                l2.append(b.y);
                l2.append(b.x + b.width);
                l2.append(b.y + b.height);
                l1.append(l2);
            }
            return l1;
        }
    };
}

namespace py = pybind11;
using beaver::CadProcessor;

void exportCadTypes (py::module &module) {
    py::class_<CadProcessor>(module,"CadProcessor")
         .def(py::init<>())
         .def("select", &CadProcessor::select)
         .def("add", &CadProcessor::add)
         .def("annotate", &CadProcessor::annotate)
         .def("extract", &CadProcessor::extract)
         .def("components", &CadProcessor::components)
         ;
}

PYBIND11_MODULE(cad_core, m)
{
    m.doc() = "";
    beaver::exportGeoTypes(m);
    exportCadTypes(m);
}

