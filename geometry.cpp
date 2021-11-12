#include <cmath>
#include <sstream>
#include "cad.h"

namespace beaver {

    template <int D>
    Point_<D> *Point_create (py::args args) {
        if (args.size() == 0) return new Point_<D>();
        CHECK(args.size() == D);
        Point_<D> *p = new Point_<D>();
        for (int i = 0; i < D; ++i) {
            p->at(i) = args[i].cast<int>();
        }
        return p;
    }

    template <int D>
    py::list Point_unpack (Point_<D> *p) {
        py::list v;
        for (int i = 0; i < D; ++i) {
            v.append(p->at(i));
        }
        return v;
    }

    template <int D, int O>
    int Point_getitem_static (Point_<D> *p) {
        CHECK(O < D);
        return p->at(O);
    }

    template <int D>
    int Point_getitem (array<int, D> const &array, int offset) {
        return array[offset];
    }

    template <int D>
    string Point_str (Point_<D> *p) {
        std::ostringstream ss;
        ss << "(";
        for (int i = 0; i < D; ++i) {
            if (i) ss << ",";
            ss << p->at(i);
        }
        ss << ")";
        return ss.str();
    }

    template <int D>
    Box_<D> *Box_create (py::args args) {
        Box_<D> *p = new Box_<D>();
        if (args.size() == 0) {
            ; // no initialization
        }
        else if (args.size() == 1) {
            // arg is a list
            py::list list = args[0].cast<py::list>();
            int o = 0;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < D; ++j) {
                    p->at(i)[j] = list[o++].cast<int>();
                }
            }
        }
        else if (args.size() == 2) {
            CHECK(0);
        }
        else if (args.size() == D * 2) {
            int o = 0;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < D; ++j) {
                    p->at(i)[j] = args[o++].cast<int>();
                }
            }
        }
        else CHECK(0) << "not supported";
        return p;
    }

    template <int D>
    Box_<D-1> *Box_drop (Box_<D> *box, int d) {
        return new Box_<D-1>(box->drop(d));
    }


    template <int D>
    py::object Box_getitem (Box_<D> *box, int a) {
        return py::cast(new Point_<D>(box->at(a)));
    }


    template <int D>
    string Box_str (Box_<D> *p) {
        std::ostringstream ss;
        ss << "(";
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < D; ++j) {
                if (i || j) ss << ",";
                ss << p->at(i)[j];
            }
        }
        ss << ")";
        return ss.str();
    }

    template <int D>
    void Box_expand (Box_<D> *box, py::object obj) {
        try {
            Point_<D> *p = obj.cast<Point_<D> *>();
            box->expand(*p);
        }
        catch (py::cast_error const &) {
            try {
                Box_<D> *p = obj.cast<Box_<D> *>();
                box->expand(*p);
            }
            catch (py::cast_error const &) {
                CHECK(0);
            }
        }
    }

    template <int D>
    bool Box_overlap (Box_<D> *box, Box_<D> *box2) {
        for (int i = 0; i < D; ++i) {
            int lb = std::max(box->at(0)[i], box2->at(0)[i]);
            int ub = std::min(box->at(1)[i], box2->at(1)[i]);
            if (lb > ub) return false;
        }
        return true;
    }

    template <int D>
    void Box_expandFloat (Box_<D> *box, py::tuple point) {
        CHECK(point.size() >= D);
        for (int i = 0; i < D; ++i) {
            double v = point[i].cast<double>();
            int lb = int(std::floor(v));
            int ub = int(std::ceil(v));
            if (lb < box->at(0)[i]) box->at(0)[i] = lb;
            if (ub > box->at(1)[i]) box->at(1)[i] = ub;
        }
    }

    template <int D>
    void exportPointType (py::module &module, char const *name) {
        py::class_<Point_<D>>(module, name)
             .def(py::init(&Point_create<D>))
             .def("unpack", &Point_unpack<D>)
             .def("x", &Point_getitem_static<D, 0>)
             .def("y", &Point_getitem_static<D, 1>)
             .def("z", &Point_getitem_static<D, 2>)
             .def("__getitem__", &Point_getitem<D>)
             .def("__str__", &Point_str<D>);
    }

    template <int D>
    void exportBoxType (py::module &module, char const *name) {
        auto &c = py::class_<Box_<D>>(module, name)
            .def(py::init(&Box_create<D>))
            .def("unpack", &Box_<D>::unpack)
            .def("overlap", &Box_overlap<D>)
            .def("expand", &Box_expand<D>)
            .def("expandFloat", &Box_expandFloat<D>)
            .def("__getitem__", &Box_getitem<D>)
            .def("__str__", &Box_str<D>);
        if (D == 3) {
            c.def("drop", &Box_drop<D>);
        }
    }

    void exportGeoTypes (py::module &module) {
        exportPointType<2>(module, "Point");
        exportPointType<3>(module, "Point3D");
        exportBoxType<2>(module, "Box");
        exportBoxType<3>(module, "Box3D");
    }

}
