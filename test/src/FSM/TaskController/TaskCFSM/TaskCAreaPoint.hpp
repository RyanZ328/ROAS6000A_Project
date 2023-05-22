#ifndef TASK_C_AREA_POINT_HPP
#define TASK_C_AREA_POINT_HPP

const static double edge = 0.1;

struct Point
{
    Point():x(0.0),y(0.0){}
    Point(const Point& _Point):x(_Point.x),y(_Point.y){}
    Point(double _x,double _y):x(_x),y(_y){}

    double x;
    double y;
};

struct Area
{
    Area(){}
    Area(Point _Point1,Point _Point2,Point _Point3,Point _Point4):
        Point1(_Point1),Point2(_Point2),Point3(_Point3),Point4(_Point4){}
    Area(Point& _Point1,Point& _Point2,Point& _Point3,Point& _Point4):
        Point1(_Point1),Point2(_Point2),Point3(_Point3),Point4(_Point4){}

    Point Point1;
    Point Point2;
    Point Point3;
    Point Point4;
};

const static double AreaPoint1X = 0.64;
const static double AreaPoint1Y = 3.01;

const static double AreaPoint2X = 2.655;
const static double AreaPoint2Y = 3.01;

const static double AreaPoint3X = -3.274;
const static double AreaPoint3Y = 6.4626;

const static double AreaPoint4X = -4.7545;
const static double AreaPoint4Y = 6.4626;

const static double AreaPoint5X = -5.05;
const static double AreaPoint5Y = 11.44;

const static double AreaPoint6X = -5.05;
const static double AreaPoint6Y = 13.13;

static Point Point1(AreaPoint1X,AreaPoint1Y);
static Point Point2(AreaPoint2X,AreaPoint2Y);
static Point Point3(AreaPoint3X,AreaPoint3Y);
static Point Point4(AreaPoint4X,AreaPoint4Y);
static Point Point5(AreaPoint5X,AreaPoint5Y);
static Point Point6(AreaPoint6X,AreaPoint6Y);

static Area area_a(Point(0,0),Point(0,0),Point(0,0),Point(0,0));
static Area area_b(Point(0,0),Point(0,0),Point(0,0),Point(0,0));
static Area area_c(Point(0,0),Point(0,0),Point(0,0),Point(0,0));
static Area area_d(Point(0,0),Point(0,0),Point(0,0),Point(0,0));

bool EdgeCheck(Point& pos, Point& Point1, Point& Point2);
bool AreaCheck(Point& pos, Area& area);

#endif
