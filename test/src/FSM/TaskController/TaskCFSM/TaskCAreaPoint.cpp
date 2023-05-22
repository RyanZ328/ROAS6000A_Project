#include "TaskCAreaPoint.hpp"

double Abs(double x)
{
    if (x > 0.0)
    {
        return x;
    }
    return -x;
}

bool EdgeCheck(Point &pos, Point &Point1, Point &Point2)
{
    if (Abs(Point1.x - Point2.x) < 0.05)
    {
        if (Abs(pos.x - Point1.x) > edge)
        {
            return false;
        }

        if (((pos.y - Point1.y) > 0 && (pos.y - Point2.y) < 0) || ((pos.y - Point2.y) > 0 && (pos.y - Point1.y) < 0))
        {
            return true;
        }

        return false;
    }
    else
    {
        if (Abs(pos.y - Point1.y) > edge)
        {
            return false;
        }

        if (((pos.x - Point1.x) > 0 && (pos.x - Point2.x) < 0) || ((pos.x - Point2.x) > 0 && (pos.x - Point1.x) < 0))
        {
            return true;
        }

        return false;
    }
}

bool AreaCheck(Point &pos, Area &area)
{
    double x_list[4] = {area.Point1.x, area.Point2.x, area.Point3.x, area.Point4.x};
    double y_list[4] = {area.Point1.y, area.Point2.y, area.Point3.y, area.Point4.y};

    double x_min = x_list[0], x_max = x_list[0];
    double y_min = y_list[0], y_max = y_list[0];

    for (int i = 1; i < 4; i++)
    {
        if (x_min > x_list[i])
        {
            x_min = x_list[i];
        }

        if (y_min > y_list[i])
        {
            y_min = y_list[i];
        }

        if (x_max < x_list[i])
        {
            x_max = x_list[i];
        }

        if (y_max < y_list[i])
        {
            y_max = y_list[i];
        }
    }

    if (pos.x < x_max && pos.x > x_min && pos.y > y_max && pos.y < y_min)
    {
        return true;
    }
    return false;
}