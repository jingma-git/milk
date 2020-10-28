#include "primitives.h"

Triangle::Triangle()
{
    v[0] << 0, 0, 0, 1;
    v[1] << 0, 0, 0, 1;
    v[2] << 0, 0, 0, 1;

    color[1] << 0, 0, 0;
    color[2] << 0, 0, 0;
    color[3] << 0, 0, 0;

    tex_coords[0] << 0, 0;
    tex_coords[1] << 0, 0;
    tex_coords[2] << 0, 0;
}

BBox Triangle::bbox() const
{
    BBox box;
    for (int i = 0; i < 3; i++)
    {
        if (v[i].x() < box.min_x)
        {
            box.min_x = v[i].x();
        }
        if (v[i].y() < box.min_y)
        {
            box.min_y = v[i].y();
        }
        if (v[i].z() < box.min_z)
        {
            box.min_z = v[i].z();
        }

        if (v[i].x() > box.max_x)
        {
            box.max_x = v[i].x();
        }
        if (v[i].y() > box.max_y)
        {
            box.max_y = v[i].y();
        }
        if (v[i].z() > box.max_z)
        {
            box.max_z = v[i].z();
        }
    }
    return box;
}

void Triangle::setVertex(int ind, Eigen::Vector4f ver)
{
    v[ind] = ver;
}

void Triangle::setNormal(int ind, Eigen::Vector3f n)
{
    normal[ind] = n;
}

void Triangle::setColor(int ind, float r, float g, float b)
{
    color[ind] = Eigen::Vector3f(r / 255., g / 255., b / 255.);
}

void Triangle::setTexCoord(int ind, float s, float t)
{
    tex_coords[ind] = Eigen::Vector2f(s, t);
}
