#pragma once

#include <eigen3/Eigen/Eigen>
#include <array>
#include <numeric>

struct BBox
{
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float min_z = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::min();
    float max_y = std::numeric_limits<float>::min();
    float max_z = std::numeric_limits<float>::min();
};

inline std::ostream &operator<<(std::ostream &out, const BBox &bbox)
{
    out << bbox.min_x << ", " << bbox.min_y << ", " << bbox.min_z << "---" << bbox.max_x << ", " << bbox.max_y << ", " << bbox.max_z;
    return out;
}

class Triangle
{
public:
    Eigen::Vector4f v[3];
    Eigen::Vector3f color[3];
    Eigen::Vector2f tex_coords[3];
    Eigen::Vector3f normal[3];

    Triangle();

    BBox bbox() const;
    void setVertex(int ind, Eigen::Vector4f ver);
    void setNormal(int ind, Eigen::Vector3f n);
    void setColor(int ind, float r, float g, float b);
    void setTexCoord(int ind, float s, float t);
};