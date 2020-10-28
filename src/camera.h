#pragma once

#include <eigen3/Eigen/Eigen>

class Camera
{
public:
    Eigen::Vector3f position;
    Eigen::Vector3f up;
    Eigen::Vector3f right;
};