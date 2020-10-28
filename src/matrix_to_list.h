#pragma once
#include <eigen3/Eigen/Eigen>
#include <vector>

inline void matrix_to_list(const Eigen::MatrixX3f &M, std::vector<Eigen::Vector3f> &list)
{
    using namespace Eigen;
    int n = M.rows();
    int m = M.cols();
    list.resize(n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            list[i](j) = M(i, j);
        }
    }
}

inline void matrix_to_list(const Eigen::MatrixX3i &M, std::vector<Eigen::Vector3i> &list)
{
    using namespace Eigen;
    int n = M.rows();
    int m = M.cols();
    list.resize(n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            list[i](j) = M(i, j);
        }
    }
}