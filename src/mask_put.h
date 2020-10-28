#pragma once
#include <eigen3/Eigen/Eigen>

template <typename T, typename M>
inline void mask_put(
    Eigen::DenseBase<T> &X,
    const Eigen::DenseBase<M> &mask,
    double val)
{
    assert((X.rows() == mask.rows() && X.cols() == mask.cols()) && "X and mask should have the same size");
    for (int i = 0; i < X.rows(); i++)
    {
        for (int j = 0; j < X.cols(); j++)
        {
            if (mask(i, j))
            {
                X(i, j) = val;
            }
        }
    }
}