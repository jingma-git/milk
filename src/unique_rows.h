#pragma once
#include <eigen3/Eigen/Eigen>
#include <igl/unique_rows.h>

inline void float_to_int(const Eigen::MatrixXf &data, Eigen::MatrixXi &int_data, int digits = 1)
{
    int_data = (data * std::pow(10, digits)).array().round().cast<int>();
}

inline void hashable_rows(const Eigen::MatrixXf &data, Eigen::MatrixXi &row_index, int digits = 1)
{
    Eigen::MatrixXi int_data;
    float_to_int(data, int_data, digits);

    Eigen::MatrixXi unique_int_data, IC;
    igl::unique_rows(int_data, unique_int_data, row_index, IC);
}

inline void unique_rows(const Eigen::MatrixXf &data, Eigen::MatrixXi &row_index, int digits = 1)
{
    // Returns indices of unique rows. It will return the
    // first occurrence of a row that is duplicated:
    // [[1,2], [3,4], [1,2]] will return [0,1]
    // Parameters
    // ---------
    // data : (n, m) array
    //   Floating point data
    // digits : int or None
    //   How many digits to consider
    assert(data.rows() > 0);
    hashable_rows(data, row_index, digits);
}