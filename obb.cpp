#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/slice.h>
#include <igl/slice_mask.h>
#include <igl/slice_into.h>
#include <igl/cross.h>
#include <igl/unique.h>
#include <igl/per_face_normals.h>
#include <igl/copyleft/cgal/convex_hull.h>
#include "transform.h"
#include "unique_rows.h"

using namespace std;

// Implement Optimal Bounding Box with igl and Eigen
// Usage: ./build/main RLEG

void oriented_bounds_2d(const Eigen::MatrixX2d &V, double &x_span, double &y_span, Eigen::Matrix3d &M, int debug_i = -1)
{
    using namespace Eigen;
    Eigen::MatrixX2d CvxV;
    Eigen::MatrixX2i CvxE;
    igl::copyleft::cgal::convex_hull(V, CvxV, CvxE);
    // cout << "convex hull: " << CvxV.rows() << "," << CvxV.cols() << "| " << CvxF.rows() << "," << CvxF.cols() << endl;
    // cout << "CvxV\n"
    //      << CvxV << endl;
    // cout << "CvxE\n"
    //      << CvxE << endl;
    Eigen::MatrixX2d edge_vectors(CvxE.rows(), 2), perp_vectors(CvxE.rows(), 2);
    for (int i = 0; i < CvxE.rows(); i++)
    {
        int start_v = CvxE(i, 0);
        int end_v = CvxE(i, 1);

        edge_vectors(i, 0) = CvxV(end_v, 0) - CvxV(start_v, 0);
        edge_vectors(i, 1) = CvxV(end_v, 1) - CvxV(start_v, 1);
        // cout << "edge" << i << ": " << CvxV(end_v) - CvxV(start_v) << "|" << edge_vectors(i) << endl;
    }
    // cout << "edge_vectors:\n"
    //      << edge_vectors << endl;
    edge_vectors.rowwise().normalize();
    // cout << "edge_vectors_norm:\n"
    //      << edge_vectors << endl;
    perp_vectors.col(0) = -edge_vectors.col(1);
    perp_vectors.col(1) = edge_vectors.col(0);

    // project every hull point on every edge vector
    Eigen::MatrixXd x = edge_vectors * CvxV.transpose();
    Eigen::MatrixXd y = perp_vectors * CvxV.transpose();
    Eigen::MatrixX4d bounds(x.rows(), 4);
    bounds.col(0) = x.rowwise().minCoeff();
    bounds.col(1) = y.rowwise().minCoeff();
    bounds.col(2) = x.rowwise().maxCoeff();
    bounds.col(3) = y.rowwise().maxCoeff();

    // # (2,) float of smallest rectangle size
    Eigen::VectorXd x_extent = bounds.col(2) - bounds.col(0);
    Eigen::VectorXd y_extent = bounds.col(3) - bounds.col(1);
    Eigen::VectorXd area = x_extent.array() * y_extent.array();
    int min_r, min_c;
    double min_area = area.minCoeff(&min_r, &min_c);

    // # find the (3,3) homogeneous transformation which moves the input
    // # points to have a bounding box centered at the origin
    // move the min_point to origin and then move center_point to origin
    Eigen::Matrix3d transform;
    double offset_x = -bounds(min_r, 0) - x_extent(min_r) * 0.5;
    double offset_y = -bounds(min_r, 1) - y_extent(min_r) * 0.5;
    double theta = atan2(edge_vectors(min_r, 1), edge_vectors(min_r, 0));
    planar_matrix(offset_x, offset_y, theta, transform);

    // # we would like to consistently return an OBB with
    // # the largest dimension along the X axis rather than
    // # the long axis being arbitrarily X or Y.
    if (x_extent(min_r) < y_extent(min_r))
    {
        Eigen::Matrix3d flip;
        planar_matrix(M_PI / 2, flip);

        M = flip * transform;
        x_span = y_extent(min_r);
        y_span = x_extent(min_r);

        // cout << "------------------------------------ largest dimension at y|"
        //      << " x " << x_extent(min_r) << ", y " << y_extent(min_r) << endl
        //      << " edge_vector: " << edge_vectors(min_r, 1) << ", " << edge_vectors(min_r, 0) << endl
        //      << " tan: " << atan2(edge_vectors(min_r, 1), edge_vectors(min_r, 0)) << endl
        //      << " theta " << theta * 180 / M_PI << endl
        //      << "transform: " << endl
        //      << transform << endl
        //      << "M: " << endl
        //      << M << endl
        //      << "----------------------------------- " << debug_i << endl;
    }
    else
    {
        M = transform;
        x_span = x_extent(min_r);
        y_span = y_extent(min_r);
        // cout << "------------------------------------ largest dimension at x|"
        //      << " x " << x_extent(min_r) << ", y " << y_extent(min_r) << endl
        //      << " edge_vector: " << edge_vectors(min_r, 1) << ", " << edge_vectors(min_r, 0) << endl
        //      << " tan: " << atan2(edge_vectors(min_r, 1), edge_vectors(min_r, 0)) << endl
        //      << " theta " << theta * 180 / M_PI << endl
        //      << "transform: " << endl
        //      << transform << endl
        //      << "M: " << endl
        //      << M << endl
        //      << "----------------------------------- " << debug_i << endl;
    }
}

int main(int argc, char **argv)
{
    using namespace Eigen;
    MatrixXf V, CvxV, CvxN, OrientedCvxN, spherical_coords;
    MatrixXi F, CvxF;
    Eigen::Matrix<float, Eigen::Dynamic, 4> V4d, CvxV4d;
    // igl::readOFF("./data/cube.off", V, F);
    std::string obj_name = std::string(argv[1]);
    igl::readOBJ("./data/parts/" + obj_name + ".obj", V, F);
    V4d.resize(V.rows(), 4);
    V4d.block(0, 0, V.rows(), 3) = V;
    V4d.col(3) = Eigen::VectorXd::Ones(V4d.rows());

    igl::copyleft::cgal::convex_hull(V, CvxV, CvxF);
    igl::per_face_normals(CvxV, CvxF, CvxN);
    CvxV4d.resize(CvxV.rows(), 4);
    CvxV4d.block(0, 0, CvxV.rows(), 3) = CvxV;
    CvxV4d.col(3) = Eigen::VectorXd::Ones(CvxV4d.rows());

    cout << "V, F=" << V.rows() << ", " << F.rows() << endl;
    cout << "CvxV, CvxF=" << CvxV.rows() << ", " << CvxF.rows() << "x" << CvxF.cols() << endl;

    vector_hemisphere(CvxN, OrientedCvxN);
    vector_to_spherical(OrientedCvxN, spherical_coords);
    // cout << OrientedN << endl;
    MatrixXi row_index;
    // cout << "sp coords:\n";
    // cout << spherical_coords << endl;
    unique_rows(spherical_coords, row_index);
    double min_volume = std::numeric_limits<double>::max();
    Eigen::Vector3f min_extents;
    Eigen::Matrix4f min_2D;
    Eigen::Matrix4f rotation_Z;
    for (int i = 0; i < row_index.rows(); i++)
    {
        int r_idx = row_index(i);
        double theta = spherical_coords(r_idx, 0);
        double phi = spherical_coords(r_idx, 1);
        // to_2D: matrices which will rotate each hull normal to [0,0,1]
        Eigen::Matrix4f S, to_2D;
        spherical_matrix(theta, phi, S);
        to_2D = S.inverse();

        Eigen::MatrixX3f projected = ((to_2D * CvxV4d.transpose()).transpose()).block(0, 0, CvxV4d.rows(), 3);
        double height = projected.col(2).maxCoeff() - projected.col(2).minCoeff();
        Eigen::Matrix3f rotation_2D;
        double box_x, box_y;
        oriented_bounds_2d(projected.block(0, 0, projected.rows(), 2), box_x, box_y, rotation_2D, i);
        double volume = box_x * box_y * height;
        if (volume < min_volume)
        {
            printf("%d volume=%.6f box_x:%.6f box_y:%.6f height:%.6f\n", i, volume, box_x, box_y, height);
            min_volume = volume;
            min_extents = {box_x, box_y, height};
            min_2D = to_2D;
            rotation_2D(0, 2) = 0.0;
            rotation_2D(1, 2) = 0.0;
            Eigen::Matrix4f rotation_3D;
            planar_matrix_to_3D(rotation_2D, rotation_3D);
            rotation_Z = rotation_3D;
            cout << "----------------------->>>>Rotation\n"
                 << rotation_Z << endl;
            cout << "----------------------->>>>min_2D\n"
                 << min_2D << endl;

            // # combine the 2D OBB transformation with the 2D projection transform
            Eigen::Matrix4f to_origin = rotation_Z * min_2D;

            // # transform points using our matrix to find the translation for the
            // # transform
            Eigen::MatrixX3f transformedCvxV = (to_origin * V4d.transpose()).transpose().block(0, 0, V4d.rows(), 3);
            // igl::writeOBJ("./result/cvx_proj_" + std::to_string(i) + "_" + obj_name + ".obj", projected, CvxF);
            // igl::writeOBJ("./result/cvx_rot_" + std::to_string(i) + "_" + obj_name + ".obj", transformedCvxV, CvxF);
        }
    }

    cout << "----------------------->>>>Final Rotation\n"
         << rotation_Z << endl;
    cout << "----------------------->>>>Final min_2D\n"
         << min_2D << endl;
    // # combine the 2D OBB transformation with the 2D projection transform
    Eigen::Matrix4f to_origin = rotation_Z * min_2D;

    // # transform points using our matrix to find the translation for the
    // # transform
    Eigen::MatrixX3f transformedCvxV = (to_origin * CvxV4d.transpose()).transpose().block(0, 0, CvxV4d.rows(), 3);
    auto box_extents = transformedCvxV.colwise().maxCoeff() - transformedCvxV.colwise().minCoeff();
    Eigen::Vector3f box_center = transformedCvxV.colwise().minCoeff() + 0.5 * box_extents;
    to_origin.block(0, 3, 3, 1) = -box_center;
    cout << "min volume: " << min_volume << endl;
    cout << "min_extents: " << min_extents << "\n box extents: " << box_extents << endl;

    // #return ordered 3D extents
    // x is the longest, then y, then z
    int min_order, max_order, middle_order;
    double min_ext = min_extents.minCoeff(&min_order);
    double max_ext = min_extents.maxCoeff(&max_order);
    for (int i = 0; i < 3; i++)
    {
        if (i != min_order && i != max_order)
        {
            middle_order = i;
        }
    }

    Eigen::Matrix4d flip = Eigen::Matrix4d::Zero();
    int order[3] = {max_order, middle_order, min_order};
    flip(0, order[0]) = 1.0;
    flip(1, order[1]) = 1.0;
    flip(2, order[2]) = 1.0;
    flip(3, 3) = 1.0;
    cout << "flip:\n"
         << flip << endl;
    to_origin = flip * to_origin;
    double obb_x = min_extents(max_order);
    double obb_y = min_extents(middle_order);
    double obb_z = min_extents(min_order);
    cout << "obb: " << obb_x << ", " << obb_y << ", " << obb_z << endl;
    cout << "to_origin:\n"
         << to_origin << endl;

    transformedCvxV = (to_origin * CvxV4d.transpose()).transpose().block(0, 0, CvxV4d.rows(), 3);
    igl::writeOBJ("./result/cvx_rot_" + obj_name + "_final.obj", transformedCvxV, CvxF);
    Eigen::MatrixX3d transformedV = (to_origin * V4d.transpose()).transpose().block(0, 0, V4d.rows(), 3);
    igl::writeOBJ("./result/rot_" + obj_name + "_final.obj", transformedV, F);
    return 0;
}

// int main()
// {
//     using namespace Eigen;
//     MatrixX2d V(4, 2);
//     V << -1, 1,
//         1, -1,
//         1, 1,
//         -1, 1;
//     Eigen::Matrix3d rot;
//     double x_span, y_span;
//     oriented_bounds_2d(V, x_span, y_span, rot);
//     cout << "x_span: " << x_span << " y_span: " << y_span << endl;
//     cout << "rot\n"
//          << rot << endl;
//     return 0;
// }