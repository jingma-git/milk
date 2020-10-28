#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/slice.h>
#include <igl/slice_mask.h>
#include <igl/slice_into.h>
#include <igl/cross.h>
#include <igl/per_face_normals.h>
#include <igl/copyleft/cgal/convex_hull.h>
#include <igl/matrix_to_list.h>
#include <opencv2/opencv.hpp>
#include "unique_rows.h"
#include "transform.h"
using namespace std;

// Implement Optimal Bounding Box with igl and Eigen
// How many ToyParts's real_obb lies in the view giving the maximum projection mask area?
// What's the difference between max_area_view_obb, min_volume_view_obb and real_obb?
// What if trying to rotate under max_area_view to find OBB?

std::string view_names[6] = {"front", "back", "left", "right", "top", "bottom"};
enum VIEW
{
    FRONT = 0,
    BACK,
    LEFT,
    RIGHT,
    TOP,
    BOTTOM,
    NUMS
};

void oriented_bounds_2d(const Eigen::MatrixX2f &V, float &x_span, float &y_span, Eigen::Matrix3f &M, int debug_i = -1)
{
    using namespace Eigen;
    Eigen::MatrixX2f CvxV;
    Eigen::MatrixX2i CvxE;
    igl::copyleft::cgal::convex_hull(V, CvxV, CvxE);
    // cout << "convex hull: " << CvxV.rows() << "," << CvxV.cols() << "| " << CvxF.rows() << "," << CvxF.cols() << endl;
    // cout << "CvxV\n"
    //      << CvxV << endl;
    // cout << "CvxE\n"
    //      << CvxE << endl;
    Eigen::MatrixX2f edge_vectors(CvxE.rows(), 2), perp_vectors(CvxE.rows(), 2);
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
    Eigen::MatrixXf x = edge_vectors * CvxV.transpose();
    Eigen::MatrixXf y = perp_vectors * CvxV.transpose();
    Eigen::MatrixX4f bounds(x.rows(), 4);
    bounds.col(0) = x.rowwise().minCoeff();
    bounds.col(1) = y.rowwise().minCoeff();
    bounds.col(2) = x.rowwise().maxCoeff();
    bounds.col(3) = y.rowwise().maxCoeff();

    // # (2,) float of smallest rectangle size
    Eigen::VectorXf x_extent = bounds.col(2) - bounds.col(0);
    Eigen::VectorXf y_extent = bounds.col(3) - bounds.col(1);
    Eigen::VectorXf area = x_extent.array() * y_extent.array();
    int min_r, min_c;
    float min_area = area.minCoeff(&min_r, &min_c);

    // # find the (3,3) homogeneous transformation which moves the input
    // # points to have a bounding box centered at the origin
    // move the min_point to origin and then move center_point to origin
    Eigen::Matrix3f transform;
    float offset_x = -bounds(min_r, 0) - x_extent(min_r) * 0.5;
    float offset_y = -bounds(min_r, 1) - y_extent(min_r) * 0.5;
    float theta = atan2(edge_vectors(min_r, 1), edge_vectors(min_r, 0));
    planar_matrix(offset_x, offset_y, theta, transform);

    // # we would like to consistently return an OBB with
    // # the largest dimension along the X axis rather than
    // # the long axis being arbitrarily X or Y.
    if (x_extent(min_r) < y_extent(min_r))
    {
        Eigen::Matrix3f flip;
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

template <typename T>
void transform_to_origin(const Eigen::DenseBase<T> &CvxV4d,
                         Eigen::Matrix4f &to_origin,
                         Eigen::Vector3f &min_extents, float &obb_x, float &obb_y, float &obb_z,
                         bool align_axis = true)
{
    // # transform points using our matrix to find the translation for the
    // # transform
    Eigen::MatrixX3f transformedCvxV = (to_origin * CvxV4d.transpose()).transpose().block(0, 0, CvxV4d.rows(), 3);
    auto box_extents = transformedCvxV.colwise().maxCoeff() - transformedCvxV.colwise().minCoeff();
    Eigen::Vector3f box_center = transformedCvxV.colwise().minCoeff() + 0.5 * box_extents;
    to_origin.block(0, 3, 3, 1) = -box_center;
    cout << "min_extents: " << min_extents << "\n box extents: " << box_extents << endl;

    if (align_axis)
    {
        // #return ordered 3D extents
        // x is the longest, then y, then z
        int min_order, max_order, middle_order;
        float min_ext = min_extents.minCoeff(&min_order);
        float max_ext = min_extents.maxCoeff(&max_order);
        for (int i = 0; i < 3; i++)
        {
            if (i != min_order && i != max_order)
            {
                middle_order = i;
            }
        }

        Eigen::Matrix4f flip = Eigen::Matrix4f::Zero();
        int order[3] = {max_order, middle_order, min_order};
        flip(0, order[0]) = 1.0;
        flip(1, order[1]) = 1.0;
        flip(2, order[2]) = 1.0;
        flip(3, 3) = 1.0;
        cout << "flip:\n"
             << flip << endl;
        to_origin = flip * to_origin;
        obb_x = min_extents(max_order);
        obb_y = min_extents(middle_order);
        obb_z = min_extents(min_order);
        cout << "obb: " << obb_x << ", " << obb_y << ", " << obb_z << endl;
        cout << "to_origin:\n"
             << to_origin << endl;
    }
}

void view_obb_min_volume(const Eigen::MatrixXf &V4d, const Eigen::MatrixXi &F, std::string obj_name = "")
{
    // select the view whose obb volume is minumm
    float min_volume = std::numeric_limits<float>::max();
    float max_area = std::numeric_limits<float>::min();
    Eigen::Vector3f min_extents;
    Eigen::Matrix4f min_2D;
    Eigen::Matrix4f rotation_Z;

    for (int i = 0; i < VIEW::NUMS; i++)
    {

        // to_2D: matrices which will rotate each hull normal to [0,0,1]
        Eigen::Matrix4f S, to_2D;
        if (i == FRONT)
        {
            euler_matrix(0, 0, 0, S);
        }
        else if (i == BACK)
        {
            euler_matrix(0, M_PI, 0, S);
        }
        else if (i == LEFT)
        {
            euler_matrix(0, M_PI / 2, 0, S);
        }
        else if (i == RIGHT)
        {
            euler_matrix(0, -M_PI / 2, 0, S);
        }
        else if (i == TOP)
        {
            euler_matrix(M_PI / 2, 0, 0, S);
        }
        else
        {
            euler_matrix(-M_PI / 2, 0, 0, S);
        }

        to_2D = S;

        Eigen::MatrixX3f projected = ((to_2D * V4d.transpose()).transpose()).block(0, 0, V4d.rows(), 3);
        float height = projected.col(2).maxCoeff() - projected.col(2).minCoeff();
        Eigen::Matrix3f rotation_2D;
        float box_x, box_y;
        oriented_bounds_2d(projected.block(0, 0, projected.rows(), 2), box_x, box_y, rotation_2D, i);
        float area = box_x * box_y;
        float volume = area * height;
        cout << "view" << i << ": area=" << area << ", height=" << height << ", volume=" << volume << ", box_x:" << box_x << " box_y:" << box_y << endl;
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
            Eigen::MatrixX3f transformedV = (to_origin * V4d.transpose()).transpose().block(0, 0, V4d.rows(), 3);
            igl::writeOBJ("./result/rot_" + std::to_string(i) + "_" + obj_name + ".obj", transformedV, F);
        }
        Eigen::MatrixX3f projV = (to_2D * V4d.transpose()).transpose().block(0, 0, V4d.rows(), 3);
        igl::writeOBJ("./result/proj_" + std::to_string(i) + "_" + obj_name + ".obj", projV, F);
    }
}

int view_obb_max_area(const Eigen::MatrixXf &V4d, const Eigen::MatrixXi &F, std::string obj_name = "")
{
    using namespace Eigen;
    // select the view whose mask area is maximum
    // return best_view_id
    int best_view_id = VIEW::FRONT;
    float min_volume = std::numeric_limits<float>::max();
    float max_area = std::numeric_limits<float>::min();
    Eigen::Vector3f min_extents;
    Eigen::Matrix4f min_2D;
    Eigen::Matrix4f rotation_Z;

    for (int i = 0; i < VIEW::NUMS; i++)
    {

        // to_2D: matrices which will rotate each hull normal to [0,0,1]
        Eigen::Matrix4f S, to_2D;
        if (i == FRONT)
        {
            euler_matrix(0, 0, 0, S);
        }
        else if (i == BACK)
        {
            euler_matrix(0, M_PI, 0, S);
        }
        else if (i == LEFT)
        {
            euler_matrix(0, M_PI / 2, 0, S);
        }
        else if (i == RIGHT)
        {
            euler_matrix(0, -M_PI / 2, 0, S);
        }
        else if (i == TOP)
        {
            euler_matrix(M_PI / 2, 0, 0, S);
        }
        else
        {
            euler_matrix(-M_PI / 2, 0, 0, S);
        }

        to_2D = S;

        Eigen::MatrixX3f projected = ((to_2D * V4d.transpose()).transpose()).block(0, 0, V4d.rows(), 3);
        float height = projected.col(2).maxCoeff() - projected.col(2).minCoeff();
        Eigen::Matrix3f rotation_2D;
        float box_x, box_y;
        oriented_bounds_2d(projected.block(0, 0, projected.rows(), 2), box_x, box_y, rotation_2D, i);
        float area = box_x * box_y;
        float volume = area * height;
        cout << "view" << i << ": area=" << area << ", height=" << height << ", volume=" << volume << ", box_x:" << box_x << " box_y:" << box_y << endl;
        if (area > max_area)
        {
            best_view_id = i;
            max_area = area;
            min_extents = {box_x, box_y, height};
            min_2D = to_2D;
            rotation_2D(0, 2) = 0.0;
            rotation_2D(1, 2) = 0.0;
            Eigen::Matrix4f rotation_3D;
            planar_matrix_to_3D(rotation_2D, rotation_3D);
            rotation_Z = rotation_3D;
        }
    }

    // # combine the 2D OBB transformation with the 2D projection transform
    Eigen::Matrix4f to_origin = rotation_Z * min_2D;
    float obb_x, obb_y, obb_z;
    transform_to_origin(V4d, to_origin, min_extents, obb_x, obb_y, obb_z, false);

    if (obj_name != "")
    {
        cout << "best_view: " << view_names[best_view_id] << endl;
        // # transform points using our matrix to find the translation for the
        MatrixX3f projV = (min_2D * V4d.transpose()).transpose().block(0, 0, V4d.rows(), 3);
        igl::writeOBJ("./result/max_area_proj_" + std::to_string(best_view_id) + "_" + obj_name + ".obj", projV, F);

        MatrixX3f transformedV = (to_origin * V4d.transpose()).transpose().block(0, 0, V4d.rows(), 3);
        igl::writeOBJ("./result/max_area_rot_" + std::to_string(best_view_id) + "_" + obj_name + ".obj", transformedV, F);
    }

    return best_view_id;
}

int obb(const Eigen::MatrixXf &V4d, const Eigen::MatrixXi &F, std::string obj_name = "")
{
    // return best_view_id which gives the minimum volume OBB

    using namespace std;
    using namespace Eigen;

    int best_view_id = VIEW::FRONT;
    MatrixXf ViewN(6, 3); // view normal // theta, phi
    ViewN << 0, 0, 1,     //front view
        0, 0, -1,         //back
        -1, 0, 0,         //left
        1, 0, 0,          //right
        0, 1, 0,          // top
        0, -1, 0;         //bottom
    MatrixXf CvxV, CvxN;
    MatrixXi CvxF;
    Eigen::Matrix<float, Eigen::Dynamic, 4> CvxV4d;

    igl::copyleft::cgal::convex_hull(V4d.block(0, 0, V4d.rows(), 3), CvxV, CvxF);
    igl::per_face_normals(CvxV, CvxF, CvxN);
    CvxV4d.resize(CvxV.rows(), 4);
    CvxV4d.block(0, 0, CvxV.rows(), 3) = CvxV;
    CvxV4d.col(3) = Eigen::VectorXf::Ones(CvxV4d.rows());
    cout << "CvxV, CvxF=" << CvxV.rows() << ", " << CvxF.rows() << "x" << CvxF.cols() << endl;

    return best_view_id;
}

int main(int argc, char **argv)
{
    using namespace Eigen;
    MatrixXf V;
    MatrixXi F;

    // igl::readOFF("./data/cube.off", V, F);
    std::string obj_name = std::string(argv[1]);
    igl::readOBJ("./data/parts/" + obj_name + ".obj", V, F);
    cout << "V, F=" << V.rows() << ", " << F.rows() << endl;
    cout << V.row(0) << endl;

    // Eigen::Matrix<float, Eigen::Dynamic, 4> V4d;
    // V4d.resize(V.rows(), 4);
    // V4d.block(0, 0, V.rows(), 3) = V;
    // V4d.col(3) = Eigen::VectorXf::Ones(V4d.rows());
    // view_obb_max_area(V4d, F, obj_name);
    // obb(V4d, F, obj_name);
    return 0;
}
