#pragma once
#include <eigen3/Eigen/Eigen>
#include "mask_put.h"

inline void euler_matrix(float ai, float aj, float ak, Eigen::Matrix4f &M)
{
    // rotation around x, y, z
    float si = sin(ai);
    float sj = sin(aj);
    float sk = sin(ak);

    float ci = cos(ai);
    float cj = cos(aj);
    float ck = cos(ak);

    float cc = ci * ck;
    float cs = ci * sk;
    float sc = si * ck;
    float ss = si * sk;

    int i = 0, j = 1, k = 2;
    M = Eigen::Matrix4f::Identity();
    M(i, i) = cj * ck;
    M(i, j) = sj * sc - cs;
    M(i, k) = sj * cc + ss;
    M(j, i) = cj * sk;
    M(j, j) = sj * ss + cc;
    M(j, k) = sj * cs - sc;
    M(k, i) = -sj;
    M(k, j) = cj * si;
    M(k, k) = cj * ci;
}

inline void spherical_matrix(float theta, float phi, Eigen::Matrix4f &mat)
{
    // Give a spherical coordinate vector, find the rotation that will
    // transform a [0,0,1] vector to those coordinates

    // Parameters
    // -----------
    // theta: float, rotation angle in radians around positive y axis
    // phi:   float, rotation angle in radians around positive z axis

    // Returns
    // ----------
    // matrix: (4,4) rotation matrix where the following will
    //          be a cartesian vector in the direction of the
    //          input spherical coordinates:
    //             np.dot(matrix, [0,0,1,0])
    euler_matrix(0.0, theta, phi, mat);
}

inline void planar_matrix(float offset_x, float offset_y, float theta, Eigen::Matrix3f &M)
{
    // """
    // 2D homogeonous transformation matrix.

    // Parameters
    // ----------
    // offset : (2,) float
    //   XY offset
    // theta : float
    //   Rotation around Z in radians
    //  Returns
    // ----------
    // matrix : (3, 3) flat
    //   Homogeneous 2D transformation matrix
    // """

    using namespace Eigen;
    M = Matrix3f::Identity();
    float s = sin(theta);
    float c = cos(theta);

    M(0, 0) = c;
    M(0, 1) = -s;
    M(1, 0) = s;
    M(1, 1) = c;
    M(0, 2) = offset_x;
    M(1, 2) = offset_y;
}

inline void planar_matrix(float theta, Eigen::Matrix3f &M)
{
    // """
    // 2D homogeonous transformation matrix.

    // Parameters
    // ----------
    // theta : float
    //   Rotation around Z in radians
    //  Returns
    // ----------
    // matrix : (3, 3) flat
    //   Homogeneous 2D transformation matrix
    // """

    using namespace Eigen;
    M = Matrix3f::Identity();
    float s = sin(theta);
    float c = cos(theta);

    M(0, 0) = c;
    M(0, 1) = -s;
    M(1, 0) = s;
    M(1, 1) = c;
}

inline void planar_matrix_to_3D(const Eigen::Matrix3f &rotation_2D, Eigen::Matrix4f &rotation_3D)
{
    // """
    // Given a 2D homogeneous rotation matrix convert it to a 3D rotation
    // matrix that is rotating around the Z axis

    // Parameters
    // ----------
    // matrix_2D: (3,3) float, homogeneous 2D rotation matrix

    // Returns
    // ----------
    // matrix_3D: (4,4) float, homogeneous 3D rotation matrix
    // """
    using namespace Eigen;
    rotation_3D = Matrix4f::Identity();
    // translation
    rotation_3D(0, 3) = rotation_2D(0, 2);
    rotation_3D(1, 3) = rotation_2D(1, 2);
    // rotation
    rotation_3D(0, 0) = rotation_2D(0, 0);
    rotation_3D(0, 1) = rotation_2D(0, 1);
    rotation_3D(1, 0) = rotation_2D(1, 0);
    rotation_3D(1, 1) = rotation_2D(1, 1);
}

inline void vector_to_spherical(const Eigen::MatrixXf &cartesian, Eigen::MatrixXf &spherical_coords)
{
    // convert cartesian points to spherical unit vectors
    assert(cartesian.cols() == 3);
    spherical_coords.resize(cartesian.rows(), 2);

    auto unit = cartesian.rowwise().normalized();
    auto x = unit.col(0);
    auto y = unit.col(1);
    auto z = unit.col(2);
    // cout << unit << endl;

    for (int i = 0; i < cartesian.rows(); i++)
    {
        spherical_coords(i, 0) = acos(z(i));        // theta, rotation around positive y axis, starting from positive z axis
        spherical_coords(i, 1) = atan2(y(i), x(i)); // phi, rotation around positive z axis, starting from positive x axis
    }
}

inline void vector_hemisphere(const Eigen::MatrixXf &N, Eigen::MatrixXf &OrientedN)
{
    using namespace Eigen;
    const double TOL_ZERO = 1e-13;
    auto negative = N.array() < -TOL_ZERO;
    auto zero = (N.array() > -TOL_ZERO) && (N.array() < TOL_ZERO);
    // cout << negative << endl;
    // cout << zero << endl;

    // move all negative Z to positive
    // for zero Z vectors, move all negative Y to positive
    // for zero Y vectors, move all negative X to positive
    VectorXf signs = VectorXf::Ones(N.rows());
    mask_put(signs, negative.col(2), -1.0);
    // all on-plane vectors with negative Y values
    mask_put(signs, (zero.col(2).array() && negative.col(1).array()), -1.0);
    // all on-plane vectors with zero Y values and negative X values
    mask_put(signs, (zero.col(2).array() && zero.col(1).array() && negative.col(0).array()), -1.0);

    OrientedN = N.array().colwise() * signs.array();
}

inline void look_at(const Eigen::Vector3f &eye,
                    const Eigen::Vector3f &target,
                    const Eigen::Vector3f &up,
                    Eigen::Matrix4f &view)
{
    using namespace Eigen;
    using namespace std;
    view = Matrix4f::Identity();

    Vector3f f = (eye - target).normalized(); //front
    Vector3f s = up.cross(f).normalized();
    Vector3f u = f.cross(s);

    view(0, 0) = s.x();
    view(0, 1) = s.y();
    view(0, 2) = s.z();

    view(1, 0) = u.x();
    view(1, 1) = u.y();
    view(1, 2) = u.z();

    view(2, 0) = f.x();
    view(2, 1) = f.y();
    view(2, 2) = f.z();

    view(0, 3) = -s.dot(eye);
    view(1, 3) = -u.dot(eye);
    view(2, 3) = -f.dot(eye);
}

void ortho_proj(float r, float t, float zNear, float zFar, Eigen::Matrix4f &proj)
{
    proj = Eigen::Matrix4f::Identity();
    proj(0, 0) = 1.0 / r;
    proj(1, 1) = 1.0 / t;
    proj(2, 2) = -2.0 / (zFar - zNear);
    proj(2, 3) = -(zFar + zNear) / (zFar - zNear);
}

void perspective_proj(float r, float t, float zNear, float zFar, Eigen::Matrix4f &proj)
{
    assert(zNear > 0 && "You should not place the near plane at zero position, nothing will show!");
    proj = Eigen::Matrix4f::Identity();
    proj(0, 0) = zNear / r;
    proj(1, 1) = zNear / t;
    proj(2, 2) = -(zFar + zNear) / (zFar - zNear);
    proj(2, 3) = -2 * zFar * zNear / (zFar - zNear);
    proj(3, 2) = -1;
}