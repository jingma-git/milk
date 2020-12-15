#include <iostream>
#include <cmath>
#include "transform.h"
#include "matrix_to_list.h"
using namespace std;
using namespace Eigen;

void testAtan2()
{
    cout << atan2(+0., +0.) << endl;
    cout << atan2(-0., +0.) << endl;
    cout << atan2(+0., -0.) << endl;
    cout << atan2(-0., -0.) << endl;
    cout << "===================" << endl;
    cout << atan2(0.0, +0.0) << endl;
    cout << atan2(0.0, -0.0) << endl;
    cout << "===================" << endl;
    cout << atan2(1., 0.) << endl;
    cout << atan2(-1., 0.) << endl;
    cout << atan2(1., -0.) << endl;
    cout << atan2(-1., -0.) << endl;
}

void testLookAt()
{
    using namespace Eigen;

    Vector3f eye(0, 0, 2);
    Vector3f target(0, 0, 0);
    Vector3f up(0, 1, 0);
    Matrix4f view;
    look_at(eye, target, up, view);

    cout << view << endl;
}

void testMatrixToList()
{
    MatrixXf V(2, 3);
    V << 0, 0, 0,
        0, 1, 0;
    vector<Vector3f> v_list;
    matrix_to_list(V, v_list);
    for (int i = 0; i < V.rows(); i++)
    {
        cout << i << ": " << v_list[i].transpose() << endl;
    }

    MatrixXi F(2, 3);
    F << 0, 1, 2,
        1, 2, 3;
    vector<Vector3i> f_list;
    matrix_to_list(F, f_list);
    for (int i = 0; i < F.rows(); i++)
    {
        cout << i << ": " << f_list[i].transpose() << endl;
    }
}

void testColwise()
{
    MatrixXf V = MatrixXf::Random(2, 3);
    cout << "V\n"
         << V << endl;
    Vector3f scale(1, 2, 3);
    // V.col(0) *= scale(0);
    // V.col(1) *= scale(1);
    // V.col(2) *= scale(2);
    cout << (scale.transpose() * V.transpose()).transpose() << endl;
    cout << "Scale\n"
         << V << endl;
}

void testRound()
{
    float x = 0.;
    int y = static_cast<int>(x + 0.5) - 1;
    cout << y << endl;
}

void testPixelCoord()
{
    float multiplier = 1000;
    float width = 256;
    for (int wididx = 0; wididx < width; wididx++)
    {
        float x0 = 1.0 / width * (2 * wididx + 1 - width);
        cout << wididx << ": " << x0 << endl;
    }
}

void testCamera()
{
    Vector3f center(0, 0, 1);
    Matrix3f R = Matrix3f::Identity();
    // euler_matrix(0, 180 / M_PI, 0, R);
    Matrix4f M;
    inverse_extrinsic_matrix(R, center, M);
    cout << M << endl;
}
int main()
{
    testCamera();
    return 0;
}