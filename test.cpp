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

void testQR()
{
    typedef typename Eigen::Triplet<double> T;
    SparseMatrix<double> Q(2, 2), A, Q_, R_;
    double theta = M_PI / 4.0;
    vector<T> data;
    data.push_back(T(0, 0, cos(theta)));
    data.push_back(T(0, 1, -sin(theta)));
    data.push_back(T(1, 0, sin(theta)));
    data.push_back(T(1, 1, cos(theta)));
    Q.setFromTriplets(data.begin(), data.end());
    cout << Q.toDense() << endl;
    Q = 1 / sqrt(2) * Q;
    A = 2 * Q;

    SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> solver;
    solver.compute(A);
    if (solver.info() == Success)
    {
        cout << "rank:" << solver.rank() << endl;
        MatrixXd q = solver.matrixQ();
        MatrixXd r = solver.matrixR();
        cout << "Q:\n"
             << q << endl;
        cout << "R:\n"
             << r << endl;
        VectorXd x(2);
        x << 1, 0;
        VectorXd y = A * x;
        VectorXd sol = solver.solve(y);
        cout << "sol:\n"
             << sol << endl;
        cout << "x= Q_t * y if Q is othornmal and Qt*Q*x=Qt*y" << endl;
        MatrixXd p = solver.colsPermutation();
        cout << "p:\n"
             << p << endl;
    }
}

int main()
{
    // testCamera();
    testQR();
    return 0;
}