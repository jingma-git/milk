#include <iostream>
#include <igl/grad.h>
#include <igl/doublearea.h>
#include <igl/cotmatrix.h>
using namespace std;
using namespace Eigen;

void make_plane(MatrixXd &V, MatrixXi &F)
{
    const int N = 4;
    const double len = 1.0 / N;
    int v_num = (N + 1) * (N + 1);
    V.resize(v_num, 3);
    F.resize(2 * N * N, 3);

    for (int v_idx = 0; v_idx < v_num; v_idx++)
    {
        int r = v_idx / (N + 1);
        int c = v_idx % (N + 1);
        V.row(v_idx) = RowVector3d(r * len, c * len, 0);
    }

    int f_idx = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int v0 = i * (N + 1) + j;
            int v1 = i * (N + 1) + (j + 1);
            int v2 = (i + 1) * (N + 1) + (j + 1);
            int v3 = (i + 1) * (N + 1) + j;

            F.row(f_idx) = RowVector3i(v0, v2, v1);
            F.row(f_idx + 1) = RowVector3i(v0, v3, v2);
            f_idx += 2;
        }
    }
}

int main()
{
    Eigen::MatrixXd V, U;
    Eigen::MatrixXi F;
    make_plane(V, F);
    cout << "m faces: " << F.rows() << ", n vertices: " << V.rows() << endl;
    SparseMatrix<double> L;
    igl::cotmatrix(V, F, L);

    SparseMatrix<double> G, K;
    igl::grad(V, F, G);
    cout << "G: " << G.rows() << ", " << G.cols() << endl;
    // cout << G << endl;

    VectorXd dblA;
    igl::doublearea(V, F, dblA);                               // twice the area for each input triangle;
    const auto &T = (dblA.replicate(3, 1) * 0.5).asDiagonal(); // triangle true area: m*3 x m*3
    cout << "T:" << T.rows() << ", " << T.cols() << endl;

    K = -G.transpose() * T * G; // the real laplacian is negative on the diagonal, so add '-' to make the matrix positive
    cout << "error: " << (K - L).norm() << endl;
    return 0;
}