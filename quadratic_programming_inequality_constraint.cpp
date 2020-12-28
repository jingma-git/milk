
#include <igl/active_set.h>
#include <igl/boundary_facets.h>
#include <igl/cotmatrix.h>
#include <igl/invert_diag.h>
#include <igl/jet.h>
#include <igl/massmatrix.h>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Sparse>
#include <iostream>

using namespace Eigen;
using namespace std;

Eigen::VectorXi b;
Eigen::VectorXd B, bc, lx, ux, Beq, Bieq, Z;
Eigen::SparseMatrix<double> Q, Aeq, Aieq;

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

void solve(igl::opengl::glfw::Viewer &viewer)
{
    using namespace std;
    igl::active_set_params as;
    as.max_iter = 8;
    igl::active_set(Q, B, b, bc, Aeq, Beq, Aieq, Bieq, lx, ux, as, Z);
    viewer.data().set_data(Z);
}

bool key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int mod)
{
    switch (key)
    {
    case '.':
        Beq(0) *= 2.0;
        solve(viewer);
        return true;
    case ',':
        Beq(0) /= 2.0;
        solve(viewer);
        return true;
    case ' ':
        solve(viewer);
        return true;
    default:
        return false;
    }
}

int main(int argc, char *argv[])
{
    using namespace Eigen;
    using namespace std;
    MatrixXd V;
    MatrixXi F;
    // igl::readOFF("data/cheburashka.off", V, F);
    make_plane(V, F);

    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, F);
    viewer.data().show_lines = false;
    viewer.callback_key_down = &key_down;

    // One fixed point
    b.resize(1, 1);
    // point on belly.
    // b << 2556;
    // point on plane center
    b << 12;
    bc.resize(1, 1);
    bc << 1;

    // Construct Laplacian and mass matrix
    SparseMatrix<double> L, M, Minv;
    igl::cotmatrix(V, F, L);
    igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);
    //M = (M/M.diagonal().maxCoeff()).eval();
    igl::invert_diag(M, Minv);
    // Bi-Laplacian
    Q = L.transpose() * (Minv * L);
    // Zero linear term
    B = VectorXd::Zero(V.rows(), 1);

    // Lower and upper bound
    lx = VectorXd::Zero(V.rows(), 1);
    ux = VectorXd::Ones(V.rows(), 1);

    // Equality constraint constrain solution to sum to 1
    Beq.resize(1, 1);
    Beq(0) = 0.08;
    Aeq = M.diagonal().sparseView().transpose();
    // (Empty inequality constraints)
    solve(viewer);
    cout << "Press '.' to increase scale and resolve." << endl
         << "Press ',' to decrease scale and resolve." << endl;

    viewer.launch();
}
