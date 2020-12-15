#include <igl/boundary_facets.h>
#include <igl/unique.h>
#include <igl/colon.h>
#include <igl/setdiff.h>
#include <igl/cotmatrix.h>
#include <igl/slice.h>
#include <igl/doublearea.h>
#include <igl/massmatrix.h>
#include <igl/slice_into.h>

#include <igl/opengl/glfw/Viewer.h>

#include <Eigen/Sparse>

#include <iostream>
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
    const int g_h = 10.0; // global_height
    MatrixXd V;
    MatrixXi F;
    make_plane(V, F);

    //find boundary edges E, vertices b
    MatrixXi E;
    VectorXi b, IA, IC;
    igl::boundary_facets(F, E);
    igl::unique(E, b, IA, IC);
    // List of all vertex indices
    VectorXi all, in;
    igl::colon<int>(0, V.rows() - 1, all);
    // List of interior indices
    igl::setdiff(all, b, in, IA);

    // Construct and slice up Laplacian
    SparseMatrix<double> L, L_all_in;
    igl::cotmatrix(V, F, L);
    igl::slice(L, all, in, L_all_in);
    cout << L << endl;
    cout << L_all_in << endl;

    // Dirichlet boundary conditions from height-field
    VectorXd bc;
    VectorXd h; //height field
    //
    SparseMatrix<double> M;
    igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
    h = M.diagonal() * g_h;

    for (int i = 0; i < b.rows(); i++)
    {
        h(b(i)) = 0.0;
    }
    // cout << "height" << endl;
    // cout << h << endl;

    // // solve PDE
    Eigen::SparseQR<Eigen::SparseMatrix<double>, COLAMDOrdering<int>> solver(-L_all_in);
    VectorXd h_sol = solver.solve(h);

    for (int i = 0; i < in.rows(); i++)
    {
        V(in(i), 2) = h_sol(i);
        cout << i << ": " << V(in(i), 2) << endl;
    }

    // copy
    // Plot the mesh with pseudocolors
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, F);
    viewer.data().show_lines = false;
    // viewer.data().set_data(Z);
    viewer.launch();
    return 0;
}