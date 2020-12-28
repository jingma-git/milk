#include <igl/boundary_facets.h>
#include <igl/cotmatrix.h>
#include <igl/invert_diag.h>
#include <igl/jet.h>
#include <igl/massmatrix.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Sparse>
#include <iostream>

using namespace Eigen;
using namespace std;

void make_plane(MatrixXd &V, MatrixXi &F)
{
    const int N = 1;
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

int main(int argc, char *argv[])
{
    MatrixXd V;
    MatrixXi F;
    // igl::readOFF("data/cheburashka.off", V, F);
    // cout << "V: " << V.rows() << ", F: " << F.rows() << endl;
    make_plane(V, F);
    cout << "V: " << V.rows() << ", F: " << F.rows() << endl;

    // // Two fixed points
    // VectorXi b(2, 1);
    // // Left hand, left foot
    // b << 4331, 5957;
    // VectorXd bc(2, 1);
    // bc << 1, -1;

    // Two fixed points
    VectorXi b(2, 1);
    // bottom left, upper right
    b << 0, V.rows() - 1;
    VectorXd bc(2, 1);
    bc << 1, -1;

    // Construct Laplacian and mass matrix
    SparseMatrix<double> L, M, Minv, Q;
    igl::cotmatrix(V, F, L);
    igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);
    igl::invert_diag(M, Minv);
    // Bi-Laplacian
    Q = L * (Minv * L);
    cout << "L" << endl
         << L << endl
         << "M" << endl
         << M << endl
         << " Bi-L" << endl
         << Q << endl;
    // Zero linear term
    VectorXd B = VectorXd::Zero(V.rows(), 1);

    VectorXd Z, Z_const;
    {
        // Alternative, short hand
        igl::min_quad_with_fixed_data<double> mqwf;
        // Empty constraints
        VectorXd Beq;
        SparseMatrix<double> Aeq;
        igl::min_quad_with_fixed_precompute(Q, b, Aeq, true, mqwf);
        igl::min_quad_with_fixed_solve(mqwf, B, bc, Beq, Z);
        cout << "mqwf.Auu:\n";
        cout << mqwf.Auu << endl; // 2x2
        cout << "mqwf.preY:\n";
        cout << mqwf.preY << endl; // 2x1
        cout << "Z:" << endl;
        cout << Z << endl;
        MatrixXd L = mqwf.llt.matrixL();
        MatrixXd U = mqwf.llt.matrixU();
        cout << "L: " << endl;
        cout << L << endl;
        cout << "U: " << endl;
        cout << U << endl;
    }

    {
        cout << "------------------- Aeq ------------------------" << endl;
        igl::min_quad_with_fixed_data<double> mqwf;
        // Constraint forcing difference of two points to be 0
        SparseMatrix<double> Aeq(1, V.rows());
        // // Right hand, right foot
        // Aeq.insert(0, 6074) = 1;
        // Aeq.insert(0, 6523) = -1;

        // vertex 8 at upper right corner, vertex 2 at bottom right corner
        Aeq.insert(0, V.rows() - 1) = 1;
        Aeq.insert(0, 2) = -1;
        Aeq.makeCompressed();

        VectorXd Beq(1, 1);
        Beq(0) = 0;
        igl::min_quad_with_fixed_precompute(Q, b, Aeq, true, mqwf);
        cout << "mqwf.Aequ:\n"
             << mqwf.Aequ << endl;
        cout << "mqwf.NA:\n";
        cout << mqwf.NA << endl; // 3 x 3
        cout << "mqwf.preY:\n";
        cout << mqwf.preY << endl; // 2x1
        igl::min_quad_with_fixed_solve(mqwf, B, bc, Beq, Z_const);
        cout << "Z:" << endl;
        cout << Z_const << endl;
    }

    // Use same color axes
    const double min_z = std::min(Z.minCoeff(), Z_const.minCoeff());
    const double max_z = std::max(Z.maxCoeff(), Z_const.maxCoeff());

    // Plot the mesh with pseudocolors
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, F);
    viewer.data().show_lines = false;
    viewer.data().set_data(Z, min_z, max_z);

    viewer.callback_key_down =
        [&Z, &Z_const, &min_z, &max_z](igl::opengl::glfw::Viewer &viewer, unsigned char key, int mod) -> bool {
        if (key == ' ')
        {
            static bool toggle = true;
            viewer.data().set_data(toggle ? Z_const : Z, min_z, max_z);
            toggle = !toggle;
            return true;
        }
        else
        {
            return false;
        }
    };
    cout << "Press [space] to toggle between unconstrained and constrained." << endl;
    viewer.launch();
}
