#include <igl/readTGF.h>
#include <igl/directed_edge_parents.h>
#include <igl/readMESH.h>
#include <igl/readDMAT.h>
#include <igl/boundary_conditions.h>
#include <igl/column_to_quats.h>
#include <igl/forward_kinematics.h>
#include <igl/normalize_row_sums.h>
#include <igl/lbs_matrix.h>
#include <igl/active_set.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/harmonic.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/deform_skeleton.h>
#include <igl/min_quad_with_fixed.h>

#include <iostream>
using namespace std;
using namespace Eigen;

typedef vector<Quaterniond, aligned_allocator<Quaterniond>> RotationList;
const Eigen::RowVector3d sea_green(70. / 255., 252. / 255., 167. / 255.);

MatrixXd V, W, U, C, Q, M;
MatrixXi T, F, BE;
VectorXi P;
RotationList pose;
double anim_t = 1.0;
double anim_t_dir = -0.03;
int selected = 0;
struct BBW_data
{
    igl::active_set_params active_set_params;
};

void bbw(const MatrixXd &V,
         const MatrixXi &T,
         const VectorXi &b,
         const MatrixXd &bc,
         BBW_data &bbw_data,
         MatrixXd &W)
{
    int n = V.rows();  // vertex nums
    int m = bc.cols(); // bone nums

    // bi-laplacian operator
    SparseMatrix<double> L, M, M_inv, Bi_l;
    igl::cotmatrix(V, T, L);
    igl::massmatrix(V, T, igl::MASSMATRIX_TYPE_DEFAULT, M);
    igl::invert_diag(M, M_inv);
    Bi_l = L * M_inv * L;
    cout << "Bi-laplace: " << Q.rows() << ", " << Q.cols() << endl;

    igl::active_set_params eff_params = bbw_data.active_set_params;

    // setup constraint for active set
    VectorXd lw = VectorXd::Zero(n);
    VectorXd uw = VectorXd::Ones(n);
    VectorXd B = VectorXd::Zero(n);
    SparseMatrix<double> Aeq(0, n), Aieq(0, n);
    VectorXd Beq(0, 1), Bieq(0, 1);

    // initial guess
    igl::min_quad_with_fixed_data<double> mqwf;
    igl::min_quad_with_fixed_precompute(Bi_l, b, Aeq, true, mqwf);
    igl::min_quad_with_fixed_solve(mqwf, B, bc, Beq, W);

    for (int i = 0; i < m; i++)
    {
        VectorXd Wi;
        Wi = W.col(i);
        VectorXd bci = bc.col(i);
        cout << "solve for handle " << i << endl;

        igl::SolverStatus ret = igl::active_set(Bi_l, B, b, bci, Aeq, Beq, Aieq, Bieq, lw, uw, eff_params, Wi);
        switch (ret)
        {
        case igl::SOLVER_STATUS_CONVERGED:
            break;
        case igl::SOLVER_STATUS_MAX_ITER:
            cerr << "active_set: max iter without convergence." << endl;
            break;
        case igl::SOLVER_STATUS_ERROR:
        default:
            cerr << "active_set error." << endl;
            break;
        }

        W.col(i) = Wi;
    }
}

bool pre_draw(igl::opengl::glfw::Viewer &viewer)
{
    using namespace Eigen;
    using namespace std;
    if (viewer.core().is_animating)
    {
        // Interpolate pose and identity
        RotationList anim_pose(pose.size());
        for (int e = 0; e < pose.size(); e++)
        {
            anim_pose[e] = pose[e].slerp(anim_t, Quaterniond::Identity());
            // anim_pose[e] = pose[e];
        }
        // Propagate relative rotations via FK to retrieve absolute transformations
        RotationList vQ;
        vector<Vector3d> vT;
        igl::forward_kinematics(C, BE, P, anim_pose, vQ, vT);
        const int dim = C.cols();
        MatrixXd T(BE.rows() * (dim + 1), dim);
        for (int e = 0; e < BE.rows(); e++)
        {
            Affine3d a = Affine3d::Identity();
            a.translate(vT[e]);
            a.rotate(vQ[e]);
            T.block(e * (dim + 1), 0, dim + 1, dim) =
                a.matrix().transpose().block(0, 0, dim + 1, dim);
        }
        // Compute deformation via LBS as matrix multiplication
        U = M * T;

        // Also deform skeleton edges
        MatrixXd CT;
        MatrixXi BET;
        igl::deform_skeleton(C, BE, T, CT, BET);

        viewer.data().set_vertices(U);
        viewer.data().set_edges(CT, BET, sea_green);
        viewer.data().compute_normals();
        anim_t += anim_t_dir;
        anim_t_dir *= (anim_t >= 1.0 || anim_t <= 0.0 ? -1.0 : 1.0);
    }
    return false;
}

bool key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int mods)
{
    switch (key)
    {
    case ' ':
        viewer.core().is_animating = !viewer.core().is_animating;
        break;
    case '.':
        selected++;
        selected = std::min(std::max(selected, 0), (int)W.cols() - 1);
        viewer.data().set_data(W.col(selected));
        break;
    case ',':
        selected--;
        selected = std::min(std::max(selected, 0), (int)W.cols() - 1);
        viewer.data().set_data(W.col(selected));
        break;
    }
    return true;
}

int main()
{
    igl::readMESH("/home/server/MaJing/cpp_proj/milk/data/hand.mesh", V, T, F);
    U = V;
    cout << "V: " << V.rows() << " T: " << T.rows() << ", " << T.cols() << " F: " << F.rows() << "," << F.cols() << endl;

    igl::readTGF("/home/server/MaJing/cpp_proj/milk/data/hand.tgf", C, BE);
    cout << "joints: " << C.rows() << endl;
    cout << "bones: " << BE.rows() << ", " << BE.cols() << endl;

    igl::directed_edge_parents(BE, P);
    igl::readDMAT("/home/server/MaJing/cpp_proj/milk/data/hand-pose.dmat", Q);
    igl::column_to_quats(Q, pose);
    cout << "joint pose: " << pose.size() << endl;

    VectorXi b;
    MatrixXd bc;
    // project vertex to Bone Edges, if Bone is far from Mesh Surface for triangular mesh, there won't be boundary conditions
    // b:  #b list of boundary indices (indices into V of vertices which have known, fixed values
    // bc: #b by #weights list of known/fixed values for boundary vertices
    igl::boundary_conditions(V, T, C, VectorXi(), BE, MatrixXi(), b, bc);
    cout << "boundary weights: " << bc.rows() << "," << bc.cols() << endl;

    // compute weight matrix: #bones x #vertices
    BBW_data bbw_data;
    bbw_data.active_set_params.max_iter = 8;
    bbw(V, T, b, bc, bbw_data, W);

    // Normalize weights to sum to one
    igl::normalize_row_sums(W, W);
    // precompute linear blend skinning matrix
    igl::lbs_matrix(V, W, M);

    // while (true)
    // {
    //     RotationList anim_pose(pose.size());
    //     for (int e = 0; e < pose.size(); e++)
    //     {
    //         anim_pose[e] = pose[e].slerp(anim_t, Quaterniond::Identity());
    //     }
    //     RotationList vQ;
    //     vector<Vector3d> vT;
    //     igl::forward_kinematics(C, BE, P, anim_pose, vQ, vT);

    //     const int dim = C.cols();
    //     MatrixXd T(BE.rows() * (dim + 1), dim); //bone transformation matrix, #bones*(3+1) x 3
    //     for (int e = 0; e < BE.rows(); e++)
    //     {
    //         Affine3d a = Affine3d::Identity();
    //         a.translate(vT[e]);
    //         a.rotate(vQ[e]);
    //         T.block(e * (dim + 1), 0, dim + 1, dim) = a.matrix().transpose().block(0, 0, dim + 1, dim);
    //     }

    //     U = M * T;

    //     anim_t += anim_t_dir;
    //     anim_t_dir *= (anim_t >= 1.0 || anim_t <= 0.0 ? -1.0 : 1.0);
    //     if (anim_t < 0)
    //         break;
    // }

    // Plot the mesh with pseudocolors
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(U, F);
    viewer.data().set_data(W.col(selected));
    viewer.data().set_edges(C, BE, sea_green);
    viewer.data().show_lines = false;
    viewer.data().show_overlay_depth = false;
    viewer.data().line_width = 1;
    viewer.callback_pre_draw = &pre_draw;
    viewer.callback_key_down = &key_down;
    viewer.core().is_animating = false;
    viewer.core().animation_max_fps = 30.;
    cout << "Press '.' to show next weight function." << endl
         << "Press ',' to show previous weight function." << endl
         << "Press [space] to toggle animation." << endl;
    viewer.launch();
    return 0;
}