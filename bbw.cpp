#include <igl/readDMAT.h>
#include <igl/readTGF.h>
#include <igl/readMESH.h>

#include <Eigen/Eigen>
#include <Eigen/Geometry>

#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/colon.h>
#include <igl/setdiff.h>
#include <igl/project_to_line.h>
#include <igl/EPS.h>

#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/bbw.h>

#include <igl/opengl/glfw/Viewer.h>

#include <map>

using namespace Eigen;
using namespace std;

typedef std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> RotationList;
bool column_to_quats(const Eigen::VectorXd &Q,
                     RotationList &vQ)
{
    if (Q.size() % 4 != 0)
    {
        return false;
    }

    const int nQ = Q.size() / 4;
    vQ.resize(nQ);
    for (int q = 0; q < nQ; q++)
    {
        vQ[q] = Eigen::Quaterniond(Q(q * 4 + 3), Q(q * 4 + 0), Q(q * 4 + 1), Q(q * 4 + 2));
    }
    return true;
}

void directed_edge_parents(const MatrixXi &E, MatrixXi &P)
{
    VectorXi I = VectorXi::Constant(E.maxCoeff() + 1, 1, -1), roots, _;
    using namespace igl;
    slice_into(colon<int>(0, E.rows() - 1), E.col(1), I); //number on position i, represents the directed vertex
    setdiff(E.col(0), E.col(1), roots, _);
    std::for_each(roots.data(), roots.data() + roots.size(), [&](int r) { I(r) = -1; });
    slice(I, E.col(0), P);
}

// Inputs:
//   V  #V by dim list of domain vertices
//   Ele  #Ele by simplex-size list of simplex indices
//   C  #C by dim list of handle positions
//   P  #P by 1 list of point handle indices into C
//   BE  #BE by 2 list of bone edge indices into C
//   CE  #CE by 2 list of cage edge indices into *P*
// Outputs:
//   b  #b list of boundary indices (indices into V of vertices which have
//     known, fixed values)
//   bc #b by #weights list of known/fixed values for boundary vertices
//     (notice the #b != #weights in general because #b will include all the
//     intermediary samples along each bone, etc.. The ordering of the
//     weights corresponds to [P;BE]
// Returns false if boundary conditions are suspicious:
//   P and BE are empty
//   bc is empty
//   some column of bc doesn't have a 0 (assuming bc has >1 columns)
//   some column of bc doesn't have a 1 (assuming bc has >1 columns)
bool boundary_conditions(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi & /*Ele*/,
    const Eigen::MatrixXd &C,
    const Eigen::VectorXi &P,
    const Eigen::MatrixXi &BE,
    const Eigen::MatrixXi &CE,
    Eigen::VectorXi &b,
    Eigen::MatrixXd &bc)
{
    using namespace igl;
    vector<int> bci;
    vector<int> bcj;
    vector<double> bcv;

    //TODO point handles
    for (int p = 0; p < P.size(); p++)
    {
    }

    //bone edges
    for (int e = 0; e < BE.rows(); e++)
    {
        for (int i = 0; i < V.rows(); i++)
        {
            VectorXd tip = C.row(BE(e, 0));
            VectorXd tail = C.row(BE(e, 1));

            double t, sqrd;
            igl::project_to_line(
                V(i, 0), V(i, 1), V(i, 2),
                tip(0), tip(1), tip(2),
                tail(0), tail(1), tail(2),
                t, sqrd);
            if (t >= -FLOAT_EPS && t <= (1.0f + FLOAT_EPS) && sqrd <= FLOAT_EPS)
            {
                bci.push_back(i);
                bcj.push_back(P.size() + e);
                bcv.push_back(1.0);
            }
        }
    }

    //TODO cage edges
    for (int e = 0; e < CE.rows(); e++)
    {
    }

    // find unique boundary indices
    vector<int> vb = bci;
    sort(vb.begin(), vb.end());
    vb.erase(std::unique(vb.begin(), vb.end()), vb.end());

    b.resize(vb.size());
    bc = MatrixXd::Zero(vb.size(), P.size() + BE.rows());
    map<int, int> bim; // map from boundary index to index in boundary
    int i = 0;
    for (vector<int>::iterator bit = vb.begin(); bit != vb.end(); bit++)
    {
        b(i) = *bit;
        bim[*bit] = i;
        i++;
    }

    for (i = 0; i < (int)bci.size(); i++)
    {
        int b_i = bim[bci[i]];
        int b_j = bcj[i];
        bc(b_i, b_j) = bcv[i];
    }

    // Normalize across rows so that conditions sum to one
    for (i = 0; i < bc.rows(); i++)
    {
        double sum = bc.row(i).sum();
        bc.row(i) /= sum;
    }

    // Check every weight function has at least one boundary value of 1 and one valuee of 0
}

void harmonic(const MatrixXd &V, const MatrixXi &F,
              const int k, SparseMatrix<double> &Q)
{
    SparseMatrix<double> L, M;
    igl::cotmatrix(V, F, L);
    if (k > 1)
    {
        igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_DEFAULT, M);
    }
    Q = -L;
    if (k == 1)
        return;
    SparseMatrix<double> Mi;
    igl::invert_diag(M, Mi);

    for (int p = 1; p < k; p++)
    {
        Q = (Q * Mi * -L);
    }
}

void min_quad_solve(const SparseMatrix<double> &L, const VectorXi &b,
                    const MatrixXd &bc, MatrixXd &sol)
{
    using namespace igl;
    SparseMatrix<double> L_in_in, L_in_b;
    Eigen::VectorXi all = colon<int>(0, L.rows() - 1), in, _;

    setdiff(all, b, in, _);
    assert(b.rows() + in.rows() == all.rows());
    slice(L, in, in, L_in_in);
    slice(L, in, b, L_in_b);

    // Solve PDE
    SimplicialLLT<SparseMatrix<double>> solver(-L_in_in);
    MatrixXd Z_in = solver.solve(L_in_b * bc);
    for (int i = 0; i < Z_in.rows(); i++)
    {
        sol(in(i)) = Z_in(i);
    }
    for (int i = 0; i < b.rows(); i++)
    {
        sol(b(i)) = bc(i);
    }
}

enum
{
    CONVERGED = 0,
    ERROR,
    MAX_ITER
};
// A_unkown * Z_unknown = -A_known * Y
int active_set(const SparseMatrix<double> &A, const VectorXi &known,
               const MatrixXd &Y, const VectorXd &p_lx, const VectorXd &p_ux, VectorXd &Z)
{
    const int n = A.rows();
    const int nk = known.size();
    VectorXd lx = p_lx;
    VectorXd ux = p_ux;

    // initialize active set
    VectorXi as_lx = VectorXi::Constant(n, 1, false);
    VectorXi as_ux = VectorXi::Constant(n, 1, false);

    VectorXd old_Z = VectorXd::Constant(n, 1, std::numeric_limits<double>::max());
    int iter = 0;
    int status;
    while (true)
    {
        // Find Breaches of constants
        int new_as_lx = 0;
        int new_as_ux = 0;
        for (int z = 0; z < n; z++)
        {
            auto it = std::find(known.data(), known.data() + nk, z);
            if (it != known.data() + nk)
            {
                cout << "contains " << z << " " << (*it) << endl;
                continue;
            }

            if (Z(z) < lx(z))
            {
                new_as_lx += (as_lx(z) ? 0 : 1);
                as_lx(z) = true; // if weight is larger than 0, activate it, and solve
            }

            if (Z(z) > ux(z))
            {
                new_as_ux += (as_ux(z) ? 0 : 1);
                as_ux(z) = true; // if weight is larger than 0, activate it, and solve
            }
        }

        const double diff = (Z - old_Z).squaredNorm();
        if (diff < 1e-7)
        {
            status = CONVERGED;
            break;
        }

        old_Z = Z;

        const int as_lx_count = std::count(as_lx.data(), as_lx.data() + n, true);
        const int as_ux_count = std::count(as_ux.data(), as_ux.data() + n, true);
        cout << "as_lx_count=" << as_lx_count << endl;
        cout << "as_ux_count=" << as_ux_count << endl;

        VectorXi known_i;
        known_i.resize(nk + as_lx_count + as_ux_count);
        VectorXd Y_i;
        Y_i.resize(nk + as_lx_count + as_ux_count);
        cout << "known: " << known.rows() << ", " << known.cols() << endl;
        known_i.block(0, 0, known.rows(), known.cols()) = known.block(0, 0, known.rows(), known.cols());
        cout << "Y: " << Y.rows() << ", " << Y.cols() << endl;
        Y_i.block(0, 0, Y.rows(), Y.cols()) = Y.block(0, 0, Y.rows(), Y.cols());
        cout << "Y1: " << Y.rows() << ", " << Y.cols() << endl;

        int k = nk;
        for (int z = 0; z < n; z++)
        {
            if (as_lx(z))
            {
                known_i(k) = z;
                Y_i(k) = lx(z);
                k++;
            }
        }
        for (int z = 0; z < n; z++)
        {
            if (as_ux(z))
            {
                known_i(k) = z;
                Y_i(k) = ux(z);
                k++;
            }
        }
        assert(k == Y_i.size());

        MatrixXd sol;
        sol.resize(n, 1);
        min_quad_solve(A, known_i, Y_i, sol);

        Z = sol.col(0);
        iter++;
        if (iter > 8)
        {
            status = MAX_ITER;
            break;
        }
    }
    return status;
}

bool bbw(const MatrixXd &V, const MatrixXi &Ele,
         const MatrixXi &b, const MatrixXd &bc, MatrixXd &W)
{
    int n = V.rows();
    int m = bc.cols();

    SparseMatrix<double> Q;
    harmonic(V, Ele, 2, Q);

    W.resize(n, m);
    min_quad_solve(Q, b, bc, W);

    VectorXd ux = VectorXd::Ones(n);
    VectorXd lx = VectorXd::Zero(n);

    for (int i = 0; i < m; i++)
    {
        cout << "compute weight for handle " << i << " out of " << m << " boundary condition=" << b.rows() << endl;
        Eigen::VectorXd z = W.col(i);
        int status = active_set(Q, b, bc.col(i), lx, ux, z);
        if (status == CONVERGED)
        {
            cout << "handle " << i << " converged\n";
        }
        else if (status == MAX_ITER)
        {
            cout << "handle " << i << " surpass max iter\n";
        }
        W.col(i) = z;
    }
}

void normalize_row_sums(const Eigen::MatrixXd &A, MatrixXd &B)
{
    B = (A.array().colwise() / A.rowwise().sum().array());
}

void lbs_matrix(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXd &W,
    Eigen::MatrixXd &M)
{
    const int dim = V.cols();
    const int n = V.rows();
    const int m = W.cols();

    M.resize(n, (dim + 1) * m);
    for (int j = 0; j < m; j++)
    {
        VectorXd Wj = W.block(0, j, V.rows(), 1);
        for (int i = 0; i < (dim + 1); i++)
        {
            if (i < dim)
            {
                M.col(i + j * (dim + 1)) =
                    Wj.cwiseProduct(V.col(i));
            }
            else
            {
                M.col(i + j * (dim + 1)).array() = W.block(0, j, V.rows(), 1).array();
            }
        }
    }
}

MatrixXd V, U;
MatrixXi T, F;

MatrixXd C;
MatrixXi BE, P; //Control vertices, boundary edges

MatrixXd Q;
RotationList pose;

VectorXi b;
MatrixXd bc;

MatrixXd W, M; // #vertices by #bone_edges
int selected = 0;
double anim_t = 1.0;
double anim_t_dir = -0.03;

bool pre_draw(igl::opengl::glfw::Viewer &viewer)
{
    if (viewer.core().is_animating)
    {
        RotationList anim_pose(pose.size());
        for (int e = 0; e < pose.size(); e++)
        {
            anim_pose[e] = pose[e].slerp(anim_t, Quaterniond::Identity());
        }
        // Propagate relative rotations via FK to retrieve absolute transformations
        RotationList vQ;
        vector<Vector3d> vT;
        foward_kinematics(C, BE, P, anim_pose, vQ, vT);
        const int dim = C.cols();
        MatrixXd T(BE.rows() * (dim + 1), dim);
        for (int e = 0; e < BE.rows(); e++)
        {
            Affine3d a = Affine3d::Identity();
            a.translate(vT[e]);
            a.rotate(vQ[e]);
            T.block(e * (dim + 1), 0, dim + 1, dim) = a.matrix().transpose().(0, 0, dim + 1, dim);
        }
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
    igl::readMESH("./data/hand.mesh", V, T, F);
    U = V;
    cout << "# vertices=" << V.rows() << " #faces=" << F.rows() << " #tets=" << T.rows() << endl;

    igl::readTGF("./data/hand.tgf", C, BE);
    directed_edge_parents(BE, P); // retrieve parents for each source vertex in Bone Edges
    cout << "# control vertices=" << C.rows() << " # bone edges=" << BE.rows() << endl;

    igl::readDMAT("./data/hand-pose.dmat", Q);
    column_to_quats(Q, pose);

    boundary_conditions(V, T, C, VectorXi(), BE, MatrixXi(), b, bc);
    cout << "# boundary vertices=" << b.rows() << " boundary weights=" << bc.rows() << ", " << bc.cols() << endl;

    // Compute bounded biharmonic weights
    //bbw(V, T, b, bc, W);
    // compute BBW weights matrix
    igl::BBWData bbw_data;
    // only a few iterations for sake of demo
    bbw_data.active_set_params.max_iter = 8;
    bbw_data.verbosity = 2;
    if (!igl::bbw(V, T, b, bc, bbw_data, W))
    {
        return EXIT_FAILURE;
    }

    normalize_row_sums(W, W);
    lbs_matrix(V, W, M);

    igl::opengl::glfw::Viewer viewer;
    const RowVector3d red(1, 0, 0);
    viewer.data().set_mesh(U, F);
    viewer.data().set_data(W.col(selected));
    viewer.data().set_edges(C, BE, red);
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