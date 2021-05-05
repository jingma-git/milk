#include <iostream>

#include <igl/readOFF.h>
#include <igl/readDMAT.h>
#include <igl/speye.h>
#include <igl/repdiag.h>
#include <igl/covariance_scatter_matrix.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/arap_rhs.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/polar_svd3x3.h>
#include <igl/columnize.h>

using namespace std;
using namespace Eigen;

void make_plane(MatrixXd &V, MatrixXi &F, int W = 1, int H = 1, double len = 1.0)
{
    int v_num = (W + 1) * (H + 1);
    V.resize(v_num, 3);
    F.resize(2 * W * H, 3);

    for (int v_idx = 0; v_idx < v_num; v_idx++)
    {
        int r = v_idx / (W + 1);
        int c = v_idx % (W + 1);
        V.row(v_idx) = RowVector3d(c * len, r * len, 0);
    }

    int f_idx = 0;
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            int v0 = i * (W + 1) + j;
            int v1 = i * (W + 1) + (j + 1);
            int v2 = (i + 1) * (W + 1) + j;
            int v3 = (i + 1) * (W + 1) + (j + 1);

            F.row(f_idx) = RowVector3i(v0, v1, v2);
            F.row(f_idx + 1) = RowVector3i(v1, v3, v2);
            f_idx += 2;
        }
    }
}

typedef SparseMatrix<double> SpMatd;

enum ARAPEnergyType
{
    ARAP_ENERGY_TYPE_SPOKES = 0,
    ARAP_ENERGY_TYPE_SPOKES_AND_RIMS = 1,
    ARAP_ENERGY_TYPE_ELEMENTS = 2,
    ARAP_ENERGY_TYPE_DEFAULT = 3,
    NUM_ARAP_ENERGY_TYPES = 4
};

struct ARAPData
{
    int n;
    ARAPEnergyType energy;
    MatrixXd f_ext, vel; // #V x dim
    double h;            // time step
    int max_iter;
    SparseMatrix<double> K, M; // rhs pre-multiplier #V*3 * #V diffirential coordinates, mass matrix
    SparseMatrix<double> CSM;  // covariance matrix S=VV'
                               // to find best rotation by S=U*sigma*W' Ri=UW'
    igl::min_quad_with_fixed_data<double> solver_data;
    VectorXi b;
    int dim;

    ARAPData() : n(0),
                 energy(ARAP_ENERGY_TYPE_SPOKES_AND_RIMS),
                 f_ext(),
                 h(1),
                 max_iter(10),
                 K(),
                 CSM(),
                 solver_data(),
                 b(),
                 dim(-1) // force this to be set by _precomputation
                 {};
};

void arap_precomp(const MatrixXd &V,
                  const MatrixXi &F,
                  const int dim,
                  const VectorXi &b,
                  ARAPData &data)
{
    int n = V.rows();
    data.n = n;
    data.b = b;
    data.dim = V.cols();

    // MatrixXd C;
    // igl::cotmatrix_entries(V, F, C);
    // cout << "C" << endl;
    // MatrixXi edges(3, 2);
    // edges << 1, 2,
    //     2, 0,
    //     0, 1;
    // for (int f = 0; f < F.rows(); f++)
    // {
    //     for (int e = 0; e < edges.rows(); e++)
    //     {
    //         cout << "v" << F(f, edges(e, 0)) << "->v" << F(f, edges(e, 1)) << ": " << C(f, e) << endl;
    //     }
    // }

    SpMatd L;
    igl::cotmatrix(V, F, L);
    // cout << "L" << endl;
    // cout << L << endl;
    igl::ARAPEnergyType eff_energy = igl::ARAP_ENERGY_TYPE_SPOKES_AND_RIMS;
    // igl::ARAPEnergyType eff_energy = igl::ARAP_ENERGY_TYPE_SPOKES;

    // covariance scatter matrix CSM: dim*#V x dim*#V
    // fit rotations
    igl::covariance_scatter_matrix(V, F,
                                   eff_energy,
                                   data.CSM);

    SpMatd G_sum, G_sum_dim;
    igl::speye(n, G_sum);
    igl::repdiag(G_sum, data.dim, G_sum_dim);
    data.CSM = (G_sum_dim * data.CSM).eval();

    // cout << "data.CSM: " << data.CSM.rows() << "," << data.CSM.cols() << endl;
    // cout << data.CSM.toDense() << endl;

    // rhs K
    igl::arap_rhs(V, F, data.dim, eff_energy, data.K);
    // cout << "K" << endl;
    // cout << data.K.rows() << ", " << data.K.cols() << endl;

    // factorize L, and move boundary vertices to right hand side
    SpMatd Q = (-L).eval();
    igl::min_quad_with_fixed_precompute(Q, b, SpMatd(), true, data.solver_data);
}

void fit_rotations(const MatrixXd &S,
                   const bool single_precision,
                   MatrixXd &R)
{
    const int dim = S.cols();
    const int nr = S.rows() / dim;
    R.resize(dim, dim * nr);
    Matrix3d si;
    for (int r = 0; r < nr; r++)
    {
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                si(i, j) = S(i * nr + r, j);
            }
        }

        Matrix3d ri;
        if (single_precision)
        {
            // Eigen::JacobiSVD<Eigen::Matrix3d> sol(si, Eigen::ComputeFullU | Eigen::ComputeFullV);
            // ri = sol.matrixV() * sol.matrixU().transpose();
            igl::polar_svd3x3(si, ri);
        }
        assert(ri.determinant() >= 0);
        R.block(0, r * dim, dim, dim) = ri.block(0, 0, dim, dim).transpose();
        cout << "rotation" << r << endl;
        cout << ri << endl;
    }
}

void arap_solve(const MatrixXd &bc,
                ARAPData &data,
                MatrixXd &U)
{
    int n = data.n;
    MatrixXd U_prev = U;
    int iter = 0;
    // cout << "U" << endl;
    // cout << U << endl;
    // while (iter < data.max_iter)
    // {
    for (int bi = 0; bi < bc.rows(); bi++)
    {
        U.row(data.b(bi)) = bc.row(bi);
    }
    const auto &Udim = U.replicate(data.dim, 1);
    // cout << "Udim" << endl;
    // cout << Udim << endl;
    MatrixXd S = data.CSM * Udim;
    S /= S.array().abs().maxCoeff();
    // cout << "S" << endl;
    // cout << S << endl;
    int Rdim = data.dim;
    MatrixXd R(Rdim, data.CSM.rows()); // 3 x 3n
    fit_rotations(S, true, R);
    MatrixXd eff_R = R;
    int num_rots = data.K.cols() / Rdim / Rdim; // num_vertex
    // cout << "num_rots: " << num_rots << endl;
    VectorXd Rcol;
    igl::columnize(eff_R, num_rots, 2, Rcol);
    VectorXd Bcol = -data.K * Rcol;
    for (int c = 0; c < data.dim; c++)
    {
        VectorXd Uc, Bc, bcc, Beq;
        Bc = Bcol.block(c * n, 0, n, 1);
        if (bc.size() > 0)
        {
            bcc = bc.col(c);
        }
        igl::min_quad_with_fixed_solve(data.solver_data, Bc, bcc, Beq, Uc);
        U.col(c) = Uc;
    }
    iter++;
    // }
}

typedef std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>>
    RotationList;

const Eigen::RowVector3d sea_green(70. / 255., 252. / 255., 167. / 255.);

MatrixXd V, U;
MatrixXi F;
VectorXi S, b;
RowVector3d mid;
double anim_t = 0.0;
double anim_t_dir = 0.03;
ARAPData arap_data;

bool pre_draw(igl::opengl::glfw::Viewer &viewer)
{
    MatrixXd bc(b.size(), V.cols());
    for (int i = 0; i < b.size(); i++)
    {
        bc.row(i) = V.row(b(i));
        switch (S(b(i)))
        {
        case 0:
        {
            const double r = mid(0) * 0.25;
            // bc(i, 0) += r * sin(0.5 * anim_t * 2. * igl::PI);
            bc(i, 1) -= r + r * cos(igl::PI + 0.5 * anim_t * 2. * igl::PI);
            break;
        }
        case 1:
        {
            const double r = mid(1) * 0.15;
            // bc(i, 1) += r + r * cos(igl::PI + 0.15 * anim_t * 2. * igl::PI);
            // bc(i, 2) -= r * sin(0.15 * anim_t * 2. * igl::PI);
            break;
        }
        case 2:
        {
            const double r = mid(1) * 0.15;
            // bc(i, 2) += r + r * cos(igl::PI + 0.35 * anim_t * 2. * igl::PI);
            // bc(i, 0) += r * sin(0.35 * anim_t * 2. * igl::PI);
            break;
        }
        default:
            break;
        }
        arap_solve(bc, arap_data, U);
        viewer.data().set_vertices(U);
        viewer.data().compute_normals();
        if (viewer.core().is_animating)
        {
            anim_t += anim_t_dir;
        }
        return false;
    }
}

void test()
{
    make_plane(V, F, 4, 4);
    U = V;
    cout << "V: " << V.rows() << " F: " << F.rows() << endl;
    cout << "V" << endl;
    cout << V << endl;
    int cent_idx = V.rows() / 2;
    cout << "...............Cent_idx=" << cent_idx << endl;
    // cout << "F" << endl;
    // cout << F << endl;

    // b.resize(2);
    // MatrixXd bc(2, 3);
    // b << 0, 12;

    // Eigen::RowVector3d trans(5, 0, 0);
    // bc.row(0) = V.row(0) + trans;
    // bc.row(1) = V.row(12) + trans;
    b.resize(1);
    MatrixXd bc(1, 3);
    b << cent_idx;

    Eigen::RowVector3d trans(5, 0, 0);
    bc.row(0) = V.row(cent_idx) + trans;

    arap_precomp(V, F, 3, b, arap_data);
    arap_solve(bc, arap_data, U);
    cout << "after solve" << endl;
    cout << U - V << endl;
}

int main()
{

    test();
    // igl::readOFF("data/decimated-knight.off", V, F);
    // U = V;
    // igl::readDMAT("data/decimated-knight-selection.dmat", S);
    // igl::colon<int>(0, V.rows() - 1, b);
    // // stable_partition: equivalent elements do not change order, descending order
    // std::ptrdiff_t dis = stable_partition(b.data(), b.data() + b.size(), [](int i) -> bool { return S(i) >= 0; }) - b.data();
    // b.conservativeResize(dis);
    // mid = 0.5 * (V.colwise().maxCoeff() + V.colwise().minCoeff());
    // cout << b.rows() << endl;
    // cout << "mid: " << mid << endl;

    // arap_data.max_iter = 100;
    // arap_precomp(V, F, V.cols(), b, arap_data);

    // // Set color based on selection
    // MatrixXd C(F.rows(), 3);
    // RowVector3d purple(80.0 / 255.0, 64.0 / 255.0, 255.0 / 255.0);
    // RowVector3d red(1.0, 0.0, 0.0);
    // RowVector3d blue(0.0, 0.0, 1.0);
    // RowVector3d gold(255.0 / 255.0, 228.0 / 255.0, 58.0 / 255.0);
    // for (int f = 0; f < F.rows(); f++)
    // {
    //     if (S(F(f, 0)) >= 2 && S(F(f, 1)) >= 2 && S(F(f, 2)) >= 2)
    //     {
    //         C.row(f) = red;
    //     }
    //     else if (S(F(f, 0)) >= 1 && S(F(f, 1)) >= 1 && S(F(f, 2)) >= 1)
    //     {
    //         C.row(f) = blue;
    //     }
    //     else if (S(F(f, 0)) >= 0 && S(F(f, 1)) >= 0 && S(F(f, 2)) >= 0)
    //     {
    //         C.row(f) = purple;
    //     }
    //     else
    //     {
    //         C.row(f) = gold;
    //     }
    // }

    // Plot the mesh with pseudocolors
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(U, F);
    // viewer.data().set_colors(C);
    // viewer.callback_pre_draw = &pre_draw;
    // viewer.callback_key_down = &key_down;
    viewer.core().is_animating = false;
    viewer.core().animation_max_fps = 30.;
    cout << "Press [space] to toggle animation" << endl;
    viewer.launch();

    return 0;
}