#include <iostream>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOFF.h>
#include <igl/readDMAT.h>
#include <igl/speye.h>
#include <igl/repdiag.h>
#include <igl/slice_into.h>
#include <igl/covariance_scatter_matrix.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/arap_rhs.h>
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
    SparseMatrix<double> K, M; // rhs pre-multiplier #V*3 * #V diffirential coordinates, mass matrix
    SparseMatrix<double> CSM;  // covariance matrix S=VV'
                               // to find best rotation by S=U*sigma*W' Ri=UW'
    igl::min_quad_with_fixed_data<double> solver_data;
    VectorXi b;
    int dim;

    int max_iter;
    double solution_diff_threshold;
    double inactive_threshold;

    ARAPData() : n(0),
                 energy(ARAP_ENERGY_TYPE_SPOKES_AND_RIMS),
                 max_iter(10),
                 solution_diff_threshold(1.0e-14),
                 inactive_threshold(1.0e-14),
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
        // cout << "rotation" << r << endl;
        // cout << si << endl;

        Matrix3d ri;
        if (single_precision)
        {
            // Eigen::JacobiSVD<Eigen::Matrix3d> sol(si, Eigen::ComputeFullU | Eigen::ComputeFullV);
            // ri = sol.matrixV() * sol.matrixU().transpose();
            igl::polar_svd3x3(si, ri);
        }
        assert(ri.determinant() >= 0);
        R.block(0, r * dim, dim, dim) = ri.block(0, 0, dim, dim).transpose();
    }
}

void arap_solve(const MatrixXd &bc,
                ARAPData &data,
                MatrixXd &U)
{
    int n = data.n;
    MatrixXd U_prev = U;
    int iter = 0;
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
    // }
}

enum SolverStatus
{
    // Good
    SOLVER_STATUS_CONVERGED = 0,
    // OK
    SOLVER_STATUS_MAX_ITER = 1,
    // Bad
    SOLVER_STATUS_ERROR = 2,
    NUM_SOLVER_STATUSES = 3,
};

SolverStatus arap_layer(const MatrixXd &V,
                        const MatrixXi &F,
                        const VectorXi &b,
                        const MatrixXd &bc,
                        const SparseMatrix<double> &Aeq,
                        const VectorXd &Beq,
                        const SparseMatrix<double> &Aieq,
                        const VectorXd &Bieq,
                        const VectorXd &p_lx,
                        const VectorXd &p_ux,
                        ARAPData &data,
                        MatrixXd &U)
{
    using namespace std;
    using namespace Eigen;

    int n = V.rows();
    int dim = V.cols();
    assert(dim == 3 && dim == bc.cols());

    SolverStatus ret = SOLVER_STATUS_ERROR;
    data.dim = V.cols();
    //----------------------------------Precompute Laplace & Covariance Scatter Matrix-----------------------
    SparseMatrix<double> L;
    igl::cotmatrix(V, F, L);
    igl::ARAPEnergyType eff_energy = igl::ARAP_ENERGY_TYPE_SPOKES_AND_RIMS;
    // igl::ARAPEnergyType eff_energy = igl::ARAP_ENERGY_TYPE_SPOKES;

    // covariance scatter matrix CSM: dim*#V x dim*#V
    // fit rotations
    igl::covariance_scatter_matrix(V, F,
                                   eff_energy,
                                   data.CSM);

    SparseMatrix<double> G_sum, G_sum_dim;
    igl::speye(n, G_sum);
    igl::repdiag(G_sum, data.dim, G_sum_dim);
    data.CSM = (G_sum_dim * data.CSM).eval();
    // rhs K
    igl::arap_rhs(V, F, data.dim, eff_energy, data.K);
    SparseMatrix<double> Q = (-L).eval();

    //------Active Set with Quadratic Programing to solve system with inequality constraints-----------------------
    VectorXd lx, ux;
    if (p_lx.size() == 0)
    {
        lx = VectorXd::Constant(n, 1, -std::numeric_limits<double>::max());
    }
    else
    {
        lx = p_lx;
    }

    if (p_ux.size() == 0)
    {
        ux = VectorXd::Constant(n, 1, std::numeric_limits<double>::max());
    }
    else
    {
        ux = p_ux;
    }
    const int nk = bc.rows();

    VectorXi as_lx = VectorXi::Constant(n, 1, false);
    VectorXi as_ux = VectorXi::Constant(n, 1, false);
    VectorXi as_ieq = VectorXi::Constant(Aieq.rows(), 1, false);

    int iter = 0;
    MatrixXd U_prev = U;
    VectorXd Z = U.col(2);
    VectorXd old_Z = VectorXd::Constant(n, 1, std::numeric_limits<double>::max());
    while (true)
    {
        int new_as_lx = 0;
        int new_as_ux = 0;
        int new_as_ieq = 0;
        // 1. check which unknowns violate the constraints and should be activated in this iteration
        if (Z.size() > 0)
        {
            for (int i = 0; i < n; i++)
            {
                if (Z(i) < lx(i))
                {
                    new_as_lx += (as_lx(i) ? 0 : 1);
                    as_lx(i) = true;
                    break;
                }

                if (Z(i) > ux(i))
                {
                    new_as_ux += (as_ux(i) ? 0 : 1);
                    as_ux(i) = true;
                    break;
                }
            }

            if (Aieq.rows() > 0)
            {
                VectorXd AieqZ;
                AieqZ = Aieq * Z;
                for (int a = 0; a < Aieq.rows(); a++)
                {
                    if (AieqZ(a) > Bieq(a))
                    {
                        new_as_ieq += (as_ieq(a) ? 0 : 1);
                        as_ieq(a) = true;
                    }
                }
            }

            const double diff = (Z - old_Z).squaredNorm();
            printf("Iter%d: energy=%.6f\n", iter, diff);
            if (diff < data.solution_diff_threshold)
            {
                ret = SOLVER_STATUS_CONVERGED;
                printf("Iter%d: ................Converged\n", iter);
                break;
            }
            old_Z = Z;
            // cout << Z << endl;
        }

        const int as_lx_count = std::count(as_lx.data(), as_lx.data() + n, true);
        const int as_ux_count = std::count(as_ux.data(), as_ux.data() + n, true);
        const int as_ieq_count = std::count(as_ieq.data(), as_ieq.data() + as_ieq.size(), true);
        printf("Iter%d: as_lx=%d as_ux=%d, as_ieq=%d\n", iter, as_lx_count, as_ux_count, as_ieq_count);

        // 2. prepare fixed value, lower and upper bound
        VectorXi known_i;
        known_i.resize(nk + as_lx_count + as_ux_count, 1);
        MatrixXd Y_i;
        Y_i.resize(nk + as_lx_count + as_ux_count, dim);
        {
            known_i.block(0, 0, b.rows(), b.cols()) = b;
            Y_i.block(0, 0, bc.rows(), bc.cols()) = bc;
            int k = nk;
            for (int z = 0; z < n; z++)
            {
                if (as_lx(z))
                {
                    known_i(k) = z;
                    for (int d = 0; d < dim; d++)
                    {
                        Y_i(k, d) = lx(z);
                    }
                    k++;
                }
            }

            for (int z = 0; z < n; z++)
            {
                if (as_ux(z))
                {
                    known_i(k) = z;
                    for (int d = 0; d < dim; d++)
                    {
                        Y_i(k, d) = ux(z);
                    }
                    k++;
                }
            }
            assert(k == Y_i.rows());
            assert(k == known_i.rows());
        }

        // 3. Gather active constraints
        //    3.1 filter active constraints
        VectorXi as_ieq_list(as_ieq_count, 1);
        VectorXd Beq_i;
        Beq_i.resize(Beq.rows() + as_ieq_count, 1);
        Beq_i.head(Beq.rows()) = Beq;
        {
            int k = 0;
            for (int a = 0; a < as_ieq.size(); a++)
            {
                if (as_ieq(a))
                {
                    assert(k < as_ieq_list.size());
                    as_ieq_list(k) = a;
                    Beq_i(Beq.rows() + k, 0) = Bieq(k, 0);
                    k++;
                }
            }
            assert(k == as_ieq_count);
        }
        //    3.2 extract active constraint rows and append them to equality equation
        SparseMatrix<double> Aeq_i, Aieq_i;
        igl::slice(Aieq, as_ieq_list, 1, Aieq_i);
        igl::cat(1, Aeq, Aieq_i, Aeq_i);
        // 4. solve unknowns
        VectorXd sol;
        if (known_i.size() == Q.rows()) // Everything is fixed
        {
            Z.resize(Q.rows(), 1);
            igl::slice_into(Y_i.col(2), known_i, 1, Z);
            sol.resize(0, Y_i.cols());
            assert(Aeq_i.rows() == 0 && "All fixed but linearly constrained");
        }
        else
        {
            // ---------4.1 Local Step, fit best rotation matrix
            for (int bi = 0; bi < bc.rows(); bi++)
            {
                U.row(b(bi)) = bc.row(bi);
            }
            const auto &Udim = U.replicate(data.dim, 1);
            MatrixXd S = data.CSM * Udim;
            S /= S.array().abs().maxCoeff();
            int Rdim = data.dim;
            MatrixXd R(Rdim, data.CSM.rows()); // 3 x 3n
            fit_rotations(S, true, R);
            MatrixXd eff_R = R;
            int num_rots = data.K.cols() / Rdim / Rdim; // num_vertex

            VectorXd Rcol;
            igl::columnize(eff_R, num_rots, 2, Rcol);
            VectorXd Bcol = -data.K * Rcol;
            // ---------4.2 Solve, z-dim should conform to the inequality constraint
            for (int c = 0; c < data.dim; c++)
            {
                cout << "Solve for column " << c << endl;
                // 4.1 factoring/precomputation, provided LHS matrix and boundary indices
                SparseMatrix<double> Aeq_c;
                if (c == 2)
                {
                    Aeq_c = Aeq_i;
                }
                if (!igl::min_quad_with_fixed_precompute(Q, b, Aeq_c, true, data.solver_data))
                {
                    cerr << "Error: min_quad_with_fixed precomputation failed." << endl;
                    if (iter > 0 && Aeq_i.rows() > Aeq.rows())
                    {
                        cerr << "  *Are you sure rows of [Aeq;Aieq] are linearly independent?*" << endl;
                    }
                    ret = SOLVER_STATUS_ERROR;
                    break;
                }
                // ---------4.2.2 Global Step, find optimal vertex positions
                VectorXd Uc, Bc, bcc, Beq_c;
                if (c == 2) // z-dim
                {
                    Beq_c = Beq_i;
                }
                Bc = Bcol.block(c * n, 0, n, 1);
                if (bc.size() > 0)
                {
                    bcc = bc.col(c);
                }
                if (!igl::min_quad_with_fixed_solve(data.solver_data, Bc, bcc, Beq_c, Uc, sol))
                {
                    cerr << "Error: min_quad_with_fixed solve failed." << endl;
                    ret = SOLVER_STATUS_ERROR;
                    break;
                }
                U.col(c) = Uc;
                assert(data.solver_data.Auu_sym);

                // ---------4.2.3 active set post-process: inactivate variables already conforming to constraints
                if (c == 2)
                {
                    Z = Uc;
                    SparseMatrix<double> Qk;
                    igl::slice(Q, known_i, 1, Qk);
                    VectorXd Bk;
                    igl::slice(Bc, known_i, Bk);
                    MatrixXd Lambda_known_i = -(0.5 * Qk * Z + 0.5 * Bk);
                    // reverse lambda values for lx
                    Lambda_known_i.block(nk, 0, as_lx_count, 1) =
                        (-1 * Lambda_known_i.block(nk, 0, as_lx_count, 1)).eval();
                    // Extract Lagrange multipliers for Aieq_i (always at back of sol)
                    VectorXd Lambda_Aieq_i(Aieq_i.rows(), 1);
                    for (int l = 0; l < Aieq_i.rows(); l++)
                    {
                        Lambda_Aieq_i(Aieq_i.rows() - 1 - l) = sol(sol.rows() - 1 - l);
                    }
                    // remove from active set
                    for (int l = 0; l < as_lx_count; l++)
                    {
                        if (Lambda_known_i(nk + l) < data.inactive_threshold)
                        {
                            as_lx(known_i(nk + l)) = false;
                        }
                    }
                    for (int u = 0; u < as_ux_count; u++)
                    {
                        if (Lambda_known_i(nk + as_lx_count + u) < data.inactive_threshold)
                        {
                            as_ux(known_i(nk + as_lx_count + u)) = false;
                        }
                    }
                    for (int a = 0; a < as_ieq_count; a++)
                    {
                        if (Lambda_Aieq_i(a) < data.inactive_threshold)
                        {
                            as_ieq(int(as_ieq_list(a))) = false;
                        }
                    }
                }
            }
        }

        iter++;
        if (data.max_iter > 0 && iter >= data.max_iter)
        {
            ret = SOLVER_STATUS_MAX_ITER;
            cout << "Solve without converged, exceed maximum iteration\n"
                 << endl;
            break;
        }
    }
}

typedef std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>>
    RotationList;

const Eigen::RowVector3d sea_green(70. / 255., 252. / 255., 167. / 255.);

MatrixXd V, U;
MatrixXi F;
VectorXi b;
VectorXd lx, ux;
RowVector3d mid;
double anim_t = 0.0;
double anim_t_dir = 0.03;
ARAPData arap_data;
SparseMatrix<double> Aeq, Aieq;
VectorXd Beq, Bieq;

void test()
{
    make_plane(V, F, 4, 4);
    U = V;
    // cout << "V: " << V.rows() << " F: " << F.rows() << endl;
    // cout << "V" << endl;
    // cout << V << endl;
    // cout << "F" << endl;
    // cout << F << endl;

    b.resize(1);
    MatrixXd bc(1, 3);
    b << 12;
    bc << 0, 0, 1;

    Aieq.resize(1, V.rows());
    Aieq.reserve(0.1 * V.rows());
    Aieq.insert(0, 0) = -1.0;
    Aieq.insert(0, 12) = 1.0;
    Bieq.resize(1, 1);
    Bieq(0, 0) = 0.0;

    // arap_precomp(V, F, 3, b, arap_data);
    // arap_solve(bc, arap_data, U);
    arap_layer(V, F, b, bc, Aeq, Beq, Aieq, Bieq, lx, ux, arap_data, U);
}

int main()
{

    test();
    // Plot the mesh with pseudocolors
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(U, F);
    viewer.launch();

    return 0;
}