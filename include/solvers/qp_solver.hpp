#ifndef QP_SOLVER_H
#define QP_SOLVER_H

#include <Eigen/Dense>
#ifdef QP_SOLVER_USE_SPARSE
#include <Eigen/Sparse>
#endif
#include <cmath>
#include <limits>

#ifndef SOLVER_ASSERT
#define SOLVER_ASSERT(x) eigen_assert(x)
#endif

namespace qp_solver {

#ifdef QP_SOLVER_USE_SPARSE
template <int _n, int _m, typename _Scalar = double>
struct QP {
    using Scalar = _Scalar;
    enum {
        n=_n,
        m=_m
    };
    Eigen::SparseMatrix<Scalar> P;
    Eigen::Matrix<int, n, 1> P_col_nnz;
    Eigen::Matrix<Scalar, n, 1> q;
    Eigen::SparseMatrix<Scalar> A;
    Eigen::Matrix<int, n, 1> A_col_nnz;
    Eigen::Matrix<Scalar, m, 1> l, u;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
#else
template <int _n, int _m, typename _Scalar = double>
struct QP {
    using Scalar = _Scalar;
    enum {
        n=_n,
        m=_m
    };
    Eigen::Matrix<Scalar, n, n> P;
    Eigen::Matrix<Scalar, n, 1> q;
    Eigen::Matrix<Scalar, m, n> A;
    Eigen::Matrix<Scalar, m, 1> l, u;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
#endif

template <typename Scalar>
struct qp_sover_settings_t {
    Scalar rho = 1e-1;          /**< ADMM rho step, 0 < rho */
    Scalar sigma = 1e-6;        /**< ADMM sigma step, 0 < sigma, (small) */
    Scalar alpha = 1.0;         /**< ADMM overrelaxation parameter, 0 < alpha < 2,
                                     values in [1.5, 1.8] give good results (empirically) */
    Scalar eps_rel = 1e-3;      /**< Relative tolerance for termination, 0 < eps_rel */
    Scalar eps_abs = 1e-3;      /**< Absolute tolerance for termination, 0 < eps_abs */
    int max_iter = 1000;        /**< Maximal number of iteration, 0 < max_iter */
    int check_termination = 25; /**< Check termination after every Nth iteration, 0 (disabled) or 0 < check_termination */
    bool warm_start = false;    /**< Warm start solver, reuses previous x,z,y */
    bool adaptive_rho = false;  /**< Adapt rho to optimal estimate */
    Scalar adaptive_rho_tolerance = 5;  /**< Minimal for rho update factor, 1 < adaptive_rho_tolerance */
    int adaptive_rho_interval = 25; /**< change rho every Nth iteration, 0 < adaptive_rho_interval,
                                         set equal to check_termination to save computation  */
    bool verbose = false;

#ifdef QP_SOLVER_PRINTING
    void print() const
    {
        printf("ADMM settings:\n");
        printf("  sigma %.2e\n", sigma);
        printf("  rho %.2e\n", rho);
        printf("  alpha %.2f\n", alpha);
        printf("  eps_rel %.1e\n", eps_rel);
        printf("  eps_abs %.1e\n", eps_abs);
        printf("  max_iter %d\n", max_iter);
        printf("  adaptive_rho %d\n", adaptive_rho);
        printf("  warm_start %d\n", warm_start);
    }
#endif
};

typedef enum {
    SOLVED,
    MAX_ITER_EXCEEDED,
    UNSOLVED,
    UNINITIALIZED
} status_t;

template <typename Scalar>
struct qp_solver_info_t {
    status_t status = UNINITIALIZED; /**< Solver status */
    int iter = 0;               /**< Number of iterations */
    int rho_updates = 0;        /**< Number of rho updates (factorizations) */
    Scalar rho_estimate = 0;    /**< Last rho estimate */
    Scalar res_prim = 0;        /**< Primal residual */
    Scalar res_dual = 0;        /**< Dual residual */

#ifdef QP_SOLVER_PRINTING
    void print() const
    {
        printf("ADMM info:\n");
        printf("  status ");
        switch (status) {
        case SOLVED:
            printf("SOLVED\n");
            break;
        case MAX_ITER_EXCEEDED:
            printf("MAX_ITER_EXCEEDED\n");
            break;
        case UNSOLVED:
            printf("UNSOLVED\n");
            break;
        default:
            printf("UNINITIALIZED\n");
        };
        printf("  iter %d\n", iter);
        printf("  rho_updates %d\n", rho_updates);
        printf("  rho_estimate %f\n", rho_estimate);
        printf("  res_prim %f\n", res_prim);
        printf("  res_dual %f\n", res_dual);
    }
#endif
};

/**
 *  minimize        0.5 x' P x + q' x
 *  subject to      l <= A x <= u
 *
 *  with:
 *    x element of R^n
 *    Ax element of R^m
 */
template <typename QPType,
#ifdef QP_SOLVER_USE_SPARSE
          template <typename, int, typename... Args> class LinearSolver = Eigen::SimplicialLDLT,
#else
          template <typename, int, typename... Args> class LinearSolver = Eigen::LDLT,
#endif
          int LinearSolver_UpLo = Eigen::Lower>
class QPSolver {
public:
    enum {
        n=QPType::n,
        m=QPType::m
    };

    using qp_t = QPType;
    using Scalar = typename QPType::Scalar;
    using var_t = Eigen::Matrix<Scalar, n, 1>;
    using constraint_t = Eigen::Matrix<Scalar, m, 1>;
    using dual_t = Eigen::Matrix<Scalar, m, 1>;
    using kkt_vec_t = Eigen::Matrix<Scalar, n + m, 1>;
#ifdef QP_SOLVER_USE_SPARSE
    using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
    using kkt_mat_t = SpMat;
#else
    using kkt_mat_t = Eigen::Matrix<Scalar, n + m, n + m>;
#endif
    using settings_t = qp_sover_settings_t<Scalar>;
    using info_t = qp_solver_info_t<Scalar>;
    using linear_solver_t = LinearSolver<kkt_mat_t, LinearSolver_UpLo>;

    static constexpr Scalar RHO_MIN = 1e-6;
    static constexpr Scalar RHO_MAX = 1e+6;
    static constexpr Scalar RHO_TOL = 1e-4;
    static constexpr Scalar RHO_EQ_FACTOR = 1e+3;
    static constexpr Scalar LOOSE_BOUNDS_THRESH = 1e+16;
    static constexpr Scalar DIV_BY_ZERO_REGUL = std::numeric_limits<Scalar>::epsilon();

    // Solver state variables
    int iter;
    var_t x;
    constraint_t z;
    dual_t y;
    var_t x_tilde;
    constraint_t z_tilde;
    constraint_t z_prev;
    dual_t rho_vec;
    dual_t rho_inv_vec;
    Scalar rho;

    // State
    Scalar res_prim;
    Scalar res_dual;
    Scalar _max_Ax_z_norm;
    Scalar _max_Px_ATy_q_norm;

    enum {
        INEQUALITY_CONSTRAINT,
        EQUALITY_CONSTRAINT,
        LOOSE_BOUNDS
    } constr_type[m]; /**< constraint type classification */

    settings_t _settings;
    info_t _info;

    kkt_mat_t kkt_mat;
    linear_solver_t linear_solver;

    // enforce 16 byte alignment https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    QPSolver() { }

    void setup(const qp_t &qp)
    {
        x.setZero();
        z.setZero();
        y.setZero();

        // Set QP constraint type
        constr_type_init(qp);

        // initialize step size (rho) vector
        rho_vec_update(_settings.rho);

        // construct KKT system and compute decomposition
        construct_KKT_mat(qp);
        compute_KKT();

        _info.status = UNSOLVED;
    }

    void update_qp(const qp_t &qp)
    {
        // Set QP constraint type
        constr_type_init(qp);

        // initialize step size (rho) vector
        rho_vec_update(_settings.rho);

        // update KKT system and do factorization
        update_KKT_mat(qp);
        factorize_KKT();

        _info.status = UNSOLVED;
    }

    void solve(const qp_t &qp)
    {
        kkt_vec_t rhs, x_tilde_nu;
        bool check_termination = false;

        if (_info.status == UNINITIALIZED) {
            SOLVER_ASSERT(_info.status == UNINITIALIZED);
            return;
        }

#ifdef QP_SOLVER_PRINTING
        if (_settings.verbose) {
            _settings.print();
        }
#endif
        if (!_settings.warm_start) {
            x.setZero();
            z.setZero();
            y.setZero();
        }

        for (iter = 1; iter <= _settings.max_iter; iter++) {
            z_prev = z;

            // update x_tilde z_tilde
            form_KKT_rhs(qp, rhs);
            x_tilde_nu = linear_solver.solve(rhs);

            x_tilde = x_tilde_nu.template head<n>();
            z_tilde = z_prev + rho_inv_vec.cwiseProduct(x_tilde_nu.template tail<m>() - y);

            // update x
            x = _settings.alpha * x_tilde + (1 - _settings.alpha) * x;

            // update z
            z = _settings.alpha * z_tilde + (1 - _settings.alpha) * z_prev + rho_inv_vec.cwiseProduct(y);
            box_projection(z, qp.l, qp.u); // euclidean projection

            // update y
            y = y + rho_vec.cwiseProduct(_settings.alpha * z_tilde + (1 - _settings.alpha) * z_prev - z);

            if (_settings.check_termination != 0 && iter % _settings.check_termination == 0) {
                check_termination = true;
            } else {
                check_termination = false;
            }

            if (check_termination) {
                update_state(qp);

#ifdef QP_SOLVER_PRINTING
                if (_settings.verbose) {
                    print_status(qp);
                }
#endif
                if (termination_criteria(qp)) {
                    _info.status = SOLVED;
                    break;
                }
            }

            if (_settings.adaptive_rho && iter % _settings.adaptive_rho_interval == 0) {
                if (!check_termination) {
                    // state was not yet updated
                    update_state(qp);
                }
                Scalar new_rho = rho_estimate(rho, qp);
                new_rho = fmax(RHO_MIN, fmin(new_rho, RHO_MAX));
                _info.rho_estimate = new_rho;

                if (new_rho < rho / _settings.adaptive_rho_tolerance ||
                    new_rho > rho * _settings.adaptive_rho_tolerance) {
                    rho_vec_update(new_rho);
                    update_KKT_rho();
                    /* Note: KKT Sparsity pattern unchanged by rho update. Only factorize. */
                    factorize_KKT();
                }
            }
        }

        if (iter > _settings.max_iter) {
            _info.status = MAX_ITER_EXCEEDED;
        }
        _info.iter = iter;

#ifdef QP_SOLVER_PRINTING
        if (_settings.verbose) {
            _info.print();
        }
#endif
    }

    inline const var_t& primal_solution() const { return x; }
    inline var_t& primal_solution() { return x; }

    inline const dual_t& dual_solution() const { return y; }
    inline dual_t& dual_solution() { return y; }

    inline const settings_t& settings() const { return _settings; }
    inline settings_t& settings() { return _settings; }

    inline const info_t& info() const { return _info; }
    inline info_t& info() { return _info; }

private:
    /* Construct the KKT matrix of the form
     *
     * [[ P + sigma*I,        A' ],
     *  [ A,           -1/rho.*I ]]
     *
     * If LinearSolver_UpLo parameter is Eigen::Lower, then only the lower
     * triangular part is constructed to optimize memory.
     *
     * Note: For Eigen::ConjugateGradient it is advised to set Upper|Lower for
     *       best performance.
     */
    void construct_KKT_mat(const qp_t &qp)
    {
        static_assert(LinearSolver_UpLo == Eigen::Lower ||
                      LinearSolver_UpLo == (Eigen::Upper|Eigen::Lower),
                      "LinearSolver_UpLo must be Lower or Upper|Lower");

#ifdef QP_SOLVER_USE_SPARSE
        kkt_mat.resize(n+m,n+m); // sets all elements to 0
        Eigen::Matrix<int, n+m, 1> nnz_col;

        nnz_col.setConstant(1);
        nnz_col.template head<n>() = qp.P_col_nnz + qp.A_col_nnz;
        if (LinearSolver_UpLo == (Eigen::Upper|Eigen::Lower)) {
            // We don't know nnz of A rows, assume worst case
            nnz_col.template tail<m>().array() += n;
        }
        kkt_mat.reserve(nnz_col);

        // top left:  P + sigma*I
        sparse_insert_at(kkt_mat, 0, 0, qp.P);
        for (int i = 0; i < n; i++) {
            kkt_mat.coeffRef(i,i) += _settings.sigma;
        }

        // bottom left:  A
        sparse_insert_at(kkt_mat, n, 0, qp.A);

        // top right:  A'
        if (LinearSolver_UpLo == (Eigen::Upper|Eigen::Lower)) {
            sparse_insert_at(kkt_mat, 0, n, qp.A.transpose());
        }

        // bottom right:  -1/rho.*I
        for (int i = 0; i < m; i++) {
            kkt_mat.insert(n+i, n+i) = -rho_inv_vec(i);
        }

        kkt_mat.makeCompressed();
#else
        kkt_mat.template topLeftCorner<n, n>() = qp.P + _settings.sigma * qp.P.Identity();
        if (LinearSolver_UpLo == (Eigen::Upper|Eigen::Lower)) {
            kkt_mat.template topRightCorner<n, m>() = qp.A.transpose();
        }
        kkt_mat.template bottomLeftCorner<m, n>() = qp.A;
        kkt_mat.template bottomRightCorner<m, m>() = -1.0 * rho_inv_vec.asDiagonal();
#endif
    }

    /** KKT matrix value update, assumes same sparsity pattern */
    void update_KKT_mat(const qp_t &qp)
    {
#ifdef QP_SOLVER_USE_SPARSE
        // construct_KKT_mat(qp);
        // TODO: more efficient update?
        for (int k = 0; k < kkt_mat.outerSize(); ++k) {
            for (typename SpMat::InnerIterator it(kkt_mat, k); it; ++it) {
                int row, col;
                row = it.row();
                col = it.col();
                if (row < n) {
                    if (col < n) {
                        // top left:  P + sigma*I
                        it.valueRef() = qp.P.coeff(row, col) + _settings.sigma;
                    } else {
                        // top right:  A'
                        it.valueRef() = qp.A.coeff(col, row-n);
                    }
                } else {
                    if (col < n) {
                        // bottom left:  A
                        it.valueRef() = qp.A.coeff(row-n, col);
                    } else {
                        // bottom right:  -1/rho.*I
                        it.valueRef() = -rho_inv_vec(row-n);
                    }
                }
            }
        }
#else
        construct_KKT_mat(qp);
#endif
    }

    void update_KKT_rho()
    {
#ifdef QP_SOLVER_USE_SPARSE
        // Note: optimize by writing kkt_mat.valuePtr(). needs to be compressed and Eigen::Lower.
        for (int i = 0; i < m; i++) {
            kkt_mat.coeffRef(n+i, n+i) = -rho_inv_vec(i);
        }
#else
        kkt_mat.template bottomRightCorner<m, m>() = -1.0 * rho_inv_vec.asDiagonal();
#endif
    }

    void factorize_KKT()
    {
#ifdef QP_SOLVER_USE_SPARSE
        linear_solver.factorize(kkt_mat);
#else
        linear_solver.compute(kkt_mat);
#endif
        SOLVER_ASSERT(linear_solver.info() == Eigen::Success);
    }

    void compute_KKT()
    {
        linear_solver.compute(kkt_mat);
        SOLVER_ASSERT(linear_solver.info() == Eigen::Success);
    }


#ifdef QP_SOLVER_USE_SPARSE
    void sparse_insert_at(SpMat &dst, int row, int col, const SpMat &src) const
    {
        for (int k = 0; k < src.outerSize(); ++k) {
            for (typename SpMat::InnerIterator it(src, k); it; ++it) {
                dst.insert(row + it.row(), col + it.col()) = it.value();
            }
        }
    }
#endif

    void form_KKT_rhs(const qp_t &qp, kkt_vec_t& rhs)
    {
        rhs.template head<n>() = _settings.sigma * x - qp.q;
        rhs.template tail<m>() = z - rho_inv_vec.cwiseProduct(y);
    }

    void box_projection(constraint_t& z, const constraint_t& l, const constraint_t& u)
    {
        z = z.cwiseMax(l).cwiseMin(u);
    }

    void constr_type_init(const qp_t &qp)
    {
        for (int i = 0; i < qp.l.RowsAtCompileTime; i++) {
            if (qp.l[i] < -LOOSE_BOUNDS_THRESH && qp.u[i] > LOOSE_BOUNDS_THRESH) {
                constr_type[i] = LOOSE_BOUNDS;
            } else if (qp.u[i] - qp.l[i] < RHO_TOL) {
                constr_type[i] = EQUALITY_CONSTRAINT;
            } else {
                constr_type[i] = INEQUALITY_CONSTRAINT;
            }
        }
    }

    void rho_vec_update(Scalar rho0)
    {
        for (int i = 0; i < rho_vec.RowsAtCompileTime; i++) {
            switch (constr_type[i]) {
            case LOOSE_BOUNDS:
                rho_vec[i] = RHO_MIN;
                break;
            case EQUALITY_CONSTRAINT:
                rho_vec[i] = RHO_EQ_FACTOR*rho0;
                break;
            case INEQUALITY_CONSTRAINT: /* fall through */
            default:
                rho_vec[i] = rho0;
            };
        }
        rho_inv_vec = rho_vec.cwiseInverse();
        rho = rho0;
        _info.rho_updates += 1;
    }

    void update_state(const qp_t& qp)
    {
        Scalar norm_Ax, norm_z;
        norm_Ax = (qp.A*x).template lpNorm<Eigen::Infinity>();
        norm_z = z.template lpNorm<Eigen::Infinity>();
        _max_Ax_z_norm = fmax(norm_Ax, norm_z);

        Scalar norm_Px, norm_ATy, norm_q;
        norm_Px = (qp.P*x).template lpNorm<Eigen::Infinity>();
        norm_ATy = (qp.A.transpose()*y).template lpNorm<Eigen::Infinity>();
        norm_q = qp.q.template lpNorm<Eigen::Infinity>();
        _max_Px_ATy_q_norm = fmax(norm_Px, fmax(norm_ATy, norm_q));

        _info.res_prim = residual_prim(qp);
        _info.res_dual = residual_dual(qp);
    }

    Scalar rho_estimate(const Scalar rho0, const qp_t &qp) const
    {
        Scalar rp_norm, rd_norm;
        rp_norm = _info.res_prim / (_max_Ax_z_norm + DIV_BY_ZERO_REGUL);
        rd_norm = _info.res_dual / (_max_Px_ATy_q_norm + DIV_BY_ZERO_REGUL);

        Scalar rho_new = rho0 * sqrt(rp_norm/(rd_norm + DIV_BY_ZERO_REGUL));
        return rho_new;
    }

    Scalar eps_prim(const qp_t &qp) const
    {
        return _settings.eps_abs + _settings.eps_rel * _max_Ax_z_norm;
    }

    Scalar eps_dual(const qp_t &qp) const
    {
        return _settings.eps_abs + _settings.eps_rel * _max_Px_ATy_q_norm ;
    }

    Scalar residual_prim(const qp_t &qp) const
    {
        return (qp.A*x - z).template lpNorm<Eigen::Infinity>();
    }

    Scalar residual_dual(const qp_t &qp) const
    {
        return (qp.P*x + qp.q + qp.A.transpose()*y).template lpNorm<Eigen::Infinity>();
    }

    bool termination_criteria(const qp_t &qp)
    {
        // check residual norms to detect optimality
        if (_info.res_prim <= eps_prim(qp) && _info.res_dual <= eps_dual(qp)) {
            return true;
        }

        return false;
    }

#ifdef QP_SOLVER_PRINTING
    void print_status(const qp_t &qp) const
    {
        Scalar obj = 0.5 * x.dot(qp.P*x) + qp.q.dot(x);

        if (iter == _settings.check_termination) {
            printf("iter   obj       rp        rd\n");
        }
        printf("%4d  %.2e  %.2e  %.2e\n", iter, obj, _info.res_prim, _info.res_dual);
    }
#endif
};

} // namespace qp_solver

#endif // QP_SOLVER_H
