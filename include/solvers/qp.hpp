#pragma once

#ifdef QP_SOLVER_SPARSE
#include <Eigen/Sparse>
#endif

#include <Eigen/Dense>
#include <limits>
#include <vector>

#define QP_SOLVER_PRINTING

namespace qp_solver {

/** Quadratic Problem
 *  minimize        0.5 x' P x + q' x
 *  subject to      l <= A x <= u
 */
template <typename Scalar = double>
struct QuadraticProblem {
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
#ifdef QP_SOLVER_SPARSE
    using Matrix = Eigen::SparseMatrix<Scalar>;
    Eigen::Matrix<int, Eigen::Dynamic, 1> P_col_nnz;
    Eigen::Matrix<int, Eigen::Dynamic, 1> A_col_nnz;
#else
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
#endif
    const Matrix *P;
    const Vector *q;
    const Matrix *A;
    const Vector *l;
    const Vector *u;
};

template <typename Scalar>
struct QPSolverSettings {
    Scalar rho = 1e-1;          /**< ADMM rho step, 0 < rho */
    Scalar sigma = 1e-6;        /**< ADMM sigma step, 0 < sigma, (small) */
    Scalar alpha = 1.0;         /**< ADMM overrelaxation parameter, 0 < alpha < 2,
                                     values in [1.5, 1.8] give good results (empirically) */
    Scalar eps_rel = 1e-3;      /**< Relative tolerance for termination, 0 < eps_rel */
    Scalar eps_abs = 1e-3;      /**< Absolute tolerance for termination, 0 < eps_abs */
    int max_iter = 1000;        /**< Maximal number of iteration, 0 < max_iter */
    int check_termination = 25; /**< Check termination after every Nth iteration, 0 (disabled) or 0
                                   < check_termination */
    bool warm_start = false;    /**< Warm start solver, reuses previous x,z,y */
    bool adaptive_rho = false;  /**< Adapt rho to optimal estimate */
    Scalar adaptive_rho_tolerance =
        5; /**< Minimal for rho update factor, 1 < adaptive_rho_tolerance */
    int adaptive_rho_interval = 25; /**< change rho every Nth iteration, 0 < adaptive_rho_interval,
                                         set equal to check_termination to save computation  */
    bool verbose = false;

#ifdef QP_SOLVER_PRINTING
    void print() const {
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

typedef enum { SOLVED, MAX_ITER_EXCEEDED, UNSOLVED, NUMERICAL_ISSUES, UNINITIALIZED } QPSolverStatus;

template <typename Scalar>
struct QPSolverInfo {
    QPSolverStatus status = UNINITIALIZED; /**< Solver status */
    int iter = 0;                          /**< Number of iterations */
    int rho_updates = 0;                   /**< Number of rho updates (factorizations) */
    Scalar rho_estimate = 0;               /**< Last rho estimate */
    Scalar res_prim = 0;                   /**< Primal residual */
    Scalar res_dual = 0;                   /**< Dual residual */

#ifdef QP_SOLVER_PRINTING
    void print() const {
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
            case NUMERICAL_ISSUES:
                printf("NUMERICAL_ISSUES\n");
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
template <typename SCALAR>
class QPSolver {
   public:
    using Scalar = SCALAR;
    using QP = QuadraticProblem<Scalar>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
#ifdef QP_SOLVER_SPARSE
    using Matrix = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
    using LinearSolver = Eigen::SimplicialLDLT<Matrix, Eigen::Lower>;
#else
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using LinearSolver = Eigen::LDLT<Matrix, Eigen::Lower>;
#endif
    using Settings = QPSolverSettings<Scalar>;
    using Info = QPSolverInfo<Scalar>;

    enum { INEQUALITY_CONSTRAINT, EQUALITY_CONSTRAINT, LOOSE_BOUNDS } ConstraintType;

    static constexpr Scalar RHO_MIN = 1e-6;
    static constexpr Scalar RHO_MAX = 1e+6;
    static constexpr Scalar RHO_TOL = 1e-4;
    static constexpr Scalar RHO_EQ_FACTOR = 1e+3;
    static constexpr Scalar LOOSE_BOUNDS_THRESH = 1e+16;
    static constexpr Scalar DIV_BY_ZERO_REGUL = std::numeric_limits<Scalar>::epsilon();

    // enforce 16 byte alignment
    // https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /** Constructor */
    QPSolver() = default;

    /** Setup solver for QP. */
    void setup(const QP &qp);

    /** Update solver for QP of same size as initial setup. */
    void update_qp(const QP &qp);

    /** Solve the QP. */
    void solve(const QP &qp);

    inline const Vector &primal_solution() const { return x; }
    inline Vector &primal_solution() { return x; }

    inline const Vector &dual_solution() const { return y; }
    inline Vector &dual_solution() { return y; }

    inline const Settings &settings() const { return settings_; }
    inline Settings &settings() { return settings_; }

    inline const Info &info() const { return info_; }
    inline Info &info() { return info_; }


    /* Public funcitions for unit testing */
    static void constr_type_init(const Vector& l, const Vector& u, Eigen::VectorXi &constr_type);

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
    void construct_KKT_mat(const QP &qp);

    /** KKT matrix value update, assumes same sparsity pattern */
    void update_KKT_mat(const QP &qp);

    void update_KKT_rho();
    bool factorize_KKT();
    bool compute_KKT();
#ifdef QP_SOLVER_SPARSE
    void sparse_insert_at(Matrix &dst, int row, int col, const Matrix &src) const
#endif
        void form_KKT_rhs(const QP &qp, Vector &rhs);

    void box_projection(Vector &z, const Vector &l, const Vector &u);

    void constr_type_init(const QP &qp);
    void rho_vec_update(Scalar rho0);
    void update_state(const QP &qp);
    Scalar rho_estimate(const Scalar rho0, const QP &qp) const;
    Scalar eps_prim(const QP &qp) const;
    Scalar eps_dual(const QP &qp) const;
    Scalar residual_prim(const QP &qp) const;
    Scalar residual_dual(const QP &qp) const;

    bool termination_criteria(const QP &qp);

#ifdef QP_SOLVER_PRINTING
    void print_status(const QP &qp) const;
#endif

    size_t n;  //< number of variables
    size_t m;  //< number of constraints

    // Solver state variables
    int iter;
    Vector x;  //< primal variable, size n
    Vector z;  //< additional variable, size m
    Vector y;  //< dual variable, size m
    Vector x_tilde;
    Vector z_tilde;
    Vector z_prev;
    Vector rho_vec;
    Vector rho_inv_vec;
    Scalar rho;

    Vector rhs;
    Vector x_tilde_nu;

    // State
    Scalar res_prim;
    Scalar res_dual;
    Scalar max_Ax_z_norm_;
    Scalar max_Px_ATy_q_norm_;

    Eigen::VectorXi constr_type; /**< constraint type classification */

    Settings settings_;
    Info info_;

    Matrix kkt_mat;
    LinearSolver linear_solver;
};

extern template class QPSolver<double>;
extern template class QPSolver<float>;

}  // namespace qp_solver
