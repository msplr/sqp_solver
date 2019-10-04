#pragma once

#include <Eigen/Dense>
#include <limits>

namespace sqp {

template <typename Scalar>
struct sqp_settings_t {
    Scalar tau = 0.5;       /**< line search iteration decrease, 0 < tau < 1 */
    Scalar eta = 0.25;      /**< line search parameter, 0 < eta < 1 */
    Scalar rho = 0.5;       /**< line search parameter, 0 < rho < 1 */
    Scalar eps_prim = 1e-3; /**< primal step termination threshold, eps_prim > 0 */
    Scalar eps_dual = 1e-3; /**< dual step termination threshold, eps_dual > 0 */
    int max_iter = 100;
    int line_search_max_iter = 100;
    void (*iteration_callback)(void* solver) = nullptr;

    bool validate() {
        bool valid;
        valid = 0.0 < tau && tau < 1.0 && 0.0 < eta && eta < 1.0 && 0.0 < rho && rho < 1.0 &&
                eps_prim < 0.0 && eps_dual < 0.0 && max_iter > 0 && line_search_max_iter > 0;
        return valid;
    }
};

struct sqp_status_t {
    enum { SOLVED, MAX_ITER_EXCEEDED, INVALID_SETTINGS } value;
};

struct sqp_info_t {
    int iter;
    int qp_solver_iter;
    sqp_status_t status;
};

template <typename Scalar_ = double>
struct NonLinearProblem {
   public:
    using Scalar = Scalar_;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    int num_var;
    int num_ineq;
    int num_eq;

    virtual void objective(const Vector& x, Scalar& obj) = 0;
    virtual void objective_linearized(const Vector& x, Vector& grad, Scalar& obj) = 0;
    virtual void constraint(const Vector& x, Vector& b_eq, Vector& b_ineq, Vector& lb,
                            Vector& ub) = 0;
    virtual void constraint_linearized(const Vector& x, Matrix& A_eq, Vector& b_eq, Matrix& A_ineq,
                                       Vector& b_ineq, Vector& lb, Vector& ub) = 0;
};

/*
 * minimize     f(x)
 * subject to   ce(x)  = 0
 *              ci(x) <= 0
 *              l <= x <= u
 */
template <typename Scalar_>
class SQP {
   public:
    using Scalar = Scalar_;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Problem = NonLinearProblem<Scalar>;

    using settings_t = sqp_settings_t<Scalar>;

    // Constants
    static constexpr Scalar DIV_BY_ZERO_REGUL = std::numeric_limits<Scalar>::epsilon();

    // enforce 16 byte alignment
    // https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SQP();
    ~SQP() = default;

    void solve(Problem& prob, const Vector& x0, const Vector& lambda0);
    void solve(Problem& prob);

   private:
    void _solve(Problem& prob);

    bool termination_criteria(const Vector& x, Problem& prob) const;
    void solve_qp(Problem& prob, Vector& p, Vector& lambda);

    /** Line search in direction p using l1 merit function. */
    Scalar line_search(Problem& prob, const Vector& p);

    /** L1 norm of constraint violation */
    Scalar constraint_norm(const Vector& x, Problem& prob) const;

    /** L_inf norm of constraint violation */
    Scalar max_constraint_violation(const Vector& x, Problem& prob) const;

   private:
    // Solver state variables
    Vector _x;
    Vector _lambda;
    Vector _step_prev;
    Vector _grad_L;

    Vector b_eq;
    Matrix A_eq;
    Vector b_ineq;
    Matrix A_ineq;
    Vector lb_, ub_;

    Matrix P;
    Matrix A;
    Vector q;

    settings_t _settings;
    sqp_info_t _info;

    // info
    Scalar _dual_step_norm;
    Scalar _primal_step_norm;
    Scalar _cost;

   public:
    inline const Vector& primal_solution() const { return _x; }
    inline Vector& primal_solution() { return _x; }

    inline const Vector& dual_solution() const { return _lambda; }
    inline Vector& dual_solution() { return _lambda; }

    inline const settings_t& settings() const { return _settings; }
    inline settings_t& settings() { return _settings; }

    inline const sqp_info_t& info() const { return _info; }
    inline sqp_info_t& info() { return _info; }
};

extern template class SQP<double>;

}  // namespace sqp
