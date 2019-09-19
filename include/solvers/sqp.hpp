#ifndef SQP_H
#define SQP_H

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include "qp_solver.hpp"
#include "bfgs.hpp"

#ifdef SOLVER_DEBUG
#include <Eigen/Eigenvalues>
#endif

#ifndef SOLVER_ASSERT
#define SOLVER_ASSERT(x) eigen_assert(x)
#endif

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
    void (*iteration_callback)(void *solver) = nullptr;

    bool validate()
    {
        bool valid;
        valid = 0.0 < tau && tau < 1.0 &&
                0.0 < eta && eta < 1.0 &&
                0.0 < rho && rho < 1.0 &&
                eps_prim < 0.0 &&
                eps_dual < 0.0 &&
                max_iter > 0 &&
                line_search_max_iter > 0;
        return valid;
    }
};

struct sqp_status_t {
    enum {
        SOLVED,
        MAX_ITER_EXCEEDED,
        INVALID_SETTINGS
    } value;
};

struct sqp_info_t {
    int iter;
    int qp_solver_iter;
    sqp_status_t status;
};

/*
 * minimize     f(x)
 * subject to   ce(x)  = 0
 *              ci(x) <= 0
 *              l <= x <= u
 */
template <typename _Problem>
class SQP {
public:
    using Problem = _Problem;
    enum {
        VAR_SIZE = Problem::VAR_SIZE,
        NUM_EQ = Problem::NUM_EQ,
        NUM_INEQ = Problem::NUM_INEQ,
        NUM_CONSTR = NUM_EQ + NUM_INEQ + VAR_SIZE
    };

    using Scalar = typename Problem::Scalar;
    using qp_t = qp_solver::QP<VAR_SIZE, NUM_CONSTR, Scalar>;
    using qp_solver_t = qp_solver::QPSolver<qp_t>;
    using settings_t = sqp_settings_t<Scalar>;

    using var_t = Eigen::Matrix<Scalar, VAR_SIZE, 1>;
    using dual_t = Eigen::Matrix<Scalar, NUM_CONSTR, 1>;
    using gradient_t = Eigen::Matrix<Scalar, VAR_SIZE, 1>;
    using hessian_t = Eigen::Matrix<Scalar, VAR_SIZE, VAR_SIZE>;

    using constr_eq_t = Eigen::Matrix<Scalar, NUM_EQ, 1>;
    using jacobian_eq_t = Eigen::Matrix<Scalar, NUM_EQ, VAR_SIZE>;
    using constr_ineq_t = Eigen::Matrix<Scalar, NUM_INEQ, 1>;
    using jacobian_ineq_t = Eigen::Matrix<Scalar, NUM_INEQ, VAR_SIZE>;
    using constr_box_t = var_t;

    // Constants
    static constexpr Scalar DIV_BY_ZERO_REGUL = std::numeric_limits<Scalar>::epsilon();

    // Solver state variables
    var_t _x;
    dual_t _lambda;
    var_t _step_prev;
    gradient_t _grad_L;

    constr_eq_t b_eq;
    jacobian_eq_t A_eq;
    constr_ineq_t b_ineq;
    jacobian_ineq_t A_ineq;
    constr_box_t lbx, ubx;

    qp_t _qp;
    qp_solver_t _qp_solver;

    settings_t _settings;
    sqp_info_t _info;

    // info
    Scalar _dual_step_norm;
    Scalar _primal_step_norm;
    Scalar _cost;

    // enforce 16 byte alignment https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void solve(Problem &prob, const var_t &x0, const dual_t &lambda0)
    {
        _x = x0;
        _lambda = lambda0;
        _solve(prob);
    }

    void solve(Problem &prob)
    {
        _x.setZero();
        _lambda.setZero();
        _solve(prob);
    }

    SQP()
    {
        _qp_solver.settings().warm_start = true;
        _qp_solver.settings().check_termination = 10;
        _qp_solver.settings().eps_abs = 1e-4;
        _qp_solver.settings().eps_rel = 1e-4;
        _qp_solver.settings().max_iter = 100;
        _qp_solver.settings().adaptive_rho = true;
        _qp_solver.settings().adaptive_rho_interval = 50;
        _qp_solver.settings().alpha = 1.6;
    }

    inline const var_t& primal_solution() const { return _x; }
    inline var_t& primal_solution() { return _x; }

    inline const dual_t& dual_solution() const { return _lambda; }
    inline dual_t& dual_solution() { return _lambda; }

    inline const settings_t& settings() const { return _settings; }
    inline settings_t& settings() { return _settings; }

    inline const sqp_info_t& info() const { return _info; }
    inline sqp_info_t& info() { return _info; }

private:

    void _solve(Problem &prob)
    {
        var_t p; // search direction
        dual_t p_lambda; // dual search direction
        Scalar alpha; // step size

        // initialize
        _info.qp_solver_iter = 0;

        int& iter = _info.iter;
        for (iter = 1; iter <= _settings.max_iter; iter++) {
            // Solve QP
            solve_qp(prob, p, p_lambda);
            p_lambda -= _lambda;

            alpha = line_search(prob, p);

            // take step
            _x = _x + alpha * p;
            _lambda = _lambda + alpha * p_lambda;

            // update step info
            _step_prev = alpha * p;
            _primal_step_norm = alpha * p.template lpNorm<Eigen::Infinity>();
            _dual_step_norm = alpha * p_lambda.template lpNorm<Eigen::Infinity>();

            if (_settings.iteration_callback != nullptr) {
                _settings.iteration_callback(this);
            }

            if (termination_criteria(_x, prob)) {
                _info.status.value = sqp_status_t::SOLVED;
                break;
            }
        }

        if (iter > _settings.max_iter) {
            _info.status.value = sqp_status_t::MAX_ITER_EXCEEDED;
        }
    }

#ifdef SOLVER_DEBUG
    bool _is_posdef(hessian_t H)
    {
        Eigen::EigenSolver<hessian_t> eigensolver(H);
        for (int i = 0; i < eigensolver.eigenvalues().rows(); i++) {
            double v = eigensolver.eigenvalues()(i).real();
            if (v <= 0) {
                return false;
            }
        }
        return true;
    }
#endif

    bool termination_criteria(const var_t &x, Problem &prob) const
    {
        if (_primal_step_norm <= _settings.eps_prim &&
            _dual_step_norm <= _settings.eps_dual &&
            max_constraint_violation(x, prob) <= _settings.eps_prim) {
            return true;
        }
        return false;
    }

    void solve_qp(Problem &prob, var_t &p, dual_t &lambda)
    {
        /* QP from linearized NLP:
         * minimize     0.5 x'.P.x + q'.x
         * subject to   Ae.x + be  = 0
         *              Ai.x + bi <= 0
         *              l <= x <= u
         *
         * with:
         *   P      Hessian of Lagrangian
         *   q      cost gradient
         *   Ae,be  linearized equality constraint
         *   Ai,bi  linearized inequality constraint
         *   l,u    box constraint bounds
         *
         * transform to:
         * minimize     0.5 x'.P.x + q'.x
         * subject to   l <= A.x <= u
         *
         * Where the constraint bounds l,u are l=u for equality constraints or
         * set to +-INFINITY if unbounded.
         */
        enum {
            EQ_IDX = 0,
            INEQ_IDX = NUM_EQ,
            BOX_IDX = NUM_INEQ + NUM_EQ,
        };
        const Scalar UNBOUNDED = 1e20;

        gradient_t& grad_f = _qp.q;
        hessian_t& B = _qp.P;

        prob.cost_linearized(_x, grad_f, _cost);
        prob.constraint_linearized(_x, A_eq, b_eq, A_ineq, b_ineq, lbx, ubx);

        Eigen::Ref<constr_eq_t> lambda_eq = _lambda.template segment<NUM_EQ>(EQ_IDX);
        Eigen::Ref<constr_ineq_t> lambda_ineq = _lambda.template segment<NUM_INEQ>(INEQ_IDX);
        Eigen::Ref<constr_box_t> lambda_box = _lambda.template segment<VAR_SIZE>(BOX_IDX);

        gradient_t y = -_grad_L;
        _grad_L = grad_f +
                  A_eq.transpose() * lambda_eq +
                  A_ineq.transpose() * lambda_ineq +
                  lambda_box;

        // BFGS update
        if (_info.iter == 1) {
            B.setIdentity();
        } else {
            y += _grad_L; // y = grad_L_prev - grad_L
            BFGS_update(B, _step_prev, y);
#ifdef SOLVER_DEBUG
            SOLVER_ASSERT(_is_posdef(B));
#endif
        }

        // Equality constraints
        // from        A.x + b  = 0
        // to    -b <= A.x     <= -b
        _qp.u.template segment<NUM_EQ>(EQ_IDX) = -b_eq;
        _qp.l.template segment<NUM_EQ>(EQ_IDX) = -b_eq;
        _qp.A.template block<NUM_EQ, VAR_SIZE>(EQ_IDX, 0) = A_eq;

        // Inequality constraints
        // from          A.x + b <= 0
        // to    -INF <= A.x     <= -b
        _qp.u.template segment<NUM_INEQ>(INEQ_IDX) = -b_ineq;
        _qp.l.template segment<NUM_INEQ>(INEQ_IDX).setConstant(-UNBOUNDED);
        _qp.A.template block<NUM_INEQ, VAR_SIZE>(INEQ_IDX, 0) = A_ineq;

        // Box constraints
        // from     l <= x + p <= u
        // to     l-x <= p     <= u-x
        _qp.u.template segment<VAR_SIZE>(BOX_IDX) = ubx - _x;
        _qp.l.template segment<VAR_SIZE>(BOX_IDX) = lbx - _x;
        _qp.A.template block<VAR_SIZE, VAR_SIZE>(BOX_IDX, 0).setIdentity();

        // solve the QP
        if (_info.iter == 1) {
            _qp_solver.setup(_qp);
        } else {
            _qp_solver.update_qp(_qp);
        }
        _qp_solver.solve(_qp);

        _info.qp_solver_iter += _qp_solver.info().iter;

        p = _qp_solver.x;
        lambda = _qp_solver.y;
    }

    /** Line search in direction p using l1 merit function. */
    Scalar line_search(Problem &prob, const var_t& p)
    {
        // Note: using members _cost and _qp.q, which are updated in solve_qp().

        Scalar mu, phi_l1, Dp_phi_l1;
        gradient_t& cost_gradient = _qp.q;
        const Scalar tau = _settings.tau; // line search step decrease, 0 < tau < settings.tau

        Scalar constr_l1 = constraint_norm(_x, prob);

        // TODO: get mu from merit function model using hessian of Lagrangian
        mu = cost_gradient.dot(p) / ((1 - _settings.rho) * constr_l1);

        phi_l1 = _cost + mu * constr_l1;
        Dp_phi_l1 = cost_gradient.dot(p) - mu * constr_l1;

        Scalar alpha = 1.0;
        for (int i = 1; i < _settings.line_search_max_iter; i++) {
            Scalar cost_step;
            var_t x_step = _x + alpha*p;
            prob.cost(x_step, cost_step);

            Scalar phi_l1_step = cost_step + mu * constraint_norm(x_step, prob);
            if (phi_l1_step <= phi_l1 + alpha * _settings.eta * Dp_phi_l1) {
                // accept step
                break;
            } else {
                alpha = tau * alpha;
            }
        }
        return alpha;
    }

    /** L1 norm of constraint violation */
    Scalar constraint_norm(const var_t &x, Problem &prob) const
    {
        Scalar cl1 = DIV_BY_ZERO_REGUL;
        constr_eq_t c_eq;
        constr_ineq_t c_ineq;
        constr_box_t lbx, ubx;

        prob.constraint(x, c_eq, c_ineq, lbx, ubx);

        // c_eq = 0
        cl1 += c_eq.template lpNorm<1>();

        // c_ineq <= 0
        cl1 += c_ineq.cwiseMax(0.0).sum();

        // l <= x <= u
        cl1 += (lbx - x).cwiseMax(0.0).sum();
        cl1 += (x - ubx).cwiseMax(0.0).sum();

        return cl1;
    }

    /** L_inf norm of constraint violation */
    Scalar max_constraint_violation(const var_t &x, Problem &prob) const
    {
        Scalar c = 0;
        constr_eq_t c_eq;
        constr_ineq_t c_ineq;
        constr_box_t lbx, ubx;

        prob.constraint(x, c_eq, c_ineq, lbx, ubx);

        // c_eq = 0
        if (NUM_EQ > 0) {
            c = fmax(c, c_eq.template lpNorm<Eigen::Infinity>());
        }

        // c_ineq <= 0
        if (NUM_INEQ > 0) {
            c = fmax(c, c_ineq.maxCoeff());;
        }

        // l <= x <= u
        c = fmax(c, (lbx - x).maxCoeff());
        c = fmax(c, (x - ubx).maxCoeff());

        return c;
    }
};

} // namespace sqp

#endif // SQP_H
