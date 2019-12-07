#include <Eigen/Eigenvalues>
#include <cmath>
#include <iostream>
#include <solvers/bfgs.hpp>
#include <solvers/qp.hpp>
#include <solvers/sqp.hpp>

#ifndef SOLVER_ASSERT
#define SOLVER_ASSERT(x)  // NOP
#endif

namespace sqp {

template <typename Scalar>
class QPSolver {
   public:
    qp_solver::QuadraticProblem<Scalar> qp_;
    qp_solver::QPSolver<Scalar> qp_solver_;

    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void solve(const Matrix& P, const Vector& q, const Matrix& A, const Vector& l, const Vector& u,
               Vector& prim, Vector& dual) {
        qp_solver_.settings().warm_start = true;
        qp_solver_.settings().check_termination = 10;
        qp_solver_.settings().eps_abs = 1e-4;
        qp_solver_.settings().eps_rel = 1e-4;
        qp_solver_.settings().max_iter = 100;
        qp_solver_.settings().adaptive_rho = true;
        qp_solver_.settings().adaptive_rho_interval = 50;
        qp_solver_.settings().alpha = 1.6;

        // std::cout << P << std::endl;
        qp_.P = &P;
        // std::cout << q.transpose() << std::endl;
        qp_.q = &q;
        // std::cout << A << std::endl;
        qp_.A = &A;
        // std::cout << l.transpose() << std::endl;
        qp_.l = &l;
        // std::cout << u.transpose() << std::endl;
        qp_.u = &u;

        qp_solver_.setup(qp_);
        // qp_solver_.update_qp(qp_);
        qp_solver_.solve(qp_);

        prim = qp_solver_.primal_solution();
        dual = qp_solver_.dual_solution();
    }
};

template <typename T>
void SQP<T>::solve(Problem& prob, const Vector& x0, const Vector& lambda0) {
    _x = x0;
    _lambda = lambda0;
    _solve(prob);
}

template <typename T>
void SQP<T>::solve(Problem& prob) {
    const int nx = prob.num_var;
    const int nc = prob.num_ineq + prob.num_eq + prob.num_var;

    _x.setZero(nx);
    _lambda.setZero(nc);
    _solve(prob);
}

template <typename T>
SQP<T>::SQP() {}

template <typename T>
void SQP<T>::_solve(Problem& prob) {
    Vector p;         // search direction
    Vector p_lambda;  // dual search direction
    Scalar alpha;     // step size

    const int nx = prob.num_var;
    const int neq = prob.num_eq;
    const int nineq = prob.num_ineq;
    const int nc = prob.num_ineq + prob.num_eq + prob.num_var;

    printf("nx %d\n", nx);
    printf("neq %d\n", neq);
    printf("nineq %d\n", nineq);
    printf("nc %d\n", nc);

    p.resize(nx);
    p_lambda.resize(nc);
    _step_prev.resize(nx);
    _grad_L.resize(nx);

    b_eq.resize(neq);
    A_eq.resize(neq, nx);
    b_ineq.resize(nineq);
    A_ineq.resize(nineq, nx);
    lb_.resize(nx);
    ub_.resize(nx);

    P.resize(nx, nx);
    q.resize(nx);
    A.resize(nc, nx);

    // initialize
    _info.qp_solver_iter = 0;

    int& iter = _info.iter;
    for (iter = 1; iter <= _settings.max_iter; iter++) {
        // Solve QP
        printf("solve_qp\n");
        solve_qp(prob, p, p_lambda);
        p_lambda -= _lambda;

        printf("line_search\n");
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
        printf("iter %d\n", iter);
        std::cout << "x " << _x.transpose() << std::endl;
        std::cout << "y " << _lambda.transpose() << std::endl;
    }
    printf("DONE iter %d\n", iter);
    if (iter > _settings.max_iter) {
        _info.status.value = sqp_status_t::MAX_ITER_EXCEEDED;
    }
}

template <typename Matrix>
bool _is_posdef_eigen(Matrix H) {
    Eigen::EigenSolver<Matrix> eigensolver(H);
    for (int i = 0; i < eigensolver.eigenvalues().rows(); i++) {
        double v = eigensolver.eigenvalues()(i).real();
        if (v <= 0) {
            return false;
        }
    }
    return true;
}

template <typename Matrix>
bool _is_posdef(Matrix H) {
    Eigen::LLT<Matrix> llt(H);
    if (llt.info() == Eigen::NumericalIssue) {
        return false;
    }
    return true;
}

template <typename T>
bool SQP<T>::termination_criteria(const Vector& x, Problem& prob) const {
    printf("termination_criteria\n");
    if (_primal_step_norm <= _settings.eps_prim && _dual_step_norm <= _settings.eps_dual &&
        max_constraint_violation(x, prob) <= _settings.eps_prim) {
        return true;
    }
    return false;
}

template <typename T>
void SQP<T>::solve_qp(Problem& prob, Vector& p, Vector& lambda) {
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
    const int EQ_IDX = 0;
    const int INEQ_IDX = prob.num_eq;
    const int BOX_IDX = prob.num_ineq + prob.num_eq;
    const Scalar UNBOUNDED = 1e20;

    const int nc = prob.num_ineq + prob.num_eq + prob.num_var;
    const int nx = prob.num_var;

    Vector& grad_f = q;
    Matrix& B = P;

    prob.objective_linearized(_x, grad_f, _cost);
    prob.constraint_linearized(_x, A_eq, b_eq, A_ineq, b_ineq, lb_, ub_);

    Eigen::Ref<Vector> lambda_eq = _lambda.segment(EQ_IDX, prob.num_eq);
    Eigen::Ref<Vector> lambda_ineq = _lambda.segment(INEQ_IDX, prob.num_ineq);
    Eigen::Ref<Vector> lambda_box = _lambda.segment(BOX_IDX, prob.num_var);

    Vector y = -_grad_L;
    _grad_L = grad_f + A_eq.transpose() * lambda_eq + A_ineq.transpose() * lambda_ineq + lambda_box;

    printf("BFGS\n");
    // BFGS update
    if (_info.iter == 1) {
        B.setIdentity();
    } else {
        y += _grad_L;  // y = grad_L_prev - grad_L
        BFGS_update(B, _step_prev, y);
        SOLVER_ASSERT(_is_posdef(B));
    }
    std::cout << B << std::endl;

    Vector l, u;
    l.resize(nc);
    u.resize(nc);
    A.resize(nc, nx);

    // Equality constraints
    // from        A.x + b  = 0
    // to    -b <= A.x     <= -b
    u.segment(EQ_IDX, prob.num_eq) = -b_eq;
    l.segment(EQ_IDX, prob.num_eq) = -b_eq;
    A.block(EQ_IDX, 0, prob.num_eq, prob.num_var) = A_eq;

    // Inequality constraints
    // from          A.x + b <= 0
    // to    -INF <= A.x     <= -b
    u.segment(INEQ_IDX, prob.num_ineq) = -b_ineq;
    l.segment(INEQ_IDX, prob.num_ineq).setConstant(-UNBOUNDED);
    A.block(INEQ_IDX, 0, prob.num_ineq, prob.num_var) = A_ineq;

    // Box constraints
    // from     l <= x + p <= u
    // to     l-x <= p     <= u-x
    u.segment(BOX_IDX, prob.num_var) = ub_ - _x;
    l.segment(BOX_IDX, prob.num_var) = lb_ - _x;
    A.block(BOX_IDX, 0, prob.num_var, prob.num_var).setIdentity();

    printf("QPSolver\n");

    // solve the QP
    int iter;
    QPSolver<Scalar> qp_solver;
    qp_solver.solve(P, q, A, l, u, p, lambda);
}

/** Line search in direction p using l1 merit function. */
template <typename T>
typename SQP<T>::Scalar SQP<T>::line_search(Problem& prob, const Vector& p) {
    // Note: using members _cost and q, which are updated in solve_qp().

    Scalar mu, phi_l1, Dp_phi_l1;
    Vector& cost_gradient = q;
    const Scalar tau = _settings.tau;  // line search step decrease, 0 < tau < settings.tau

    Scalar constr_l1 = constraint_norm(_x, prob);

    // TODO: get mu from merit function model using hessian of Lagrangian
    mu = cost_gradient.dot(p) / ((1 - _settings.rho) * constr_l1);

    printf("mu %f\n", mu);
    // mu = fmax(0, mu);
    // mu *= 1e+2;

    phi_l1 = _cost + mu * constr_l1;
    Dp_phi_l1 = cost_gradient.dot(p) - mu * constr_l1;

    Scalar alpha = 1.0;
    for (int i = 1; i < _settings.line_search_max_iter; i++) {
        Scalar cost_step;
        Vector x_step = _x + alpha * p;
        prob.objective(x_step, cost_step);

        Scalar phi_l1_step = cost_step + mu * constraint_norm(x_step, prob);
        if (phi_l1_step <= phi_l1 + alpha * _settings.eta * Dp_phi_l1) {
            // accept step
            break;
        } else {
            alpha = tau * alpha;
        }
    }
    printf("alpha %f  mu %f\n", alpha, mu);
    return alpha;
}

/** L1 norm of constraint violation */
template <typename T>
typename SQP<T>::Scalar SQP<T>::constraint_norm(const Vector& x, Problem& prob) const {
    Scalar cl1 = DIV_BY_ZERO_REGUL;
    Vector c_eq;
    Vector c_ineq;
    Vector lb, ub;

    prob.constraint(x, c_eq, c_ineq, lb, ub);

    // c_eq = 0
    cl1 += c_eq.template lpNorm<1>();

    // c_ineq <= 0
    cl1 += c_ineq.cwiseMax(0.0).sum();

    // l <= x <= u
    cl1 += (lb - x).cwiseMax(0.0).sum();
    cl1 += (x - ub).cwiseMax(0.0).sum();

    return cl1;
}

/** L_inf norm of constraint violation */
template <typename T>
typename SQP<T>::Scalar SQP<T>::max_constraint_violation(const Vector& x, Problem& prob) const {
    Scalar c = 0;
    Vector c_eq;
    Vector c_ineq;
    Vector lb_, ub_;

    prob.constraint(x, c_eq, c_ineq, lb_, ub_);

    // c_eq = 0
    if (prob.num_eq > 0) {
        c = fmax(c, c_eq.template lpNorm<Eigen::Infinity>());
    }

    // c_ineq <= 0
    if (prob.num_ineq > 0) {
        c = fmax(c, c_ineq.maxCoeff());
    }

    // l <= x <= u
    c = fmax(c, (lb_ - x).maxCoeff());
    c = fmax(c, (x - ub_).maxCoeff());

    return c;
}

template class SQP<double>;
template class SQP<float>;

}  // namespace sqp
