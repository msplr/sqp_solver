#include "gtest/gtest.h"
#define SOLVER_DEBUG
#include "solvers/sqp.hpp"

using namespace sqp;

namespace sqp_test {

template <typename _Scalar, int _VAR_SIZE, int _NUM_EQ=0, int _NUM_INEQ=0>
struct ProblemBase {
    enum {
        VAR_SIZE = _VAR_SIZE,
        NUM_EQ = _NUM_EQ,
        NUM_INEQ = _NUM_INEQ,
    };

    using Scalar = double;
    using var_t = Eigen::Matrix<Scalar, VAR_SIZE, 1>;
    using grad_t = Eigen::Matrix<Scalar, VAR_SIZE, 1>;
    using hessian_t = Eigen::Matrix<Scalar, VAR_SIZE, VAR_SIZE>;

    using b_eq_t = Eigen::Matrix<Scalar, NUM_EQ, 1>;
    using A_eq_t = Eigen::Matrix<Scalar, NUM_EQ, VAR_SIZE>;
    using b_ineq_t = Eigen::Matrix<Scalar, NUM_INEQ, 1>;
    using A_ineq_t = Eigen::Matrix<Scalar, NUM_INEQ, VAR_SIZE>;
};

struct SimpleNLP_2D {
    enum {
        VAR_SIZE = 2,
        NUM_EQ = 0,
        NUM_INEQ = 2,
    };
    using Scalar = double;
    using var_t = Eigen::Vector2d;
    using grad_t = Eigen::Vector2d;

    using b_eq_t = Eigen::Matrix<Scalar, NUM_EQ, 1>;
    using A_eq_t = Eigen::Matrix<Scalar, NUM_EQ, VAR_SIZE>;
    using b_ineq_t = Eigen::Matrix<Scalar, NUM_INEQ, 1>;
    using A_ineq_t = Eigen::Matrix<Scalar, NUM_INEQ, VAR_SIZE>;

    var_t SOLUTION = {1, 1};

    void cost(const var_t& x, Scalar &cst)
    {
        cst = -x(0) -x(1);
    }

    void cost_linearized(const var_t& x, grad_t &grad, Scalar &cst)
    {
        cost(x, cst);
        grad << -1, -1;
    }

    void constraint(const var_t& x, b_eq_t& b_eq, b_ineq_t& b_ineq, var_t& lbx, var_t& ubx)
    {
        const Scalar infinity = std::numeric_limits<Scalar>::infinity();
        b_ineq << 1 - x.squaredNorm(), x.squaredNorm() - 2; // 1 <= x0^2 + x1^2 <= 2
        lbx << 0, 0; // x0 > 0 and x1 > 0
        ubx << infinity, infinity;
    }

    void constraint_linearized(const var_t& x,
                               A_eq_t& A_eq,
                               b_eq_t& b_eq,
                               A_ineq_t& A_ineq,
                               b_ineq_t& b_ineq,
                               var_t& lbx,
                               var_t& ubx)
    {
        constraint(x, b_eq, b_ineq, lbx, ubx);
        A_ineq << -2*x.transpose(),
                   2*x.transpose();
    }
};

TEST(SQPTestCase, TestSimpleNLP) {
    SimpleNLP_2D problem;
    SQP<SimpleNLP_2D> solver;
    Eigen::Vector2d x;

    // feasible initial point
    Eigen::Vector2d x0 = {1.2, 0.1};
    Eigen::Vector4d y0;
    y0.setZero();

    solver.settings().max_iter = 100;
    solver.solve(problem, x0, y0);
    x = solver.primal_solution();

    EXPECT_TRUE(x.isApprox(problem.SOLUTION, 1e-2));
    EXPECT_LT(solver.info().iter, solver.settings().max_iter);
}

TEST(SQPTestCase, InfeasibleStart) {
    SimpleNLP_2D problem;
    SQP<SimpleNLP_2D> solver;
    Eigen::Vector2d x;

    // infeasible initial point
    Eigen::Vector2d x0 = {2, -1};
    Eigen::Vector4d y0;
    y0.setOnes();

    solver.settings().max_iter = 100;
    solver.solve(problem, x0, y0);
    x = solver.primal_solution();

    EXPECT_TRUE(x.isApprox(problem.SOLUTION, 1e-2));
    EXPECT_LT(solver.info().iter, solver.settings().max_iter);
}


struct SimpleQP : public ProblemBase<double, 2, 1, 0> {
    Eigen::Matrix2d P;
    Eigen::Vector2d q;
    Eigen::Vector2d SOLUTION;

    SimpleQP()
    {
        P << 4, 1,
             1, 2;
        q << 1, 1;
        SOLUTION << 0.3, 0.7;
    }

    void cost(const var_t& x, Scalar &cst)
    {
        cst = 0.5 * x.dot(P*x) + q.dot(x);
    }

    void cost_linearized(const var_t& x, grad_t &grad, Scalar &cst)
    {
        cost(x, cst);
        grad << P*x + q;
    }

    void constraint(const var_t& x, b_eq_t& b_eq, b_ineq_t& b_ineq, var_t& lbx, var_t& ubx)
    {
        b_eq << x.sum() - 1; // x1 + x2 = 1
        lbx << 0, 0;
        ubx << 0.7, 0.7;
    }

    void constraint_linearized(const var_t& x, A_eq_t& A_eq, b_eq_t& b_eq, A_ineq_t& A_ineq, b_ineq_t& b_ineq, var_t& lbx, var_t& ubx)
    {
        constraint(x, b_eq, b_ineq, lbx, ubx);
        A_eq << 1, 1;
    }
};

TEST(SQPTestCase, TestSimpleQP) {
    SimpleQP problem;
    SQP<SimpleQP> solver;
    Eigen::Vector2d x;

    solver.solve(problem);
    x = solver.primal_solution();

    EXPECT_TRUE(x.isApprox(problem.SOLUTION, 1e-2));
    EXPECT_LT(solver.info().iter, solver.settings().max_iter);
}

} // sqp_test
