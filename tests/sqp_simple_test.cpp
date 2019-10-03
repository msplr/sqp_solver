#include <gtest/gtest.h>
#include <solvers/sqp.hpp>

using namespace sqp;

namespace sqp_test {

struct SimpleNLP : Problem<double> {
	using Vector = Problem<double>::Vector;
	using Matrix = Problem<double>::Matrix;

    const Scalar infinity = std::numeric_limits<Scalar>::infinity();
    Eigen::Vector2d SOLUTION = {1, 1};

    SimpleNLP() {
    	num_var = 2;
    	num_ineq = 2;
    	num_eq = 0;
    }

    void objective(const Vector& x, Scalar &cst)
    {
        // printf("objective\n");
        cst = -x(0) -x(1);
    }

    void objective_linearized(const Vector& x, Vector &grad, Scalar &cst)
    {
        // printf("objective_linearized\n");
        grad.resize(2);

        objective(x, cst);
        grad << -1, -1;
    }

    void constraint(const Vector& x, Vector& b_eq, Vector& b_ineq, Vector& lb, Vector& ub)
    {
        // printf("constraint\n");
        b_ineq.resize(2);
        b_eq.resize(0);
        lb.resize(2);
        ub.resize(2);

        b_ineq << 1 - x.squaredNorm(), x.squaredNorm() - 2; // 1 <= x0^2 + x1^2 <= 2
        lb << 0, 0; // x0 > 0 and x1 > 0
        ub << infinity, infinity;
    }

    void constraint_linearized(const Vector& x,
                               Matrix& A_eq,
                               Vector& b_eq,
                               Matrix& A_ineq,
                               Vector& b_ineq,
                               Vector& lb,
                               Vector& ub)
    {
        // printf("constraint_linearized\n");
        A_ineq.resize(2,2);
        A_eq.resize(0,2);

        constraint(x, b_eq, b_ineq, lb, ub);
        A_ineq << -2*x.transpose(),
                   2*x.transpose();
    }
};

TEST(SQPTestCase, TestSimpleNLP) {
    SimpleNLP problem;
    SQP<double> solver;
    Eigen::Vector2d x;

    // feasible initial point
    Eigen::Vector2d x0 = {1.2, 0.1};
    Eigen::Vector4d y0;
    y0.setZero();

    solver.settings().max_iter = 100;
    solver.solve(problem, x0, y0);
    x = solver.primal_solution();

    std::cout << problem.SOLUTION << std::endl;

    EXPECT_TRUE(x.isApprox(problem.SOLUTION, 1e-2));
    EXPECT_LT(solver.info().iter, solver.settings().max_iter);
}

TEST(SQPTestCase, InfeasibleStart) {
    SimpleNLP problem;
    SQP<double> solver;
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

} // sqp_test