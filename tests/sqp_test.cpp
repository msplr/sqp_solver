#include <gtest/gtest.h>
#include <solvers/sqp.hpp>

using namespace sqp;

namespace sqp_test {

struct SimpleNLP : public NonLinearProblem<double> {
    using Vector = NonLinearProblem<double>::Vector;
    using Matrix = NonLinearProblem<double>::Matrix;

    const Scalar infinity = std::numeric_limits<Scalar>::infinity();
    Eigen::Vector2d SOLUTION = {1, 1};

    SimpleNLP() {
        num_var = 2;
        num_constr = 3;
    }

    void objective(const Vector& x, Scalar& obj) { obj = -x.sum(); }

    void objective_linearized(const Vector& x, Vector& grad, Scalar& obj) {
        grad.resize(num_var);

        objective(x, obj);
        grad << -1, -1;
    }

    void constraint(const Vector& x, Vector& c, Vector& l, Vector& u) {
        // 1 <= x0^2 + x1^2 <= 2
        // 0 <= x0
        // 0 <= x1
        c << x.squaredNorm(), x;
        l << 1, 0, 0;
        u << 2, infinity, infinity;
    }

    void constraint_linearized(const Vector& x, Matrix& Jc, Vector& c, Vector& l, Vector& u) {
        Jc.resize(3, 2);

        constraint(x, c, l, u);
        Jc << 2 * x.transpose(), Matrix::Identity(2, 2);
    }
};

TEST(SQPTestCase, TestSimpleNLP) {
    SimpleNLP problem;
    SQP<double> solver;
    Eigen::Vector2d x;

    // feasible initial point
    Eigen::Vector2d x0 = {1.2, 0.1};
    Eigen::Vector3d y0 = Eigen::VectorXd::Zero(3);

    // TODO(mi): doesn't work for higher values
    solver.settings().line_search_max_iter = 10;
    solver.settings().max_iter = 100;

    solver.solve(problem, x0, y0);
    x = solver.primal_solution();

    // solver.info().print();
    // std::cout << "primal solution " << solver.primal_solution().transpose() << std::endl;
    // std::cout << "dual solution " << solver.dual_solution().transpose() << std::endl;

    EXPECT_TRUE(x.isApprox(problem.SOLUTION, 1e-2));
    EXPECT_LT(solver.info().iter, solver.settings().max_iter);
}

TEST(SQPTestCase, InfeasibleStart) {
    SimpleNLP problem;
    SQP<double> solver;
    Eigen::Vector2d x;

    // infeasible initial point
    Eigen::Vector2d x0 = {2, -1};
    Eigen::Vector3d y0;
    y0.setOnes();

    solver.settings().max_iter = 100;
    solver.solve(problem, x0, y0);
    x = solver.primal_solution();

    EXPECT_TRUE(x.isApprox(problem.SOLUTION, 1e-2));
    EXPECT_LT(solver.info().iter, solver.settings().max_iter);
}

struct SimpleQP : public NonLinearProblem<double> {
    using Vector = NonLinearProblem<double>::Vector;
    using Matrix = NonLinearProblem<double>::Matrix;

    Eigen::Matrix2d P;
    Eigen::Vector2d q = {1, 1};
    Eigen::Vector2d SOLUTION = {0.3, 0.7};

    SimpleQP() {
        P << 4, 1, 1, 2;
        num_var = 2;
        num_constr = 3;
    }

    void objective(const Vector& x, Scalar& obj) final { obj = 0.5 * x.dot(P * x) + q.dot(x); }

    void objective_linearized(const Vector& x, Vector& grad, Scalar& obj) final {
        objective(x, obj);
        grad = P * x + q;
    }

    void constraint(const Vector& x, Vector& c, Vector& l, Vector& u) final {
        // x1 + x2 = 1
        c << x.sum(), x;
        l << 1, 0, 0;
        u << 1, 0.7, 0.7;
    }

    void constraint_linearized(const Vector& x, Matrix& Jc, Vector& c, Vector& l, Vector& u) final {
        constraint(x, c, l, u);
        Jc << 1, 1, Matrix::Identity(2, 2);
    }
};

TEST(SQPTestCase, TestSimpleQP) {
    SimpleQP problem;
    SQP<double> solver;
    Eigen::VectorXd x0 = Eigen::Vector2d(0,0);
    Eigen::VectorXd y0 = Eigen::Vector3d(0,0,0);

    solver.solve(problem, x0, y0);

    // solver.info().print();
    // std::cout << "primal solution " << solver.primal_solution().transpose() << std::endl;
    // std::cout << "dual solution " << solver.dual_solution().transpose() << std::endl;

    EXPECT_TRUE(solver.primal_solution().isApprox(problem.SOLUTION, 1e-2));
    EXPECT_LT(solver.info().iter, solver.settings().max_iter);
}

}  // namespace sqp_test
