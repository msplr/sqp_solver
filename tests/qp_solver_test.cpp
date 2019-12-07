#include "gtest/gtest.h"
#include "solvers/qp.hpp"

using namespace qp_solver;

template <typename Scalar>
class SimpleQP : public QuadraticProblem<Scalar> {
   public:
    using BASE = QuadraticProblem<Scalar>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    SimpleQP() {
        Matrix* P = new Matrix(2, 2);
        Vector* q = new Vector(2);
        Matrix* A = new Matrix(3, 2);
        Vector* l = new Vector(3);
        Vector* u = new Vector(3);

        *P << 4, 1, 1, 2;
        *q << 1, 1;
        *A << 1, 1, 1, 0, 0, 1;
        *l << 1, 0, 0;
        *u << 1, 0.7, 0.7;

        BASE::P = P;
        BASE::q = q;
        BASE::A = A;
        BASE::l = l;
        BASE::u = u;

        SOLUTION << 0.3, 0.7;
    }
    ~SimpleQP() {
        delete BASE::P;
        delete BASE::q;
        delete BASE::A;
        delete BASE::l;
        delete BASE::u;
    }
    Eigen::Matrix<Scalar, 2, 1> SOLUTION;
};

TEST(QPSolverTest, testSimpleQP) {
    SimpleQP<double> qp;
    QPSolver<double> solver;

    solver.settings().max_iter = 1000;

    solver.setup(qp);
    solver.solve(qp);
    Eigen::Vector2d sol = solver.primal_solution();

    EXPECT_TRUE(sol.isApprox(qp.SOLUTION, 1e-2));
    EXPECT_LT(solver.info().iter, solver.settings().max_iter);
    EXPECT_EQ(solver.info().status, SOLVED);
}

TEST(QPSolverTest, testSinglePrecisionFloat) {
    SimpleQP<float> qp;
    QPSolver<float> solver;

    solver.setup(qp);
    solver.solve(qp);
    Eigen::Vector2f sol = solver.primal_solution();

    EXPECT_TRUE(sol.isApprox(qp.SOLUTION, 1e-2));
    EXPECT_LT(solver.info().iter, solver.settings().max_iter);
    EXPECT_EQ(solver.info().status, SOLVED);
}

TEST(QPSolverTest, testConstraintViolation) {
    SimpleQP<double> qp;
    QPSolver<double> solver;

    solver.settings().eps_rel = 1e-4f;
    solver.settings().eps_abs = 1e-4f;

    solver.setup(qp);
    solver.solve(qp);
    Eigen::Vector2d sol = solver.primal_solution();

    // check feasibility (with some epsilon margin)
    Eigen::Vector3d lower = (*qp.A) * sol - (*qp.l);
    Eigen::Vector3d upper = (*qp.A) * sol - (*qp.u);
    EXPECT_GE(lower.minCoeff(), -1e-3);
    EXPECT_LE(upper.maxCoeff(), 1e-3);
}

TEST(QPSolverTest, testAdaptiveRho) {
    SimpleQP<double> qp;
    QPSolver<double> solver;

    solver.settings().adaptive_rho = true;
    solver.settings().adaptive_rho_interval = 10;

    solver.setup(qp);
    solver.solve(qp);

    EXPECT_EQ(solver.info().status, SOLVED);
}

TEST(QPSolverTest, testAdaptiveRhoImprovesConvergence) {
    SimpleQP<double> qp;
    QPSolver<double> solver;

    solver.settings().warm_start = false;
    solver.settings().max_iter = 1000;
    solver.settings().rho = 0.1;

    // solve whithout adaptive rho
    solver.settings().adaptive_rho = false;
    solver.setup(qp);
    solver.solve(qp);
    int prev_iter = solver.info().iter;

    // solve with adaptive rho
    solver.settings().adaptive_rho = true;
    solver.settings().adaptive_rho_interval = 10;
    solver.solve(qp);

    auto info = solver.info();
    EXPECT_LT(info.iter, solver.settings().max_iter);
    EXPECT_LT(info.iter, prev_iter);  // adaptive rho should improve :)
    EXPECT_EQ(info.status, SOLVED);
}

TEST(QPSolverTest, TestConstraint) {
    using Solver = QPSolver<double>;

    Eigen::VectorXd l = Eigen::VectorXd(5);
    Eigen::VectorXd u = Eigen::VectorXd(5);

    int type_expect[5];
    l(0) = -10 * Solver::LOOSE_BOUNDS_THRESH;
    u(0) = 10 * Solver::LOOSE_BOUNDS_THRESH;
    type_expect[0] = Solver::LOOSE_BOUNDS;
    l(1) = -1;
    u(1) = 10 * Solver::LOOSE_BOUNDS_THRESH;
    type_expect[1] = Solver::INEQUALITY_CONSTRAINT;
    l(2) = -10 * Solver::LOOSE_BOUNDS_THRESH;
    u(2) = 2;
    type_expect[2] = Solver::INEQUALITY_CONSTRAINT;
    l(3) = -3;
    u(3) = 4;
    type_expect[3] = Solver::INEQUALITY_CONSTRAINT;
    l(4) = 42;
    u(4) = 42;
    type_expect[4] = Solver::EQUALITY_CONSTRAINT;

    Eigen::VectorXi constr_type(5);
    Solver::constr_type_init(l, u, constr_type);

    for (int i = 0; i < l.rows(); i++) {
        EXPECT_EQ(constr_type[i], type_expect[i]);
    }
}
