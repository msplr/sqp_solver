#include "gtest/gtest.h"
#include "solvers/qp_solver.hpp"
#include <Eigen/IterativeLinearSolvers>

using namespace qp_solver;

template <typename _Scalar=double>
class _SimpleQP : public QP<2, 3, _Scalar>
{
public:
    Eigen::Matrix<_Scalar, 2, 1> SOLUTION;
    _SimpleQP()
    {
        this->P << 4, 1,
                   1, 2;
        this->q << 1, 1;
        this->A << 1, 1,
                   1, 0,
                   0, 1;
        this->l << 1, 0, 0;
        this->u << 1, 0.7, 0.7;

        this->SOLUTION << 0.3, 0.7;
    }
};

using SimpleQP = _SimpleQP<double>;

TEST(QPSolverTest, testSimpleQP) {
    SimpleQP qp;
    QPSolver<SimpleQP> prob;

    prob.settings().max_iter = 1000;

    prob.setup(qp);
    prob.solve(qp);
    Eigen::Vector2d sol = prob.primal_solution();

    EXPECT_TRUE(sol.isApprox(qp.SOLUTION, 1e-2));
    EXPECT_LT(prob.iter, prob.settings().max_iter);
    EXPECT_EQ(prob.info().status, SOLVED);
}


TEST(QPSolverTest, testSinglePrecisionFloat) {
    using SimpleQPf = _SimpleQP<float>;
    SimpleQPf qp;
    QPSolver<SimpleQPf> prob;

    prob.setup(qp);
    prob.solve(qp);
    Eigen::Vector2f sol = prob.primal_solution();

    EXPECT_TRUE(sol.isApprox(qp.SOLUTION, 1e-2));
    EXPECT_LT(prob.iter, prob.settings().max_iter);
    EXPECT_EQ(prob.info().status, SOLVED);
}

TEST(QPSolverTest, testConstraintViolation) {
    SimpleQP qp;
    QPSolver<SimpleQP> prob;

    prob.settings().eps_rel = 1e-4f;
    prob.settings().eps_abs = 1e-4f;

    prob.setup(qp);
    prob.solve(qp);
    Eigen::Vector2d sol = prob.primal_solution();

    // check feasibility (with some epsilon margin)
    Eigen::Vector3d lower = qp.A*sol - qp.l;
    Eigen::Vector3d upper = qp.A*sol - qp.u;
    EXPECT_GE(lower.minCoeff(), -1e-3);
    EXPECT_LE(upper.maxCoeff(), 1e-3);
}

TEST(QPSolverTest, testAdaptiveRho) {
    SimpleQP qp;
    QPSolver<SimpleQP> prob;

    prob.settings().adaptive_rho = false;
    prob.settings().adaptive_rho_interval = 10;

    prob.setup(qp);
    prob.solve(qp);

    EXPECT_EQ(prob.info().status, SOLVED);
}

TEST(QPSolverTest, testAdaptiveRhoImprovesConvergence) {
    SimpleQP qp;
    QPSolver<SimpleQP> prob;

    prob.settings().warm_start = false;
    prob.settings().max_iter = 1000;
    prob.settings().rho = 0.1;

    // solve whithout adaptive rho
    prob.settings().adaptive_rho = false;
    prob.setup(qp);
    prob.solve(qp);
    int prev_iter = prob.info().iter;

    // solve with adaptive rho
    prob.settings().adaptive_rho = true;
    prob.settings().adaptive_rho_interval = 10;
    prob.solve(qp);

    auto info = prob.info();
    EXPECT_LT(info.iter, prob.settings().max_iter);
    EXPECT_LT(info.iter, prev_iter); // adaptive rho should improve :)
    EXPECT_EQ(info.status, SOLVED);
}

/* BUG: Eigen::ConjugateGradient fails with assert when using fixed-size matrix.
 *      Only works in Release mode. */
#ifdef EIGEN_NO_DEBUG
TEST(QPSolverTest, testConjugateGradientLinearSolver)
{
    SimpleQP qp;
    QPSolver<SimpleQP, Eigen::ConjugateGradient, Eigen::Lower | Eigen::Upper> prob;

    prob.setup(qp);
    prob.solve(qp);
    Eigen::Vector2d sol = prob.primal_solution();

    auto info = prob.info();
    EXPECT_TRUE(sol.isApprox(qp.SOLUTION, 1e-2));
    EXPECT_EQ(info.status, SOLVED);
    EXPECT_LT(info.iter, prob.settings().max_iter); // convergence test
}
#endif

TEST(QPSolverTest, TestConstraint) {
    using qp_t = QP<5, 5, double>;
    using solver_t = QPSolver<qp_t>;
    solver_t prob;

    qp_t qp;
    qp.P.setIdentity();
    qp.q.setConstant(-1);
    qp.A.setIdentity();

    int type_expect[5];
    qp.l(0) = -1e+17;
    qp.u(0) = 1e+17;
    type_expect[0] = solver_t::LOOSE_BOUNDS;
    qp.l(1) = -101;
    qp.u(1) = 1e+17;
    type_expect[1] = solver_t::INEQUALITY_CONSTRAINT;
    qp.l(2) = -1e+17;
    qp.u(2) = 123;
    type_expect[2] = solver_t::INEQUALITY_CONSTRAINT;
    qp.l(3) = -1;
    qp.u(3) = 1;
    type_expect[3] = solver_t::INEQUALITY_CONSTRAINT;
    qp.l(4) = 42;
    qp.u(4) = 42;
    type_expect[4] = solver_t::EQUALITY_CONSTRAINT;

    prob.setup(qp);

    for (int i = 0; i < qp.l.rows(); i++) {
        EXPECT_EQ(prob.constr_type[i], type_expect[i]);
    }
}
