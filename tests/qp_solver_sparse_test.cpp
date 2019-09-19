#include "gtest/gtest.h"
#define QP_SOLVER_PRINTING
#define QP_SOLVER_USE_SPARSE
#include "solvers/qp_solver.hpp"

using namespace qp_solver;

class SimpleQP : public QP<2, 3, double>
{
public:
    Eigen::Matrix<double, 2, 1> SOLUTION;
    SimpleQP()
    {
        Eigen::MatrixXd P(2,2);
        Eigen::MatrixXd A(3,2);
        P << 4, 1,
             1, 2;
        A << 1, 1,
             1, 0,
             0, 1;

        this->P = P.sparseView();
        this->P_col_nnz << 2, 2;
        this->q << 1, 1;
        this->A = A.sparseView();
        this->A_col_nnz << 2, 2;
        this->l << 1, 0, 0;
        this->u << 1, 0.7, 0.7;

        this->SOLUTION << 0.3, 0.7;
    }
};

TEST(QPSolverSparseTest, testSimpleQP) {
    SimpleQP qp;
    QPSolver<SimpleQP> prob;

    prob.settings().max_iter = 1000;
    prob.settings().adaptive_rho = true;
    prob.settings().verbose = true;

    prob.setup(qp);
    prob.solve(qp);
    Eigen::Vector2d sol = prob.primal_solution();

    EXPECT_TRUE(sol.isApprox(qp.SOLUTION, 1e-2));
    EXPECT_LT(prob.iter, prob.settings().max_iter);
    EXPECT_EQ(prob.info().status, SOLVED);
}

TEST(QPSolverSparseTest, testConjugateGradient) {
    SimpleQP qp;
    QPSolver<SimpleQP, Eigen::ConjugateGradient, Eigen::Lower | Eigen::Upper> prob;

    prob.settings().max_iter = 1000;
    prob.settings().adaptive_rho = true;
    prob.settings().verbose = true;

    prob.setup(qp);
    prob.solve(qp);
    Eigen::Vector2d sol = prob.primal_solution();

    EXPECT_TRUE(sol.isApprox(qp.SOLUTION, 1e-2));
    EXPECT_LT(prob.iter, prob.settings().max_iter);
    EXPECT_EQ(prob.info().status, SOLVED);
}

TEST(QPSolverSparseTest, testCanMultipleSolve) {
    SimpleQP qp;
    QPSolver<SimpleQP> prob;

    prob.setup(qp);
    prob.solve(qp);
    EXPECT_EQ(prob.info().status, SOLVED);

    prob.solve(qp);
    EXPECT_EQ(prob.info().status, SOLVED);
}

TEST(QPSolverSparseTest, testCanUpdateQP) {
    SimpleQP qp;
    QPSolver<SimpleQP> prob;

    prob.setup(qp);
    prob.solve(qp);
    EXPECT_TRUE(prob.primal_solution().isApprox(qp.SOLUTION, 1e-2));
    EXPECT_EQ(prob.info().status, SOLVED);

    qp.P.setIdentity(); // dont't change P_col_nnz!
    qp.q << 0, 0;
    qp.SOLUTION << 0.5, 0.5;

    prob.update_qp(qp);
    prob.solve(qp);

    EXPECT_TRUE(prob.primal_solution().isApprox(qp.SOLUTION, 1e-2));
    EXPECT_EQ(prob.info().status, SOLVED);
}
