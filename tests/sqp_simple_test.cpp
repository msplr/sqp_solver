#include <gtest/gtest.h>
#include <solvers/sqp.hpp>
#include <unsupported/Eigen/AutoDiff>

using namespace sqp;

namespace sqp_test {

struct SimpleNLP : public NonLinearProblem<double> {
    using Vector = NonLinearProblem<double>::Vector;
    using Matrix = NonLinearProblem<double>::Matrix;

    const Scalar infinity = std::numeric_limits<Scalar>::infinity();
    Eigen::Vector2d SOLUTION = {1, 1};

    SimpleNLP() {
        num_var = 2;
        num_ineq = 2;
        num_eq = 0;
    }

    void objective(const Vector& x, Scalar& cst) {
        // printf("objective\n");
        cst = -x(0) - x(1);
    }

    void objective_linearized(const Vector& x, Vector& grad, Scalar& cst) {
        // printf("objective_linearized\n");
        grad.resize(2);

        objective(x, cst);
        grad << -1, -1;
    }

    void constraint(const Vector& x, Vector& eq, Vector& ineq, Vector& lb, Vector& ub) {
        // printf("constraint\n");
        ineq.resize(2);
        eq.resize(0);
        lb.resize(2);
        ub.resize(2);

        ineq << 1 - x.squaredNorm(), x.squaredNorm() - 2;  // 1 <= x0^2 + x1^2 <= 2
        lb << 0, 0;                                        // x0 > 0 and x1 > 0
        ub << infinity, infinity;
    }

    void constraint_linearized(const Vector& x, Matrix& A_eq, Vector& eq, Matrix& A_ineq,
                               Vector& ineq, Vector& lb, Vector& ub) {
        // printf("constraint_linearized\n");
        A_ineq.resize(2, 2);
        A_eq.resize(0, 2);

        constraint(x, eq, ineq, lb, ub);
        A_ineq << -2 * x.transpose(), 2 * x.transpose();
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

template <typename Scalar_, typename _Derived>
struct NonLinearProblemAutoDiff : public NonLinearProblem<Scalar_> {
    using Scalar = Scalar_;
    using Vector = NonLinearProblem<double>::Vector;
    using Matrix = NonLinearProblem<double>::Matrix;

    using ADScalar = Eigen::AutoDiffScalar<Vector>;
    using ADVector = Eigen::Matrix<ADScalar, Eigen::Dynamic, 1>;

    template <typename vec>
    void ADVectorSeed(vec& x) {
        for (int i = 0; i < x.rows(); i++) {
            x[i].derivatives() = Vector::Unit(this->num_var, i);
        }
    }

    void objective(const Vector& x, Scalar& obj) override {
        static_cast<_Derived*>(this)->objective(x, obj);
    }

    void objective_linearized(const Vector& x, Vector& grad, Scalar& obj) override {
        ADVector ad_x = x;
        ADScalar ad_obj;
        ADVectorSeed(ad_x);
        /* Static polymorphism using CRTP */
        static_cast<_Derived*>(this)->objective(ad_x, ad_obj);
        obj = ad_obj.value();
        grad = ad_obj.derivatives();
    }

    void constraint(const Vector& x, Vector& eq, Vector& ineq, Vector& lb, Vector& ub) override {
        static_cast<_Derived*>(this)->constraint(x, eq, ineq, lb, ub);
    }

    void constraint_linearized(const Vector& x, Matrix& J_eq, Vector& eq, Matrix& J_ineq,
                               Vector& ineq, Vector& lb, Vector& ub) override {
        ADVector ad_eq(this->num_eq);
        ADVector ad_ineq(this->num_ineq);

        ADVector ad_x = x;
        ADVectorSeed(ad_x);
        static_cast<_Derived*>(this)->constraint(ad_x, ad_eq, ad_ineq, lb, ub);

        // fill equality constraint Jacobian
        for (int i = 0; i < ad_eq.rows(); i++) {
            eq[i] = ad_eq[i].value();
            Eigen::Ref<Vector> deriv = ad_eq[i].derivatives();
            J_eq.row(i) = deriv.transpose();
        }

        // fill inequality constraint Jacobian
        for (int i = 0; i < ad_ineq.rows(); i++) {
            ineq[i] = ad_ineq[i].value();
            Eigen::Ref<Vector> deriv = ad_ineq[i].derivatives();
            J_ineq.row(i) = deriv.transpose();
        }
    }
};

struct ConstrainedRosenbrock : public NonLinearProblemAutoDiff<double, ConstrainedRosenbrock> {
    const Scalar a = 1;
    const Scalar b = 100;
    Eigen::Matrix<Scalar, 2, 1> SOLUTION = {0.7071067812, 0.707106781};

    ConstrainedRosenbrock() {
        num_var = 2;
        num_ineq = 1;
        num_eq = 1;
    }

    template <typename DerivedA, typename DerivedB>
    void objective(const DerivedA& x, DerivedB& cst) {
        // printf("objective\n");
        // std::cout << "x " << x.transpose() << std::endl;

        // (a-x)^2 + b*(y-x^2)^2
        cst = pow(a - x(0), 2) + b * pow(x(1) - pow(x(0), 2), 2);
    }

    template <typename A, typename B, typename C>
    void constraint(const A& x, B& eq, C& ineq, Vector& lb, Vector& ub) {
        // printf("constraint\n");
        // std::cout << "x " << x.transpose() << std::endl;
        // std::cout << "eq " << eq.transpose() << std::endl;
        // std::cout << "ineq " << ineq.transpose() << std::endl;
        // std::cout << "lb " << lb.transpose() << std::endl;
        // std::cout << "ub " << ub.transpose() << std::endl;

        ineq.resize(num_ineq);
        eq.resize(num_eq);
        lb.resize(num_var);
        ub.resize(num_var);

        // y >= x
        ineq << x(0) - x(1);
        // x^2 + y^2 == 1
        eq << x.squaredNorm() - 1;

        const Scalar infinity = std::numeric_limits<Scalar>::infinity();
        lb << -infinity, -infinity;
        ub << infinity, infinity;

        // lb << 0, 0;
        // ub << 0.5, 1;
    }
};

TEST(SQPTestCase, TestConstrainedRosenbrock) {
    ConstrainedRosenbrock problem;
    SQP<double> solver;
    Eigen::Vector2d x0, x;
    Eigen::Vector4d y0;
    y0.setZero();

    x0 << 0, 0;
    solver.settings().max_iter = 1000;
    solver.settings().line_search_max_iter = 10;
    // solver.settings().eta = 0.5;
    solver.solve(problem, x0, y0);

    x = solver.primal_solution();

    std::cout << "iter " << solver.info().iter << std::endl;
    std::cout << "Solution " << x.transpose() << std::endl;

    EXPECT_TRUE(x.isApprox(problem.SOLUTION, 1e-2));
    EXPECT_LT(solver.info().iter, solver.settings().max_iter);
}

}  // namespace sqp_test