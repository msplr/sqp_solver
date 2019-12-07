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
        num_constr = 3;
    }

    void objective(const Vector& x, Scalar& obj) { obj = -x(0) - x(1); }

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
    Eigen::Vector3d y0;
    y0.setZero();

    solver.settings().max_iter = 100;
    solver.settings().line_search_max_iter = 10;  // TODO(mi): doesn't work for higher values
    solver.solve(problem, x0, y0);
    x = solver.primal_solution();

    // std::cout << "iter " << solver.info().iter << std::endl;
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

template <typename SCALAR, typename DERIVED>
struct NonLinearProblemAutoDiff : public NonLinearProblem<SCALAR> {
    using Scalar = SCALAR;
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
        static_cast<DERIVED*>(this)->objective(x, obj);
    }

    void objective_linearized(const Vector& x, Vector& grad, Scalar& obj) override {
        ADVector ad_x = x;
        ADScalar ad_obj;
        ADVectorSeed(ad_x);
        /* Static polymorphism using CRTP */
        static_cast<DERIVED*>(this)->objective(ad_x, ad_obj);
        obj = ad_obj.value();
        grad = ad_obj.derivatives();
    }

    void constraint(const Vector& x, Vector& c, Vector& l, Vector& u) override {
        static_cast<DERIVED*>(this)->constraint(x, c, l, u);
    }

    void constraint_linearized(const Vector& x, Matrix& Jc, Vector& c, Vector& l,
                               Vector& u) override {
        ADVector ad_c(this->num_constr);
        ADVector ad_x = x;

        ADVectorSeed(ad_x);
        static_cast<DERIVED*>(this)->constraint(ad_x, ad_c, l, u);

        // fill constraint Jacobian
        for (int i = 0; i < ad_c.rows(); i++) {
            c[i] = ad_c[i].value();
            Eigen::Ref<Vector> deriv = ad_c[i].derivatives();
            Jc.row(i) = deriv.transpose();
        }
    }
};

struct ConstrainedRosenbrock : public NonLinearProblemAutoDiff<double, ConstrainedRosenbrock> {
    const Scalar a = 1;
    const Scalar b = 100;
    Eigen::Matrix<Scalar, 2, 1> SOLUTION = {0.707106781, 0.707106781};

    ConstrainedRosenbrock() {
        num_var = 2;
        num_constr = 2;
    }

    template <typename DerivedA, typename DerivedB>
    void objective(const DerivedA& x, DerivedB& cst) {
        // (a-x)^2 + b*(y-x^2)^2
        cst = pow(a - x(0), 2) + b * pow(x(1) - pow(x(0), 2), 2);
    }

    template <typename A, typename B>
    void constraint(const A& x, B& c, Vector& l, Vector& u) {
        const Scalar infinity = std::numeric_limits<Scalar>::infinity();
        // y >= x
        // x^2 + y^2 == 1
        c << x(0) - x(1), x.squaredNorm();
        u << 0, 1;
        l << -infinity, 1;
    }
};

TEST(SQPTestCase, TestConstrainedRosenbrock) {
    ConstrainedRosenbrock problem;
    SQP<double> solver;
    Eigen::Vector2d x0, x;
    Eigen::Vector2d y0;
    y0.setZero();

    x0 << 0, 0;
    solver.settings().max_iter = 1000;
    solver.settings().line_search_max_iter = 10;
    // solver.settings().eta = 0.5;
    solver.solve(problem, x0, y0);

    x = solver.primal_solution();

    // std::cout << "iter " << solver.info().iter << std::endl;
    // std::cout << "primal solution " << x.transpose() << std::endl;
    // std::cout << "dual solution " << solver.dual_solution().transpose() << std::endl;

    EXPECT_TRUE(x.isApprox(problem.SOLUTION, 1e-2));
    EXPECT_LT(solver.info().iter, solver.settings().max_iter);
}

}  // namespace sqp_test