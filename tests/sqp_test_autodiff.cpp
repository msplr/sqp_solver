#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>
#include <solvers/sqp.hpp>

using namespace sqp;

namespace sqp_test_autodiff {

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

template <typename Scalar>
inline Scalar rosenbrock(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> x) {
    const double a = 1;
    const double b = 100;

    Scalar z = 0;
    for (int i = 0; i < x.rows() - 1; i++) {
        z += pow(a - x[i], 2) + b * pow(x[i + 1] - pow(x[i], 2), 2);
    }
    return z;
}

struct ConstrainedRosenbrock2D : public NonLinearProblemAutoDiff<double, ConstrainedRosenbrock2D> {
    const Scalar a = 1;
    const Scalar b = 100;
    Eigen::Matrix<Scalar, 2, 1> SOLUTION = {0.707106781, 0.707106781};

    ConstrainedRosenbrock2D() {
        num_var = 2;
        num_constr = 2;
    }

    template <typename DerivedA, typename DerivedB>
    void objective(const DerivedA& x, DerivedB& obj) {
        // (a-x)^2 + b*(y-x^2)^2
        // obj = pow(a - x(0), 2) + b * pow(x(1) - pow(x(0), 2), 2);
        obj = rosenbrock(x);
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

TEST(SQPAutoDiff, TestConstrainedRosenbrock2D) {
    ConstrainedRosenbrock2D problem;
    SQP<double> solver;
    Eigen::VectorXd x0 = Eigen::Vector2d(0,0);
    Eigen::VectorXd y0 = Eigen::VectorXd::Zero(2);

    solver.settings().max_iter = 100;
    // solver.settings().line_search_max_iter = 10;
    solver.settings().line_search_max_iter = 100;

    // solver.settings().eta = 0.5;
    solver.solve(problem, x0, y0);

    if (!solver.primal_solution().isApprox(problem.SOLUTION, 1e-2)) {
        solver.info().print();
        std::cout << "primal solution " << solver.primal_solution().transpose() << std::endl;
        std::cout << "dual solution " << solver.dual_solution().transpose() << std::endl;
    }

    EXPECT_TRUE(solver.primal_solution().isApprox(problem.SOLUTION, 1e-2));
    EXPECT_LT(solver.info().iter, solver.settings().max_iter);
}

struct Rosenbrock : public NonLinearProblemAutoDiff<double, Rosenbrock> {
    const Scalar a = 1;
    const Scalar b = 100;
    Vector SOLUTION;

    Rosenbrock(int n) {
        num_var = n;
        num_constr = n;
        SOLUTION = Vector::Ones(n);
        // num_constr = 0;
        // num_constr = 1;
    }

    template <typename DerivedA, typename DerivedB>
    void objective(const DerivedA& x, DerivedB& obj) {
        obj = rosenbrock(x);
    }

    template <typename A, typename B>
    void constraint(const A& x, B& c, Vector& l, Vector& u) {
        c = x;
        // const Scalar infinity = std::numeric_limits<Scalar>::infinity();
        // u.setConstant(infinity);
        u.setConstant(1e+17);
        l.setConstant(0.5);
    }
};

TEST(SQPAutoDiff, TestRosenbrock) {
    Rosenbrock problem(2);
    SQP<double> solver;

    solver.settings().max_iter = 100;
    solver.settings().line_search_max_iter = 10; // causes numerical issues
    // solver.settings().line_search_max_iter = 100;
    solver.solve(problem);

    if (!solver.primal_solution().isApprox(problem.SOLUTION, 1e-2)) {
        solver.info().print();
        std::cout << "primal solution " << solver.primal_solution().transpose() << std::endl;
        std::cout << "dual solution " << solver.dual_solution().transpose() << std::endl;
    }

    EXPECT_TRUE(solver.primal_solution().isApprox(problem.SOLUTION, 1e-2));
    EXPECT_LT(solver.info().iter, solver.settings().max_iter);
}

template <typename T>
void callback(SQP<T>& solver) {
    Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, ", ", ",", "[", "],");
    std::cout << solver.x_.transpose().format(fmt) << std::endl;
}

struct SimpleNLP : public NonLinearProblemAutoDiff<double, SimpleNLP> {
    Eigen::Vector2d SOLUTION = {1, 1};
    const Scalar infinity = std::numeric_limits<Scalar>::infinity();

    SimpleNLP() {
        num_var = 2;
        num_constr = 3;
    }

    template <typename A, typename B>
    void objective(const A& x, B& obj) {
        obj = -x.sum();
    }

    template <typename A, typename B>
    void constraint(const A& x, B& c, Vector& l, Vector& u) {
        // 1 <= x0^2 + x1^2 <= 2
        // 0 <= x0
        // 0 <= x1
        c << x.squaredNorm(), x;
        l << 1, 0, 0;  // x0 > 0 and x1 > 0
        u << 2, infinity, infinity;
    }
};

TEST(SQPAutoDiff, TestSimpleNLP) {
    SimpleNLP problem;
    SQP<double> solver;

    // feasible initial point
    Eigen::VectorXd x0 = Eigen::Vector2d(1.2, 0.1);
    Eigen::VectorXd y0 = Eigen::VectorXd::Zero(3);

    // TODO(mi): doesn't work for higher values
    solver.settings().line_search_max_iter = 10;
    solver.settings().max_iter = 100;
    // solver.settings().iteration_callback = callback<double>;

    solver.solve(problem, x0, y0);

    // solver.info().print();
    // std::cout << "primal solution " << solver.primal_solution().transpose() << std::endl;
    // std::cout << "dual solution " << solver.dual_solution().transpose() << std::endl;

    EXPECT_TRUE(solver.primal_solution().isApprox(problem.SOLUTION, 1e-2));
    EXPECT_LT(solver.info().iter, solver.settings().max_iter);
}

}  // namespace sqp_test_autodiff
