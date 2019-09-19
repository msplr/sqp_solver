#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "gtest/gtest.h"
#include "solvers/bfgs.hpp"

namespace bfgs_test {

template <typename Mat>
bool is_posdef(Mat H)
{
    Eigen::EigenSolver<Mat> eigensolver(H);
    for (int i = 0; i < eigensolver.eigenvalues().rows(); i++) {
        double v = eigensolver.eigenvalues()(i).real();
        if (v <= 0) {
            return false;
        }
    }
    return true;
}

TEST(BFGSTestCase, Test2D_posdef) {
    using Scalar = double;
    using Mat = Eigen::Matrix<Scalar, 2, 2>;
    using Vec = Eigen::Matrix<Scalar, 2, 1>;

    Vec step, delta_grad;
    Mat H; // true constant hessian;
    H << 2, 0,
         0, 1;
    Mat B = Mat::Identity();

    for (int i = 0; i < 10; i++) {
        // do some random steps
        step = {sin(i), cos(i)};

        delta_grad = H*step;
        BFGS_update(B, step, delta_grad);

        EXPECT_TRUE(is_posdef(B));
    }

    std::cout << "B\n" << B << std::endl;
    EXPECT_TRUE(B.isApprox(H, 1e-3));
}

TEST(BFGSTestCase, Test2D_indefinite) {
    using Scalar = double;
    using Mat = Eigen::Matrix<Scalar, 2, 2>;
    using Vec = Eigen::Matrix<Scalar, 2, 1>;

    Vec step, delta_grad;
    Mat H; // true constant hessian;
    H << 2, 0,
         0, -1;
    Mat B = Mat::Identity();

    for (int i = 0; i < 10; i++) {
        // do some random steps
        step = {sin(i), cos(i)};

        delta_grad = H*step;
        BFGS_update(B, step, delta_grad);

        EXPECT_TRUE(is_posdef(B));
    }
    std::cout << "B\n" << B << std::endl;
}

#if 0 // suspended for now, see issue #13
TEST(BFGSTestCase, TestSmallStep) {
    using Scalar = float;
    using Mat = Eigen::Matrix<Scalar, 2, 2>;
    using Vec = Eigen::Matrix<Scalar, 2, 1>;

    Vec step, y;
    Mat B;

    B << 418.112, 1213, 1213, 3522.27;
    EXPECT_TRUE(is_posdef(B));
    step << -1.2659e-06, 1.25816e-06;
    y << -0.00963563, -0.00957048;
    BFGS_update(B, step, y);
    EXPECT_TRUE(is_posdef(B));

    std::cout << "B\n" << B << std::endl;
}
#endif

} // namespace bfgs_test
