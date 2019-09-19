#ifndef BFGS_HPP
#define BFGS_HPP

#include <Eigen/Dense>
#include <limits>

/** Damped BFGS update
 * Implements "Procedure 18.2 Damped BFGS updating for SQP" form Numerical Optimization by Nocedal.
 *
 * @param[in,out]   B hessian matrix, is updated by this function
 * @param[in]       s step vector (x - x_prev)
 * @param[in]       y gradient change (grad - grad_prev)
 */
template <typename Mat, typename Vec>
void BFGS_update(Mat& B, const Vec& s, const Vec& y)
{
    using Scalar = typename Mat::Scalar;
    Scalar sy, sr, sBs;
    Vec Bs, r;

    Bs.noalias() = B * s;
    sBs = s.dot(Bs);
    sy = s.dot(y);

    if (sy < 0.2 * sBs) {
        // damped update to enforce positive definite B
        Scalar theta;
        theta = 0.8 * sBs / (sBs - sy);
        r.noalias() = theta * y + (1 - theta) * Bs;
        sr = theta * sy + (1 - theta) * sBs;
    } else {
        // unmodified BFGS
        r = y;
        sr = sy;
    }

    if (sr < std::numeric_limits<Scalar>::epsilon()) {
        return;
    }

    B.noalias() += -Bs * Bs.transpose() / sBs + r * r.transpose() / sr;
}

#endif /* BFGS_HPP */
