#include <cmath>
#include <iostream>
#include <solvers/qp.hpp>

#ifndef SOLVER_ASSERT
#define SOLVER_ASSERT(x) eigen_assert(x)
#endif

namespace qp_solver {

template <typename T>
void QPSolver<T>::setup(const QP &qp) {
    n = qp.P->rows();
    m = qp.A->rows();

    x = Vector::Zero(n);
    z = Vector::Zero(m);
    y = Vector::Zero(m);

    x_tilde.resize(n);
    z_tilde.resize(m);
    z_prev.resize(m);
    rho_vec.resize(m);
    rho_inv_vec.resize(m);
    kkt_mat.resize(n + m, n + m);
    rhs.resize(n + m);
    x_tilde_nu.resize(n + m);
    constr_type.resize(m);

    // Set QP constraint type
    constr_type_init(*qp.l, *qp.u, constr_type);

    // initialize step size (rho) vector
    rho_vec_update(settings_.rho);

    // construct KKT system and compute decomposition
    construct_KKT_mat(qp);
    compute_KKT();

    info_.status = UNSOLVED;
}

template <typename T>
void QPSolver<T>::update_qp(const QP &qp) {
    // Set QP constraint type
    constr_type_init(*qp.l, *qp.u, constr_type);

    // initialize step size (rho) vector
    rho_vec_update(settings_.rho);

    // update KKT system and do factorization
    update_KKT_mat(qp);
    factorize_KKT();

    info_.status = UNSOLVED;
}

template <typename T>
void QPSolver<T>::solve(const QP &qp) {
    bool check_termination = false;

    if (info_.status == UNINITIALIZED) {
        SOLVER_ASSERT(info_.status == UNINITIALIZED);
        return;
    }

#ifdef QP_SOLVER_PRINTING
    if (settings_.verbose) {
        settings_.print();
    }
#endif
    if (!settings_.warm_start) {
        x.Zero(n);
        z.Zero(m);
        y.Zero(m);
    }

    for (iter = 1; iter <= settings_.max_iter; iter++) {
        const Scalar alpha = settings_.alpha;
        z_prev = z;

        // update x_tilde z_tilde
        form_KKT_rhs(qp, rhs);
        x_tilde_nu = linear_solver.solve(rhs);

        x_tilde = x_tilde_nu.head(n);
        z_tilde = z_prev + rho_inv_vec.cwiseProduct(x_tilde_nu.tail(m) - y);

        // update x
        x = alpha * x_tilde + (1 - alpha) * x;

        // update z
        z = alpha * z_tilde + (1 - alpha) * z_prev + rho_inv_vec.cwiseProduct(y);
        box_projection(z, *qp.l, *qp.u);  // euclidean projection

        // update y
        y = y + rho_vec.cwiseProduct(alpha * z_tilde + (1 - alpha) * z_prev - z);

        if (settings_.check_termination != 0 && iter % settings_.check_termination == 0) {
            check_termination = true;
        } else {
            check_termination = false;
        }

        if (check_termination) {
            update_state(qp);

#ifdef QP_SOLVER_PRINTING
            if (settings_.verbose) {
                print_status(qp);
            }
#endif
            if (termination_criteria(qp)) {
                info_.status = SOLVED;
                break;
            }
        }

        if (settings_.adaptive_rho && iter % settings_.adaptive_rho_interval == 0) {
            if (!check_termination) {
                // state was not yet updated
                update_state(qp);
            }
            Scalar new_rho = rho_estimate(rho, qp);
            new_rho = fmax(RHO_MIN, fmin(new_rho, RHO_MAX));
            info_.rho_estimate = new_rho;

            if (new_rho < rho / settings_.adaptive_rho_tolerance ||
                new_rho > rho * settings_.adaptive_rho_tolerance) {
                rho_vec_update(new_rho);
                update_KKT_rho();
                /* Note: KKT Sparsity pattern unchanged by rho update. Only factorize. */
                factorize_KKT();
            }
        }
    }

    if (iter > settings_.max_iter) {
        info_.status = MAX_ITER_EXCEEDED;
    }
    info_.iter = iter;

#ifdef QP_SOLVER_PRINTING
    if (settings_.verbose) {
        info_.print();
    }
#endif
}

template <typename T>
void QPSolver<T>::construct_KKT_mat(const QP &qp) {
#ifdef QP_SOLVER_USE_SPARSE
    kkt_mat.resize(n + m, n + m);  // sets all elements to 0
    Eigen::Matrix<int, Eigen::Dynamic, 1> nnz_col(n + m);

    nnz_col.setConstant(1);
    nnz_col.head(n) += *qp.P_col_nnz + *qp.A_col_nnz;
    kkt_mat.reserve(nnz_col);

    // top left:  P + sigma*I
    sparse_insert_at(kkt_mat, 0, 0, *qp.P);
    for (int i = 0; i < n; i++) {
        kkt_mat.coeffRef(i, i) += settings_.sigma;
    }

    // bottom left:  A
    sparse_insert_at(kkt_mat, n, 0, *qp.A);

    // bottom right:  -1/rho.*I
    for (int i = 0; i < m; i++) {
        kkt_mat.insert(n + i, n + i) = -rho_inv_vec(i);
    }

    kkt_mat.makeCompressed();
#else
    kkt_mat.topLeftCorner(n, n) = *qp.P + settings_.sigma * Matrix::Identity(n, n);
    kkt_mat.bottomLeftCorner(m, n) = *qp.A;
    kkt_mat.bottomRightCorner(m, m) = -1.0 * rho_inv_vec.asDiagonal();
#endif
}

template <typename T>
void QPSolver<T>::update_KKT_mat(const QP &qp) {
#ifdef QP_SOLVER_USE_SPARSE
    // construct_KKT_mat(qp);
    // TODO: more efficient update?
    for (int k = 0; k < kkt_mat.outerSize(); ++k) {
        for (typename Matrix::InnerIterator it(kkt_mat, k); it; ++it) {
            int row, col;
            row = it.row();
            col = it.col();
            if (row < n) {
                if (col < n) {
                    // top left:  P + sigma*I
                    it.valueRef() = qp.P->coeff(row, col) + settings_.sigma;
                } else {
                    // top right:  A'
                    it.valueRef() = qp.A->coeff(col, row - n);
                }
            } else {
                if (col < n) {
                    // bottom left:  A
                    it.valueRef() = qp.A->coeff(row - n, col);
                } else {
                    // bottom right:  -1/rho.*I
                    it.valueRef() = -rho_inv_vec(row - n);
                }
            }
        }
    }
#else
    construct_KKT_mat(qp);
#endif
}

template <typename T>
void QPSolver<T>::update_KKT_rho() {
#ifdef QP_SOLVER_USE_SPARSE
    // Note: optimize by writing kkt_mat.valuePtr(). needs to be compressed and Eigen::Lower.
    for (int i = 0; i < m; i++) {
        kkt_mat.coeffRef(n + i, n + i) = -rho_inv_vec(i);
    }
#else
    kkt_mat.bottomRightCorner(m, m) = -1.0 * rho_inv_vec.asDiagonal();
#endif
}

template <typename T>
void QPSolver<T>::factorize_KKT() {
#ifdef QP_SOLVER_USE_SPARSE
    linear_solver.factorize(kkt_mat);
#else
    linear_solver.compute(kkt_mat);
#endif
    SOLVER_ASSERT(linear_solver.info() == Eigen::Success);
}

template <typename T>
void QPSolver<T>::compute_KKT() {
    linear_solver.compute(kkt_mat);
    SOLVER_ASSERT(linear_solver.info() == Eigen::Success);
}

#ifdef QP_SOLVER_USE_SPARSE
template <typename T>
void QPSolver<T>::sparse_insert_at(Matrix &dst, int row, int col, const Matrix &src) const {
    for (int k = 0; k < src.outerSize(); ++k) {
        for (typename Matrix::InnerIterator it(src, k); it; ++it) {
            dst.insert(row + it.row(), col + it.col()) = it.value();
        }
    }
}
#endif

template <typename T>
void QPSolver<T>::form_KKT_rhs(const QP &qp, Vector &rhs) {
    rhs.head(n) = settings_.sigma * x - *qp.q;
    rhs.tail(m) = z - rho_inv_vec.cwiseProduct(y);
}

template <typename T>
void QPSolver<T>::box_projection(Vector &z, const Vector &l, const Vector &u) {
    z = z.cwiseMax(l).cwiseMin(u);
}

template <typename T>
void QPSolver<T>::constr_type_init(const Vector& l, const Vector& u, Eigen::VectorXi &constr_type) {
    for (int i = 0; i < l.rows(); i++) {
        if (l[i] < -LOOSE_BOUNDS_THRESH && u[i] > LOOSE_BOUNDS_THRESH) {
            constr_type[i] = LOOSE_BOUNDS;
        } else if (u[i] - l[i] < RHO_TOL) {
            constr_type[i] = EQUALITY_CONSTRAINT;
        } else {
            constr_type[i] = INEQUALITY_CONSTRAINT;
        }
    }
}

template <typename T>
void QPSolver<T>::rho_vec_update(Scalar rho0) {
    for (int i = 0; i < rho_vec.rows(); i++) {
        switch (constr_type[i]) {
            case LOOSE_BOUNDS:
                rho_vec[i] = RHO_MIN;
                break;
            case EQUALITY_CONSTRAINT:
                rho_vec[i] = RHO_EQ_FACTOR * rho0;
                break;
            case INEQUALITY_CONSTRAINT: /* fall through */
            default:
                rho_vec[i] = rho0;
        };
    }
    rho_inv_vec = rho_vec.cwiseInverse();
    rho = rho0;
    info_.rho_updates += 1;
}

template <typename T>
void QPSolver<T>::update_state(const QP &qp) {
    Scalar norm_Ax, norm_z;
    norm_Ax = (*qp.A * x).template lpNorm<Eigen::Infinity>();
    norm_z = z.template lpNorm<Eigen::Infinity>();
    max_Ax_z_norm_ = fmax(norm_Ax, norm_z);

    Scalar norm_Px, norm_ATy, norm_q;
    norm_Px = (*qp.P * x).template lpNorm<Eigen::Infinity>();
    norm_ATy = (qp.A->transpose() * y).template lpNorm<Eigen::Infinity>();
    norm_q = qp.q->template lpNorm<Eigen::Infinity>();
    max_Px_ATy_q_norm_ = fmax(norm_Px, fmax(norm_ATy, norm_q));

    info_.res_prim = residual_prim(qp);
    info_.res_dual = residual_dual(qp);
}

template <typename T>
typename QPSolver<T>::Scalar QPSolver<T>::rho_estimate(const Scalar rho0, const QP &qp) const {
    Scalar rp_norm, rd_norm;
    rp_norm = info_.res_prim / (max_Ax_z_norm_ + DIV_BY_ZERO_REGUL);
    rd_norm = info_.res_dual / (max_Px_ATy_q_norm_ + DIV_BY_ZERO_REGUL);

    Scalar rho_new = rho0 * sqrt(rp_norm / (rd_norm + DIV_BY_ZERO_REGUL));
    return rho_new;
}

template <typename T>
typename QPSolver<T>::Scalar QPSolver<T>::eps_prim(const QP &qp) const {
    return settings_.eps_abs + settings_.eps_rel * max_Ax_z_norm_;
}

template <typename T>
typename QPSolver<T>::Scalar QPSolver<T>::eps_dual(const QP &qp) const {
    return settings_.eps_abs + settings_.eps_rel * max_Px_ATy_q_norm_;
}

template <typename T>
typename QPSolver<T>::Scalar QPSolver<T>::residual_prim(const QP &qp) const {
    return (*qp.A * x - z).template lpNorm<Eigen::Infinity>();
}

template <typename T>
typename QPSolver<T>::Scalar QPSolver<T>::residual_dual(const QP &qp) const {
    return (*qp.P * x + *qp.q + qp.A->transpose() * y).template lpNorm<Eigen::Infinity>();
}

template <typename T>
bool QPSolver<T>::termination_criteria(const QP &qp) {
    // check residual norms to detect optimality
    if (info_.res_prim <= eps_prim(qp) && info_.res_dual <= eps_dual(qp)) {
        return true;
    }

    return false;
}

#ifdef QP_SOLVER_PRINTING
template <typename T>
void QPSolver<T>::print_status(const QP &qp) const {
    Scalar obj = 0.5 * x.dot(*qp.P * x) + *qp.q.dot(x);

    if (iter == settings_.check_termination) {
        printf("iter   obj       rp        rd\n");
    }
    printf("%4d  %.2e  %.2e  %.2e\n", iter, obj, info_.res_prim, info_.res_dual);
}
#endif

template class QPSolver<double>;
template class QPSolver<float>;

}  // namespace qp_solver
