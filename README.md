# SQP Solver

This project implements a Sequential Quadratic Programming (SQP) Solver in C++.
The implementation follows Algorithm 18.3 from *Numerical Optimization* by J. Nocedal and S. J. Wright.
The solver class is templated to statically allocate necessary objects beforehand and allows for compile-time checks.
For solving the quadratic sub-problems the *Operator Splitting QP Solver* ([OSQP](https://github.com/oxfordcontrol/osqp)) was re-implemented in C++ using Eigen.

## Dependencies

- Eigen3 : https://eigen.tuxfamily.org/
