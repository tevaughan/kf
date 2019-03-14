/// @file       kf.hpp
/// @copyright  2019 Thomas E. Vaughan
/// @brief      Definition of vnix::kf.

#ifndef VNIX_KF_HPP
#define VNIX_KF_HPP

#include <eigen3/Eigen/Core>

/// Namespace for Thomas E. Vaughan's tools.
namespace vnix {

/// Convenient short-hand.
template <unsigned T, unsigned U> using kf_mat = Eigen::Matrix<double, T, U>;

/// Kalman filter whose state has N components and whose measurement has
/// M components.
///
/// Notation corresponds roughly to
/// [http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf](Welch, 2006).
///
/// @tparam N  Number of elements in state.
/// @tparam M  Number of elements in measurement.
template <unsigned N, unsigned M> class kf {
  kf_mat<N, 1> x_; ///< Estimate of state.
  kf_mat<M, N> H_; ///< Xform from estimated state to predicted measurement.
  kf_mat<N, N> A_; ///< A-priori, one-step propagator for estimated state.
  kf_mat<N, N> P_; ///< Estimate of covariance among errors in state.

  /// Process-noise covariance.
  ///
  /// Process-noise is whatever is not accounted for by A_ and B_ in
  /// single-step advancement.
  ///
  /// Diagonal is variance of process-noise for each element of state.
  kf_mat<N, N> Q_;

  /// Measurement-noise covariance.
  ///
  /// Measurement-noise is whatever is not accounted for by H_ in prediction
  /// of measurement.
  ///
  /// Diagonal is variance of measurement-noise for each element of
  /// measurement.
  kf_mat<M, M> R_;

public:
  /// Initialize filter.
  /// @param x  Initial estimate of state.
  /// @param H  Matrix that converts estimated state to predicted measurement.
  /// @param A  Estimated state's a-priori, one-step propagation matrix.
  /// @param P  Initial estimate of covariance among errors in state.
  /// @param Q  Process-noise covariance.
  /// @param R  Measurement-noise covariance.
  kf(kf_mat<N, 1> const &x, kf_mat<M, N> const &H, kf_mat<N, N> const &A,
     kf_mat<N, N> const &P, kf_mat<N, N> const &Q, kf_mat<M, M> const &R)
      : x_(x), H_(H), A_(A), P_(P), Q_(Q), R_(R) {}

  auto const &x() const { return x_; } ///< Estimate of state.
  auto const &H() const { return H_; } ///< Xform from state to measurement.
  auto const &A() const { return A_; } ///< A-priori propagator for state.
  auto const &P() const { return P_; } ///< Estimate of state-error covariance.
  auto const &Q() const { return Q_; } ///< Process-noise covariance.
  auto const &R() const { return R_; } ///< Measurement-noise covariance.

  /// Take a step with no measurement.
  void step() {
    x_ = (A_ * x_).eval();
    P_ = (A_ * P_).eval() * A_.transposed() + Q_;
  }

  /// Take a step whose end-state corresponds to a measurement.
  /// @param z  Elements of measurement.
  void step(kf_mat<M, 1> const &z) {
    step();
    kf_mat<M, N> const HT = H_.transposed();
    auto const K = P_ * HT * (H_ * P_ * HT + R_).inverted();
    x_ += K * (z - H_ * x_).eval();
    P_ = ((kf_mat<N, N>::Identity() - K * H_) * P_).eval();
  }
};

} // namespace vnix

#endif // ndef VNIX_KF_HPP

