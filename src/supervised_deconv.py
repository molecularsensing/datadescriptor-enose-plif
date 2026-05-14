"""
Object-oriented implementation of supervised deconvolution for MOX sensors.
Reference paper: https://doi.org/10.3390/s19184029
"""

import numpy as np

class SupervisedDeconvolution:
    """
    Estimates parameters and reconstructs signals for MOX sensors using
    supervised deconvolution based on Martinez et al. (2019).
    Requires ground truth signal "u".
    """
    def __init__(self, delta_t):
        """
        Initializes the object.

        Args:
            delta_t (float): Sampling interval in seconds.
        """
        self.delta_t = delta_t
        self.eps = 1e-12
        self.y = None
        self.u = None
        self.w_hat = None # Filter coefficients [w1, w2, w3]
        self.params_hat = None # Estimated model params {tau, r, alpha}
        self._estimation_done = False

    def feed_data(self, y:np.array, u:np.array):
        """Feeds and validates measured signal (y) and ground truth (u)."""
        self.y = np.asarray(y)
        self.u = np.asarray(u)
        self._validate_inputs()
        
        # Reset when new data is fed
        self.w_hat = None
        self.params_hat = None
        self._estimation_done = False

    def estimate_params(self):
        """Estimates filter coefficients and model parameters."""
        if self._estimation_done:
            return

        self._validate_inputs()
        self._calculate_deconvolution_filter_coeff()

        if self.w_hat is not None:
            self._estimate_model_coeff()
            self._estimation_done = True
        else:
            self.params_hat = None
            self._estimation_done = False

    def retrieve_filter_coeff(self):
        """Returns estimated filter coefficients [w1, w2, w3], or None."""
        return self.w_hat

    def retrieve_model_params(self):
        """Returns estimated model parameters {'tau', 'r', 'alpha'}, or None."""
        return self.params_hat

    def reconstruct_signal(self):
        """Reconstructs the input signal u using estimated coefficients."""
        if not self._estimation_done or self.w_hat is None:
            raise ValueError("Estimation has not been run successfully. Call estimate_params() first.")
        if self.y is None:
             raise ValueError("Sensor signal 'y' is not available.")

        y_prime = np.log(np.maximum(self.y, self.eps))
        w1, w2, w3 = self.w_hat

        # Eq 9
        u_prime_hat = w1 * y_prime[1:] + w2 * y_prime[:-1] + w3
        u_hat = np.exp(u_prime_hat)
        
        # Padding
        u_hat = np.insert(u_hat, 0, u_hat[0])
        
        return u_hat

    def _calculate_deconvolution_filter_coeff(self):
        """Internal method to compute w_hat using lstsq."""
        y_prime = np.log(np.maximum(self.y, self.eps))
        u_prime = np.log(np.maximum(self.u, self.eps))

        n = len(y_prime) - 1
        # Matrix show in eq 11
        Y_prime = np.vstack((y_prime[1:], y_prime[:-1], np.ones(n))).T
        U_prime_target = u_prime[:-1]

        # Equivalent to eq 13
        solution = np.linalg.lstsq(Y_prime, U_prime_target, rcond=None)
        self.w_hat = solution[0]


    def _estimate_model_coeff(self):
        """Internal method to derive model parameters from w_hat."""
        if self.w_hat is None:
             self.params_hat = None
             return
        # * The formulas come from eq 14
        w1, w2, w3 = self.w_hat
        self.params_hat = {'tau': np.nan, 'r': np.nan, 'alpha': np.nan}

        # Derivation of tau and a
        if np.abs(w1) > self.eps:
            a_hat = -w2 / w1
            if 0 < a_hat < 1:
                try:
                    self.params_hat['tau'] = -self.delta_t / np.log(a_hat)
                except (ValueError, OverflowError):
                    self.params_hat['tau'] = np.nan

        # Derivation of r and alpha
        if np.abs(w1 + w2) > self.eps:
            try:
                r_hat = 1.0 / (w1 + w2)
                self.params_hat['r'] = r_hat
                if not np.isnan(r_hat) and np.isfinite(r_hat):
                    if abs(w3 * r_hat) < 700:
                        self.params_hat['alpha'] = np.exp(-w3 * r_hat)
                    else:
                        self.params_hat['alpha'] = np.inf
                
            except (ZeroDivisionError, ValueError, OverflowError):
                 self.params_hat['r'] = np.nan
                 self.params_hat['alpha'] = np.nan

    def _validate_inputs(self):
        """Internal validation for input signals."""
        if self.y is None or self.u is None:
            raise ValueError("Input signals y and u must be provided via feed_data().")
        if self.y.shape != self.u.shape:
            raise ValueError("Input signals y and u must have the same shape.")
        if self.y.ndim != 1 or len(self.y) < 2:
            raise ValueError("Input signals must be 1D arrays with at least 2 data points.")