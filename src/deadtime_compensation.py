import numpy as np
from scipy.signal import correlate, savgol_filter
from scipy.stats import pearsonr
from typing import Tuple, Optional


class DeadTimeCompensator:
    """
    A class for estimating and compensating dead time between concentration 
    and conductance signals using cross-correlation analysis.
    """
    
    def __init__(self):
        """
        Initialize the DeadTimeCompensator.
        """
        self.dead_time = None
        self.original_correlation = None
        self.optimized_correlation = None
        self._is_fitted = False
    
    def fit(self, time_s: np.ndarray, conductance: np.ndarray, 
            concentration: np.ndarray) -> 'DeadTimeCompensator':
        """
        Estimate the dead time by maximizing cross-correlation between
        concentration and filtered derivative of conductance.
        
        Args:
            time_s (np.ndarray): Time vector in seconds.
            conductance (np.ndarray): Conductance signal G(t).
            concentration (np.ndarray): Concentration signal C(t).
            
        Returns:
            self: Returns the fitted compensator instance.
        """
        # Validate inputs
        if len(time_s) != len(conductance) or len(time_s) != len(concentration):
            raise ValueError("All input arrays must have the same length")
        
        self.time_s = time_s.copy()
        self.conductance = conductance.copy()
        self.concentration = concentration.copy()
        
        # Differentiate and filter the conductance
        dt = np.mean(np.diff(time_s))
        conductance_derivative = np.gradient(self.conductance, dt)
        conductance_derivative = savgol_filter(conductance_derivative, 100, 5)
        
        # Calculate original correlation (no time shift)
        self.original_correlation, _ = pearsonr(conductance_derivative, concentration)
        
        # Z-score normalization
        concentration_norm = (concentration - np.mean(concentration)) / np.std(concentration)
        derivative_norm = (conductance_derivative - np.mean(conductance_derivative)) / np.std(conductance_derivative)
        
        # Perform cross-correlation
        correlation = correlate(derivative_norm, concentration_norm, mode='full', method='fft')
        lags = np.arange(-len(concentration) + 1, len(conductance_derivative))
        
        # Find optimal deadtime (diffusion delay)
        lag_points = lags[np.argmax(correlation)]
        self.dead_time = lag_points * dt
        
        # Calculate optimized correlation with time shift
        shifted_derivative = np.interp(time_s, time_s - self.dead_time, 
                                     conductance_derivative, left=np.nan, right=np.nan)
        valid_mask = ~np.isnan(shifted_derivative)
        self.optimized_correlation, _ = pearsonr(shifted_derivative[valid_mask], 
                                               concentration[valid_mask])
        
        self._is_fitted = True
        return self
    
    def transform(self, time_s: Optional[np.ndarray] = None, 
              conductance: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Apply dead time compensation by cutting the signals to align them.
        
        Args:
            time_s (np.ndarray, optional): Time vector. If None, uses the time from fit().
            conductance (np.ndarray, optional): Conductance signal. If None, uses the fitted conductance.
            
        Returns:
            tuple: (aligned_time, aligned_conductance, aligned_concentration, dead_time)
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before transform()")
        
        if time_s is None:
            time_s = self.time_s
        if conductance is None:
            conductance = self.conductance

        dt = np.mean(np.diff(time_s))
        lag_points = int(round(self.dead_time / dt))
        
        self.lag_points = abs(lag_points)
        aligned_time = time_s[:-lag_points]
        aligned_conductance = conductance[lag_points:]
        aligned_concentration = self.concentration[:-lag_points]

        return aligned_time, aligned_conductance, aligned_concentration

    
    def fit_transform(self, time_s: np.ndarray, conductance: np.ndarray, 
                 concentration: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.fit(time_s, conductance, concentration).transform()

    
    def get_correlation_improvement(self) -> Tuple[float, float]:
        """
        Get the correlation values before and after dead time compensation.
        
        Returns:
            tuple: (original_correlation, optimized_correlation)
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before accessing correlation values")
        return self.original_correlation, self.optimized_correlation
    
    def get_dead_time(self) -> float:
        """
        Get the estimated dead time.
        
        Returns:
            float: Dead time in seconds
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before accessing dead time")
        return self.dead_time
    
    def cut_data_tail(self, data: np.ndarray) -> np.ndarray:
        return data[:-self.lag_points]
    
    def cut_data_head(self, data: np.ndarray) -> np.ndarray:
        return data[self.lag_points:]
