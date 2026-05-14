import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
new_rc_params = {"text.usetex": False, "svg.fonttype": "none"}
mpl.rcParams.update(new_rc_params)

def metrics(y_true, y_pred):
    """
    Calculates and prints various regression metrics between two signals.

    This function computes and prints the Mean Squared Error (MSE),
    Root Mean Squared Error (RMSE), Normalized Root Mean Squared Error (NRMSE),
    and the Coefficient of Determination (R-squared).

    It prints results directly to the console.

    Args:
        y_true (np.ndarray): The ground truth or original signal.
        y_pred (np.ndarray): The predicted signal.
    """
    # Mean Squared Error (MSE)
    mse = np.mean((y_true - y_pred) ** 2)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # Normalized Root Mean Squared Error (NRMSE)
    y_true_range = np.max(y_true) - np.min(y_true)
    nrmse = rmse / y_true_range

    # R-squared (R²), Coefficient of Determination
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # # --- Print Results ---
    # print("--- ERROR METRICS ---")
    # print(f"MSE:    {mse:.4f}")
    # print(f"RMSE:   {rmse:.4f}")
    # print(f"NRMSE:  {nrmse:.4f}")
    # print(f"R^2:    {r2:.4f}")

    return mse, rmse, nrmse, r2


def plot_comparision(
    x_data, 
    y_gt, 
    y_estim, 
    save_path=None
):
    """
    Generates and optionally saves a comparison plot in the 'Nature' style.
    The x-axis is always labeled 'Time (s)'.

    Args:
        x_data (array-like): The data for the x-axis (time).
        y_gt (array-like): The ground truth data for the y-axis.
        y_estim (array-like): The estimated data for the y-axis.
        save_path (str, optional): The file path to save the figure. 
                                 If None, the plot is only displayed. 
                                 Supported formats include .png, .pdf, .svg.
                                 Defaults to None.
    """
    
    # Ensure inputs are NumPy arrays for consistent filtering
    x_data = np.asarray(x_data)
    y_gt = np.asarray(y_gt)
    y_estim = np.asarray(y_estim)

    # Filter data to a specific time range (e.g., first 50 seconds)
    # mask = x_data <= 25
    # x_data = x_data[mask]
    # y_gt = y_gt[mask]
    # y_estim = y_estim[mask]

    # Use a specific style context for the plot
    # style_context = ['science', 'nature', {'text.usetex': False, 'font.size': 14}]
    with plt.style.context(style_context):
        # Create a figure and a single subplot
        fig, ax = plt.subplots(figsize=(12, 4)) # Typical size for a single-column figure

        # Plot the ground truth and estimated data
        ax.plot(x_data, y_gt, color='black', label='Ground Truth')
        ax.plot(x_data, y_estim, color='red', label='Estimation')

        # Set the labels with appropriate font sizes
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Relative concentration")

        # Add a legend to distinguish the lines
        ax.legend()

        # Add major and minor grid lines for better readability
        ax.grid(which='major', alpha=0.6, linestyle='--')
        ax.grid(which='minor', alpha=0.3, linestyle=':')
        ax.minorticks_on() # Required to show minor ticks and grid
        
        # Automatically adjust subplot params for a tight layout
        plt.tight_layout()

        # Save the figure if a path is provided
        if save_path:
            # bbox_inches='tight' removes excess white space around the figure
            # DPI is set to 300 for publication quality.
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        # Display the plot
        plt.show()