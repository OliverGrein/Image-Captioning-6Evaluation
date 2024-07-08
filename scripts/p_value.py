import numpy as np
import pandas as pd
import scipy.stats as stats

from pathlib import Path


# Number of pairs in the dataset.
N_PAIRS = 5823


def kendalls_tau_p_value(tau, n) -> float:
    """
    Calculate the p-value for Kendall's Tau.

    Args:
        tau (float): Kendall's Tau value.
        n (int): Number of pairs.
    
    Returns:
        float: The p-value.
    """
    # Calculate Z statistic for Kendall's Tau (https://www.statology.org/kendalls-tau/)
    z_statistic = tau * np.sqrt(9 * n * (n - 1) / (2 * (2 * n + 5)))

    # Calculate two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))

    return p_value


if __name__ == "__main__":
    """
    If run as main, add p-values to the correlations.csv file.
    """
    correlations_path = Path(__file__).resolve().parent.parent / "results"
    df = pd.read_csv(Path(correlations_path / "correlations.csv"))

    results = []
    # Calculate p-value for each pair of tau and n
    for i, row in df.iterrows():
        tau = row["Correlation"]
        n = N_PAIRS
        results.append((kendalls_tau_p_value(tau, n)))
    
    # Add column to dataframe
    df["p-value"] = results

    # Save to csv
    csv_filename = Path(correlations_path / "correlations.csv")
    df.to_csv(csv_filename, index=False)
