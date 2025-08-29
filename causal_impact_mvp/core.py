from causalimpact import CausalImpact
import pandas as pd
import numpy as np

def prepare_data_for_causal_impact(y, X):
  """
  Prepares treated and synthetic control data for causal impact analysis,
  ensuring they are in the correct format and aligned by time.

  Args:
    y: pandas Series representing the treated data.
    X: pandas DataFrame representing the synthetic control data.

  Returns:
    A tuple containing the treated data as a DataFrame and the synthetic
    control data as a DataFrame, aligned by index.
  """
  # Convert the treated Series to a DataFrame
  y_df = y.to_frame()

  # Ensure both DataFrames have the same index and are aligned
  # This assumes that both y and X have a compatible index representing time.
  # If the indices are not the same, you might need to perform a join or merge.
  # For this function, we assume they are meant to be aligned and check for consistency.
  if not y_df.index.equals(X.index):
      # In a real scenario, you might handle this misalignment (e.g., merge, resample)
      # For this example, we'll raise an error to indicate the issue.
      raise ValueError("Indices of treated data (y) and synthetic control data (X) do not match.")

  return y_df, X
    
def run_causal_impact_analysis(y, X, treatment_time):
  """
  Runs a causal impact analysis using treated data, synthetic control data,
  and a treatment time point, then returns a summary and visualizes the impact.

  Args:
    y: pandas Series representing the treated data.
    X: pandas DataFrame representing the synthetic control data.
    treatment_time: The time point when the intervention or treatment occurs.

  Returns:
    A string containing the summary of the causal impact analysis.
  """
    # Prepare the data
  try:
    y_prepared, X_prepared = prepare_data_for_causal_impact(y, X)
  except ValueError as e:
    print(f"Error preparing data: {e}")
    return None


  # Define pre- and post-treatment periods
  # Assuming a 0-based index for time points
  pre_treatment_period = (0, treatment_time - 1)
  post_treatment_period = (treatment_time, len(y) - 1)


  # Concatenate the treated and synthetic control data
  data_for_impact = pd.concat([y_prepared, X_prepared], axis=1)

  # Instantiate and run the CausalImpact model
  try:
    impact = CausalImpact(data_for_impact, list(pre_treatment_period), list(post_treatment_period))

    # Summarize the results
    summary = impact.summary()
    print(summary)

    # Plot the results
    impact.plot()

    return summary

  except Exception as e:
    print(f"Error during causal impact analysis: {e}")
    return None
def run_causal_impact_analysis(y, X, treatment_time):
  """
  Runs a causal impact analysis using treated data, synthetic control data,
  and a treatment time point, then returns a summary and visualizes the impact.

  Args:
    y: pandas Series representing the treated data.
    X: pandas DataFrame representing the synthetic control data.
    treatment_time: The time point when the intervention or treatment occurs.

  Returns:
    A string containing the summary of the causal impact analysis.
  """
  # Prepare the data
  try:
    y_prepared, X_prepared = prepare_data_for_causal_impact(y, X)
  except ValueError as e:
    print(f"Error preparing data: {e}")
    return None

  # Define pre- and post-treatment periods
  # Assuming a 0-based index for time points
  pre_treatment_period = (0, treatment_time - 1)
  post_treatment_period = (treatment_time, len(y) - 1)


  # Concatenate the treated and synthetic control data
  data_for_impact = pd.concat([y_prepared, X_prepared], axis=1)

  # Instantiate and run the CausalImpact model
  try:
    impact = CausalImpact(data_for_impact, list(pre_treatment_period), list(post_treatment_period))

    # Summarize the results
    summary = impact.summary()
    print(summary)

    # Plot the results
    impact.plot()

    return summary

  except Exception as e:
    print(f"Error during causal impact analysis: {e}")
    return None
      
def generate_causal_data(n_points=100, treatment_time=70, causal_effect=5, n_covariates=3):
  """
  Generates synthetic time series data for causal impact analysis and synthetic control.

  Args:
    n_points: Total number of data points in the time series.
    treatment_time: The time point when the intervention or treatment occurs.
    causal_effect: The magnitude of the causal effect introduced after the treatment time.
    n_covariates: The number of additional covariates to generate for synthetic control.

  Returns:
    A pandas DataFrame with the time series data ('y') and additional covariates.
  """
  time = np.arange(n_points)
  # Generate a base trend
  base_trend = 0.5 * time + 10

  # Introduce some seasonality
  seasonality = 5 * np.sin(time / 10)

  # Add some noise
  noise = np.random.normal(0, 2, n_points)

  # Combine components to create the base time series
  y = base_trend + seasonality + noise

  # Introduce causal effect after treatment time
  y[treatment_time:] += causal_effect

  data = pd.DataFrame({'y': y})

  # Generate additional covariates for synthetic control
  for i in range(n_covariates):
      covariate_noise = np.random.normal(0, 1.5, n_points)
      # Create covariates that are correlated with the base trend and seasonality
      data[f'x{i+1}'] = base_trend * (1 + np.random.rand() * 0.1) + seasonality * (1 + np.random.rand() * 0.1) + covariate_noise

  return data

def run_demo(n_points=150, treatment_time=100, causal_effect=10, n_covariates=3):
  """
  Generates synthetic causal data and runs a causal impact analysis demo.

  Args:
    n_points: Total number of data points in the time series.
    treatment_time: The time point when the intervention or treatment occurs.
    causal_effect: The magnitude of the causal effect introduced after the treatment time.
    n_covariates: The number of additional covariates to generate for synthetic control.
  """
  print("Generating synthetic causal data...")
  demo_data = generate_causal_data(n_points=n_points, treatment_time=treatment_time, causal_effect=causal_effect, n_covariates=n_covariates)
  display(demo_data.head())

  print("\nRunning causal impact analysis...")
  # Assuming 'y' is the treated column and other columns are covariates
  treated_series = demo_data['y']
  covariate_columns = [col for col in demo_data.columns if col.startswith('x')]
  synthetic_control_df = demo_data[covariate_columns]

  run_causal_impact_analysis(treated_series, synthetic_control_df, treatment_time)
