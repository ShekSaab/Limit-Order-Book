# Import essential libraries / packages
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from statsmodels.tsa.stattools import acf

file_name = "C:\\Users\\sirji\\OneDrive\\Desktop\\ECON 5149 Assignment\\sp500_data.xlsx"
def analyze_real_data(file_name):
    # Calculates stylized facts for S&P 500 data.
    try:
        # Load the Excel file
        df = pd.read_excel(file_name)
        
        # Convert 'date' column to datetime objects
        df['date'] = pd.to_datetime(df['date'])
        
        # Set Date as the index
        df = df.set_index('date')
        
        # Filter for the specified period (inclusive)
        df_filtered = df.loc['2013-01-01':'2018-01-01']
        
        # Compute daily log-returns
        log_returns = np.log(df_filtered['close'] / df_filtered['close'].shift(1))
        
        # Drop the first 'NaN' value created by .shift()
        log_returns = log_returns.dropna()
        
        if log_returns.empty:
            print("Error: No data found in the specified date range.")
            return

        print("----- S&P 500 Analysis -----")

        # Compute excess kurtosis 
        excess_kurt = kurtosis(log_returns, fisher=True)
        print(f"\nExcess kurtosis of returns: {excess_kurt:.4f}")
        
        # Compute ACF of squared returns
        squared_returns = log_returns**2
        
        # Calculate ACF up to 6 lags
        acf_sq_returns = acf(squared_returns, nlags=6, fft=False)[1:]
        
        print("\nACF of squared returns:")
        for i, val in enumerate(acf_sq_returns):
            print(f"  Lag {i+1}: {val:.4f}")
            

    except FileNotFoundError:
        print(f"Error: File not found at {file_name}")
        print("Make sure 'sp500_data.xlsx' is in the same directory.")
    except KeyError as e:
        print(f"Error: Column not found: {e}")
        print("Update the script to match your Excel file's column names (e.g., 'Date', 'Close').")

# Main execution
if __name__ == "__main__":
    analyze_real_data('sp500_data.xlsx')