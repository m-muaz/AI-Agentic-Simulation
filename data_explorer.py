import pandas as pd
import os

def main():
    # Define the path to the .dta file
    data_file_path = os.path.join(os.getcwd(), "replication files", "Data", "PovertyTraps_replication_data.dta")
    output_dir = os.path.join(os.getcwd(), "data", "processed")
    output_file_path = os.path.join(output_dir, "processed_agent_data.csv")

    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at {data_file_path}")
        print("Please ensure the 'replication files' directory is in the project root.")
        return

    try:
        df = pd.read_stata(data_file_path)
        print(f"Successfully loaded data from {data_file_path}")

        # Select and rename columns to match agent attributes
        # We'll take the first observation for each household (hhid5) for initial agent creation
        # This simplifies initial agent population, assuming static initial conditions.
        # More complex scenarios might involve time-series data.
        initial_df = df.drop_duplicates(subset=['hhid5']).copy()

        processed_df = pd.DataFrame()
        processed_df['agent_id'] = initial_df['hhid5'].astype(str)
        processed_df['initial_wealth'] = initial_df['pAssets']
        processed_df['income'] = initial_df['total_income_resp']
        processed_df['education'] = initial_df['resp_educ_years']
        
        # Derive loan_access: True if loan_total_value > 0, False otherwise
        processed_df['loan_access'] = (initial_df['loan_total_value'] > 0).astype(bool)
        
        # Health and consumption_preference are not directly available, use defaults or derive
        # For now, we'll use placeholder values.
        processed_df['initial_health'] = 0.7 # Placeholder
        processed_df['consumption_preference'] = 0.6 # Placeholder

        # Handle missing values: fill with 0 for numerical, False for boolean
        processed_df['initial_wealth'] = processed_df['initial_wealth'].fillna(0)
        processed_df['income'] = processed_df['income'].fillna(0)
        processed_df['education'] = processed_df['education'].fillna(0)
        processed_df['loan_access'] = processed_df['loan_access'].fillna(False)
        
        # Ensure wealth and income are non-negative
        processed_df['initial_wealth'] = processed_df['initial_wealth'].apply(lambda x: max(0, x))
        processed_df['income'] = processed_df['income'].apply(lambda x: max(0, x))

        # Basic validation/cleaning for education and health
        processed_df['education'] = processed_df['education'].apply(lambda x: max(0, min(16, x))) # Assuming max 16 years of education
        processed_df['initial_health'] = processed_df['initial_health'].apply(lambda x: max(0.0, min(1.0, x))) # Clamp health between 0 and 1

        print("\nProcessed DataFrame head:")
        print(processed_df.head())
        print("\nProcessed DataFrame info:")
        processed_df.info()
        print("\nProcessed DataFrame describe:")
        print(processed_df.describe())

        # Save the processed data to CSV
        processed_df.to_csv(output_file_path, index=False)
        print(f"\nProcessed data saved to {output_file_path}")

    except Exception as e:
        print(f"An error occurred while processing the Stata file: {e}")
        print("Please ensure you have the 'pyreadstat' engine installed for pandas to read .dta files:")
        print("pip install pyreadstat")

if __name__ == "__main__":
    main()