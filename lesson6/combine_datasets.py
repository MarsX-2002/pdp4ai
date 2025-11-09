import pandas as pd
import json
import re
import io

def extract_financials_from_sql():
    """Extract financial data from the SQL file and return as a DataFrame."""
    # Read the SQL file
    with open('financials.sql', 'r') as f:
        sql_content = f.read()
    
    # Extract all INSERT statements
    insert_statements = [line for line in sql_content.split('\n') if line.startswith('INSERT')]
    
    # Extract data using regex
    data = []
    for stmt in insert_statements:
        # Match the VALUES part of the INSERT statement
        match = re.search(r'VALUES\s*\((.*?)\);?$', stmt, re.IGNORECASE)
        if match:
            values_str = match.group(1)
            # Split by comma but handle potential commas within quotes
            values = [v.strip(" '") for v in re.split(r"\s*,\s*(?=(?:[^']*'[^']*')*[^']*$)", values_str)]
            try:
                # Convert to appropriate types
                row = {
                    'client_id': int(float(values[0])),
                    'income': float(values[1]),
                    'credit_score': int(float(values[2])),
                    'loan_amount': float(values[3]),
                    'past_due': int(float(values[4])),
                    'monthly_expenses': float(values[5]),
                    'savings_balance': float(values[6])
                }
                data.append(row)
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line: {stmt}")
                print(f"Error: {e}")
    
    return pd.DataFrame(data)

def main():
    # Read client information
    print("Reading client information...")
    client_df = pd.read_csv('client_info.csv')
    
    # Read credit risk data
    print("Reading credit risk data...")
    with open('credit_risk.json', 'r') as f:
        credit_data = json.load(f)
    credit_df = pd.DataFrame(credit_data)
    
    # Extract financial data from SQL
    print("Extracting financial data from SQL...")
    financials_df = extract_financials_from_sql()
    
    # Merge all dataframes on client_id
    print("Merging datasets...")
    # First merge client info with credit data
    combined_df = pd.merge(client_df, credit_df, on='client_id', how='left')
    # Then merge with financial data
    combined_df = pd.merge(combined_df, financials_df, on='client_id', how='left')
    
    # Save to CSV
    output_file = 'combined_credit_data.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"Total clients: {len(combined_df)}")
    print("\nFirst 5 rows of the combined dataset:")
    print(combined_df.head())
    print("\nColumn names:")
    print(combined_df.columns.tolist())
    print("\nData types:")
    print(combined_df.dtypes)

if __name__ == "__main__":
    main()
