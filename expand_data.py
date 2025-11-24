import pandas as pd
import numpy as np


def expand_dataset():
    # 1. Load existing small data
    print(">> Loading original data...")
    try:
        df = pd.read_csv('data/loan_data.csv')
    except FileNotFoundError:
        print("Error: loan_data.csv not found in data/ folder.")
        return

    original_count = len(df)
    target_count = 10000  # We want 10,000 rows total

    print(f">> Original rows: {original_count}")
    print(f">> Generating {target_count} synthetic rows...")

    # 2. Calculate how many copies we need
    # We will sample with replacement (pick random rows from original repeatedly)
    new_df = df.sample(n=target_count, replace=True).reset_index(drop=True)

    # 3. Add "Noise" to make them unique (Data Augmentation)
    # We don't want exact duplicates, we want variations.

    # -- ApplicantIncome: Add random noise +/- 10%
    noise = np.random.uniform(-0.1, 0.1, size=target_count)
    new_df['ApplicantIncome'] = new_df['ApplicantIncome'] * (1 + noise)
    new_df['ApplicantIncome'] = new_df['ApplicantIncome'].astype(int)

    # -- CoapplicantIncome: Add random noise +/- 10%
    noise = np.random.uniform(-0.1, 0.1, size=target_count)
    new_df['CoapplicantIncome'] = new_df['CoapplicantIncome'] * (1 + noise)
    new_df['CoapplicantIncome'] = new_df['CoapplicantIncome'].astype(int)

    # -- LoanAmount: Add random noise +/- $5k
    noise = np.random.uniform(-5, 5, size=target_count)
    new_df['LoanAmount'] = new_df['LoanAmount'] + noise
    new_df['LoanAmount'] = new_df['LoanAmount'].abs()  # Ensure no negative loans

    # -- Inject "Common Sense" Logic to fix the Dependents Issue --
    # We will artificially make it harder for people with 3+ dependents to be approved
    # to counter the bad correlation you found.

    # Find rows with 3+ dependents and Loan_Status = Y
    mask = (new_df['Dependents'] == '3+') & (new_df['Loan_Status'] == 'Y')

    # Flip 30% of them to 'N' (Rejection) to teach the AI that kids = cost
    random_flip = np.random.choice([True, False], size=mask.sum(), p=[0.3, 0.7])

    # Get indices to flip
    indices_to_flip = new_df[mask].index[random_flip]
    new_df.loc[indices_to_flip, 'Loan_Status'] = 'N'

    # 4. Save the big file
    new_df.to_csv('data/loan_data_large.csv', index=False)
    print(f">> Success! Saved 10,000 rows to 'data/loan_data_large.csv'")


if __name__ == "__main__":
    expand_dataset()