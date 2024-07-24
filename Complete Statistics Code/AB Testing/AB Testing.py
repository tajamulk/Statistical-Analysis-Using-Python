import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np

# Generate sample data
np.random.seed(42)  # For reproducibility
user_id = np.arange(1, 101)
group = np.random.choice(['A', 'B'], size=100)
converted = np.random.choice([0, 1], size=100)

df = pd.DataFrame({'user_id': user_id, 'group': group, 'converted': converted})

# Calculate conversion rates
conversion_rates = df.groupby('group')['converted'].mean()
print("Conversion Rates:")
print(conversion_rates)

# Create a contingency table
contingency_table = pd.crosstab(df['group'], df['converted'])

# Perform the chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi2: {chi2}, p-value: {p}")

# Draw conclusions
if p < 0.05:
    print("The difference in conversion rates is statistically significant.")
else:
    print("The difference in conversion rates is not statistically significant.")
