#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


draft_df = pd.read_csv(r"C:\Users\Omie\Desktop\DSC 530 Project\nfl_draft_prospects.csv")


# In[3]:


print("Draft Data:")
print(draft_df.head())


# In[4]:


performance_df = pd.read_csv(r"C:\Users\Omie\Desktop\DSC 530 Project\yearly_player_data.csv")


# In[5]:


print("Performance Data:")
print(performance_df.head())


# In[6]:


# Display column names
print("Draft Dataset Columns:", draft_df.columns)
print("Performance Dataset Columns:", performance_df.columns)


# In[7]:


# Convert names to lowercase and strip spaces
draft_df["player_name"] = draft_df["player_name"].str.lower().str.strip()
performance_df["player_name"] = performance_df["player_name"].str.lower().str.strip()


# In[8]:


# Merge datasets on player_name
merged_df = pd.merge(draft_df, performance_df, on="player_name", how="inner")

# Display merged dataset
print(merged_df.head())


# In[9]:


print(merged_df.isnull().sum())


# In[10]:


# Fill missing performance metrics with 0
merged_df.fillna(0, inplace=True)


# In[11]:


merged_df.to_csv("merged_nfl_draft_data.csv", index=False)


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind
import statsmodels.api as sm


# In[13]:


df = pd.read_csv(r"C:\Users\Omie\Desktop\DSC 530 Project\merged_nfl_draft_data.csv", low_memory=False)


# In[14]:


# Select key variables
variables = ['games', 'total_yards', 'total_tds', 'fantasy_points_ppr', 'round']


# In[15]:


# Histograms
variables = ['games', 'total_yards', 'total_tds', 'fantasy_points_ppr', 'round']

for var in variables:
    plt.figure(figsize=(6, 4))
    plt.hist(df[var], bins=20, edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()


# In[16]:


# Boxplots for outliers
for var in variables:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[var])
    plt.title(f'Box Plot of {var}')
    plt.show()


# In[17]:


# Summary statistics
summary_stats = df[variables].describe().T
summary_stats['mode'] = df[variables].mode().iloc[0]
summary_stats['spread'] = summary_stats['max'] - summary_stats['min']

print(summary_stats)

summary_stats.to_csv("summary_statistics.csv")


# In[18]:


# Probability Mass Function
def compute_pmf(data):
    counts = data.value_counts(normalize=True)
    return counts.sort_index()

# Compute PMF for first-round vs. later rounds
first_round_pmf = compute_pmf(df[df['round'] == 1]['games'])
later_round_pmf = compute_pmf(df[df['round'] > 1]['games'])

# Display PMF results
print("First Round PMF:\n", first_round_pmf.head())
print("Later Round PMF:\n", later_round_pmf.head())


# In[19]:


# Cumulative Distribution Function
def compute_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf

x, y = compute_cdf(df['total_yards'])

plt.figure(figsize=(6, 4))
plt.plot(x, y, marker='o', linestyle='none')
plt.xlabel('Total Yards')
plt.ylabel('CDF')
plt.title('CDF of Total Yards')
plt.grid()
plt.show()


# In[20]:


import scipy.stats as stats


# In[21]:


data = df['fantasy_points_ppr'].dropna()  

# Fit a normal distribution to the data
mu, std = stats.norm.fit(data)

# Generate values for plotting the fitted distribution
xmin, xmax = data.min(), data.max()
x = np.linspace(xmin, xmax, 100)
pdf = stats.norm.pdf(x, mu, std)

# Plot histogram and fitted normal distribution
plt.figure(figsize=(8, 5))
plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label="Histogram")
plt.plot(x, pdf, 'k', linewidth=2, label=f"Normal Fit (μ={mu:.2f}, σ={std:.2f})")

plt.title("Analytical Distribution of Fantasy Points (Normal Fit)")
plt.xlabel("Fantasy Points (PPR)")
plt.ylabel("Density")
plt.legend()
plt.grid()
plt.show()

# Perform normality test (Shapiro-Wilk Test)
shapiro_test_stat, shapiro_p_value = stats.shapiro(data.sample(500, random_state=42)) if len(data) > 500 else stats.shapiro(data)

# Store results in a DataFrame
distribution_results = {
    "Mean (μ)": mu,
    "Standard Deviation (σ)": std,
    "Shapiro-Wilk Test Statistic": shapiro_test_stat,
    "Shapiro-Wilk P-Value": shapiro_p_value
}

# Convert to DataFrame for display
distribution_df = pd.DataFrame([distribution_results])

print(distribution_df.to_string(index=False))


# In[22]:


# Scatter Plot: Draft Round vs. Fantasy Points
plt.figure(figsize=(6, 4))
plt.scatter(df['draft_round'], df['fantasy_points_ppr'])
plt.xlabel('Draft Round')
plt.ylabel('Fantasy Points')
plt.title('Draft Round vs. Fantasy Points')
plt.show()


# In[23]:


# Scatter Plot: Draft Round vs. Total Yards
plt.figure(figsize=(6, 4))
plt.scatter(df['draft_round'], df['total_yards'])
plt.xlabel('Draft Round')
plt.ylabel('Total Yards')
plt.title('Draft Round vs. Total Yards')
plt.show()


# In[24]:


# Pearson correlation
corr, _ = stats.pearsonr(df['draft_round'], df['fantasy_points_ppr'])
print(f"Pearson Correlation: {corr}")


# In[25]:


# Hypothesis Testing
first_round = df[df['round'] == 1]['fantasy_points_ppr']
later_round = df[df['round'] > 1]['fantasy_points_ppr']

t_stat, p_val = ttest_ind(first_round, later_round, equal_var=False)
print(f"T-Statistic: {t_stat}, P-Value: {p_val}")


# In[26]:


# Regression Analysis
X = df[['round']]
y = df['fantasy_points_ppr']

X = sm.add_constant(X)  # Add intercept
model = sm.OLS(y, X).fit()
print(model.summary())


# In[ ]:




