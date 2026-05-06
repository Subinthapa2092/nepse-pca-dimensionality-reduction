# Dimensionality Reduction of NEPSE Market Data Using PCA (2021-2026)
#
# Research Question:
# Can we reduce the 6 correlated NEPSE trading features (Open, High, Low, Close,
# Percent Change, and Volume) into fewer principal components without losing
# the key market information?


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D


# Step 1: Loading and Cleaning the Data

df = pd.read_csv("nepsealpha_export_price_NEPSE_2021-05-06_2026-05-06_unadjusted.csv")

df['Percent Change'] = df['Percent Change'].str.replace('%', '').astype(float)
df['Volume'] = df['Volume'].str.replace(',', '').astype(float)
df['Date'] = pd.to_datetime(df['Date'])

print(df.isnull().sum())

features = ['Open', 'High', 'Low', 'Close', 'Percent Change', 'Volume']
X = df[features].copy()
print(X.shape)
print(X)


# Step 2: Exploratory Analysis

print(X.describe().round(2))

# Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(X.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, square=True)
plt.title('Feature Correlation - NEPSE')
plt.tight_layout()
plt.show()

# Conclusion:
# The heatmap shows that Open, High, Low, and Close are almost perfectly correlated,
# meaning they carry redundant information. Volume has a moderate relationship with
# these price features, while Percent Change is largely independent. This confirms
# that OHLC features can be reduced using PCA without losing much information.

# Feature distributions
X.hist(figsize=(10, 6), bins=40, edgecolor='white')
plt.suptitle('Feature Distributions')
plt.tight_layout()
plt.show()

# Conclusion:
# Open, High, Low, and Close are approximately bell-shaped and centered around
# the 2400 level. Percent Change is tightly centered around zero indicating
# most daily movements are within 1-2%. Volume is strongly right-skewed,
# meaning most trading days have moderate activity while a few days experience
# unusually high spikes, which is common in stock market trading behavior.


# Step 3: Standardise the Features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print('Means:', X_scaled.mean(axis=0).round(4))
print('Stds: ', X_scaled.std(axis=0).round(4))

X_scaled_df = pd.DataFrame(X_scaled, columns=features)
print(X_scaled_df.head())

# Conclusion:
# All six features now have mean close to 0 and standard deviation 1.
# Without scaling, PCA would be biased toward large value features like Volume
# and would not capture the real structure of the data.


# Step 4: Performing PCA

pca = PCA(n_components=6)
pca.fit(X_scaled)

exp_var = pca.explained_variance_ratio_
print('Explained variance ratio:')
for i, v in enumerate(exp_var):
    print(f'  PC{i+1}: {v:.3f} ({v*100:.1f}%)')

cum_var = np.cumsum(exp_var)
print('\nCumulative variance:')
for i, v in enumerate(cum_var):
    print(f'  PC1-PC{i+1}: {v:.3f} ({v*100:.1f}%)')

X_pca = pca.transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(6)])
pca_df['Date'] = df['Date'].values
print(pca_df.head())

# Conclusion:
# PC1 explains 76.4%, PC2 adds 16.9%, and PC3 adds 6.7%. Together the first
# three components explain 100% of the information. PC4 to PC6 carry almost
# no useful variance. The dataset can be reduced from 6 dimensions to 2 or 3
# without losing important information.


# Step 5: Scree Plot

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

pc_labels = [f'PC{i+1}' for i in range(6)]

ax1.bar(pc_labels, exp_var * 100, color='steelblue', edgecolor='white')
ax1.plot(pc_labels, exp_var * 100, 'o-', color='navy')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Variance Explained (%)')
ax1.set_title('Scree Plot')

ax2.plot(pc_labels, cum_var * 100, 's-', color='steelblue')
ax2.axhline(90, color='red', linestyle='--', label='90% threshold')
ax2.set_xlabel('Principal Component')
ax2.set_ylabel('Cumulative Variance (%)')
ax2.set_title('Cumulative Explained Variance')
ax2.legend()

plt.tight_layout()
plt.show()

n_90 = np.argmax(cum_var >= 0.90) + 1
print(f'PCs needed for 90% variance: {n_90}')

# Conclusion:
# The scree plot shows a sharp drop after PC1. Cumulative variance exceeds 90%
# at PC2, making two components sufficient. PC4 to PC6 add negligible information.
# Retaining 2 principal components provides an efficient dimensionality reduction.


# Step 6: Loadings Heatmap and Biplot

loadings = pd.DataFrame(
    pca.components_.T,
    index=features,
    columns=[f'PC{i+1}' for i in range(6)]
)
print(loadings.round(3))

plt.figure(figsize=(8, 4))
sns.heatmap(loadings, annot=True, fmt='.2f', cmap='RdBu_r',
            vmin=-1, vmax=1, center=0)
plt.title('PCA Loadings - Feature Contribution per PC')
plt.tight_layout()
plt.show()

# Conclusion:
# PC1 captures overall price movement where Open, High, Low, Close all load strongly.
# PC2 represents Percent Change and PC3 is dominated by Volume.
# The dataset is effectively summarized by the first 2-3 principal components.

# Biplot
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.25, s=12, color='steelblue', label='Trading days')

scale = 3
colors = ['crimson', 'darkorange', 'green', 'purple', 'brown', 'teal']

for i, feat in enumerate(features):
    dx = pca.components_[0, i] * scale
    dy = pca.components_[1, i] * scale
    ax.annotate('', xy=(dx, dy), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=colors[i], lw=2.5))
    ax.plot(dx, dy, 'o', color=colors[i], markersize=5)

legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=feat)
                   for i, feat in enumerate(features)]

pc_idx = features.index('Percent Change')
dx_pc = pca.components_[0, pc_idx] * scale
dy_pc = pca.components_[1, pc_idx] * scale
ax.text(dx_pc + 0.1, dy_pc + 0.1, 'Percent Change',
        fontsize=10, color='crimson', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='none'))

ax.set_xlabel('PC1  (76.4% variance) - Price Level', fontsize=11)
ax.set_ylabel('PC2  (16.9% variance) - Daily Return', fontsize=11)
ax.set_title('Biplot - NEPSE PCA (PC1 vs PC2)', fontsize=13)
ax.axhline(0, color='gray', lw=0.5, linestyle='--')
ax.axvline(0, color='gray', lw=0.5, linestyle='--')
ax.legend(handles=legend_elements, loc='lower right',
          title='Features', fontsize=10, title_fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.show()

# Conclusion:
# PC1 mainly represents the overall price level as Open, High, Low, and Close
# are strongly aligned. PC2 captures daily movement driven by Percent Change.
# Volume has a smaller separate influence.


# Step 7: PC1 Over Time - Market Trend Interpretation

plt.figure(figsize=(12, 4))
plt.plot(df['Date'].values, pca_df['PC1'], color='steelblue', lw=0.8)
plt.axhline(0, color='red', lw=0.5, linestyle='--')
plt.title('PC1 Over Time - NEPSE Market Trend (2021-2026)')
plt.xlabel('Date')
plt.ylabel('PC1 Score')
plt.tight_layout()
plt.show()

final_df = pd.DataFrame({
    'Date': df['Date'],
    'PC1_price_level': pca_df['PC1'],
    'PC2_daily_return': pca_df['PC2']
})
final_df.to_csv('nepse_pca_scores.csv', index=False)
print('Saved!')

# Conclusion:
# PC1 shows strong growth in 2021, a prolonged decline through 2022-2023,
# and recovery from mid-2024 onward. PC1 reflects the shift from a bearish
# phase to a more stable and improving market trend.


# Step 8: Reconstruction and Validation

pca_2 = PCA(n_components=2)
X_reduced = pca_2.fit_transform(X_scaled)
X_reconstructed = pca_2.inverse_transform(X_reduced)

recon_error = np.mean((X_scaled - X_reconstructed) ** 2)
print(f'Mean Squared Reconstruction Error (2 PCs): {recon_error:.4f}')

plt.figure(figsize=(12, 4))
plt.plot(df['Date'], X_scaled[:, 3], label='Original Close (scaled)', lw=1)
plt.plot(df['Date'], X_reconstructed[:, 3], label='Reconstructed (2 PCs)',
         lw=1, linestyle='--', color='red')
plt.title('Original vs Reconstructed Close Price (2 PCs)')
plt.xlabel('Date')
plt.legend()
plt.tight_layout()
plt.show()

# Conclusion:
# The reconstructed Close price using only 2 principal components closely follows
# the original scaled Close price across the entire 2021-2026 period. The two
# lines are nearly identical, confirming that 93.3% variance retention is
# sufficient to preserve the market trend with minimal information loss.


# Final Conclusion:
#
# This project applied PCA on 5 years of NEPSE daily trading data (2021-2026)
# using 6 features: Open, High, Low, Close, Percent Change, and Volume.
#
# PC1 (76.4%) represents overall price movement where Open, High, Low,
# and Close move together.
# PC2 (16.9%) mainly captures daily return behavior driven by Percent Change.
# PC3 (6.7%) reflects variation in trading volume.
# PC4 to PC6 contribute almost nothing because the OHLC features are highly correlated.
#
# Using only 2 principal components preserves 93.3% of the total information,
# showing that NEPSE daily data is highly redundant and can be compressed
# effectively for analysis and machine learning models.