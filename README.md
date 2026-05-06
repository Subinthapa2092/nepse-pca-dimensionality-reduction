# Dimensionality Reduction of NEPSE Market Data Using PCA (2021-2026)

## Overview

This project applies Principal Component Analysis (PCA) to 5 years of NEPSE
(Nepal Stock Exchange) daily trading data from May 2021 to May 2026. The goal
is to reduce 6 correlated trading features into fewer meaningful components
without losing key market information.

## Research Question

Can we reduce the 6 correlated NEPSE trading features (Open, High, Low, Close,
Percent Change, and Volume) into fewer principal components without losing the
key market information?

## Dataset

- Source: NepseAlpha (nepsealpha.com)
- Period: May 2021 to May 2026 (1,157 trading days)
- Features: Open, High, Low, Close, Percent Change, Volume
- File: `nepsealpha_export_price_NEPSE_2021-05-06_2026-05-06_unadjusted.csv`

## Project Structure

    nepse-pca-dimensionality-reduction/
        main.py        main script
        requirements.txt           required packages
        README.md                  project documentation
        .gitignore                 files to ignore
        nepse_pca_scores.csv       output file generated after running the script

## Steps

1. Loading and Cleaning the Data
2. Exploratory Analysis (correlation heatmap and feature distributions)
3. Standardising the Features
4. Performing PCA (all 6 components)
5. Scree Plot (variance analysis)
6. Loadings Heatmap and Biplot
7. PC1 Over Time (market trend interpretation)
8. Reconstruction and Validation

## Key Findings

- PC1 (76.4%) captures overall price level. Open, High, Low, and Close move together.
- PC2 (16.9%) captures daily return behavior, driven almost entirely by Percent Change.
- PC3 (6.7%) reflects trading volume variation.
- PC4 to PC6 carry near-zero variance due to multicollinearity of OHLC features.
- Retaining 2 PCs preserves 93.3% of all information.
- Mean Squared Reconstruction Error with 2 PCs is only 0.0670.

## How to Run

1. Clone the repository

        git clone https://github.com/yourusername/nepse-pca-dimensionality-reduction.git
        cd nepse-pca-dimensionality-reduction

2. Install dependencies

        pip install -r requirements.txt

3. Place the dataset CSV file in the same folder

4. Run the script

        main.py

## Output

After running, the script will generate and display 6 plots and save a file
called `nepse_pca_scores.csv` containing the PC1 and PC2 scores for each trading day.

## Requirements

Python 3.8 or above is required. See `requirements.txt` for package versions.

## Author

Subin Thapa
