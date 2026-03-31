# COMP4037 Research Methods - Coursework 2

An interactive data visualisation dashboard exploring NHS Hospital Episode Statistics (HES) Primary Diagnosis data from 1998 to 2023, built for COMP4037 Research Methods Coursework 2.

## Requirements

Python is required. Install dependencies in terminal with:

    pip install -r requirements.txt

## Running the dashboard

    python dashboard.py

Once running, open your browser and navigate to:

    http://127.0.0.1:8050/

## Data

The dataset (`final_nhs_full.csv`) is a cleaned and consolidated version of the NHS HES Primary Diagnosis 3-character files, spanning 1998–2023. Raw data is publicly available.

## Features

- Heatmap of top and bottom conditions by category, coloured by mean length of stay
- Animated scatter plot of admissions vs. mean length of stay over time
- Hover-triggered line chart showing longitudinal trends per diagnosis
- Summary statistic cards

## Author

Morgan Jones - University of Nottingham
