# Flight Volume and Delay Analysis using Big Data

This project analyzes flight data from various sources using PySpark, Pandas, MLlib and other data-related libraries. It presents techniques for loading, cleaning, analyzing, and visualizing large volumes of flight data.

## Setup
1. Clone this repository.  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
    ```
3. Download the flight data (2016-2020) from the [Kaggle](https://www.kaggle.com/datasets/sherrytp/airline-delay-analysis) website.
4. Extract the data and place it in the `Data` directory.
5. Run the Jupyter notebook `main.ipynb`.

## Data
The data consists of the following columns:
- FL_DATE
- OP_CARRIER
- OP_CARRIER_FL_NUM
- ORIGIN
- DEST
- CRS_DEP_TIME
- DEP_TIME
- DEP_DELAY
- TAXI_OUT
- WHEELS_OFF
- WHEELS_ON
- TAXI_IN
- CRS_ARR_TIME
- ARR_TIME
- ARR_DELAY
- CANCELLED
- CANCELLATION_CODE
- DIVERTED
- CRS_ELAPSED_TIME
- ACTUAL_ELAPSED_TIME
- AIR_TIME
- DISTANCE
- CARRIER_DELAY
- WEATHER_DELAY
- NAS_DELAY
- SECURITY_DELAY
- LATE_AIRCRAFT_DELAY
- Unnamed: 27

## Cleaning
The data is cleaned by removing rows with missing values and columns that are not needed for the analysis. The missing values are replaced with 0 to avoid errors during analysis.
The data is also filtered to include only the years 2016-2018. 

## Analysis 
The analysis includes the following:

1. Flight Volume Analysis
2. Flight Delay Analysis
3. Carrier Analysis
4. Airport Analysis
5. Route Analysis

## Visualization
The analysis is visualized using various plots, including bar plots, boxplots, and line plots. The visualizations help in understanding the trends and patterns in the flight data.

## Conclusion
The project provides insights into flight volume and delays using big data tools and techniques. It demonstrates how to load, clean, analyze, and visualize large volumes of flight data efficiently.