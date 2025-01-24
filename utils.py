from pyspark.sql.functions import col, when, isnull, count

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, hour, month, dayofweek, isnull, count, mean, stddev, udf
from pyspark.sql.types import IntegerType, FloatType, StringType, DoubleType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def load_data_and_show_info(spark, data_path):
    """
    Load a CSV file into a Spark DataFrame and show the schema and first few rows.

    Args:
        spark (pyspark.sql.SparkSession): Spark session object.
        data_path (str): Path to the CSV file.

    Returns:
        pyspark.sql.DataFrame: Loaded Spark DataFrame.
    """
    # Load the dataset
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    return df

def calculate_null_statistics(spark, df):
    """
    Calculate the null counts and percentages for all columns in a Spark DataFrame.

    Args:
        df (pyspark.sql.DataFrame): Input Spark DataFrame.

    Returns:
        pyspark.sql.DataFrame: A Spark DataFrame with columns: "Column", "Null Count", "Null Percentage".
    """
    # Calculate null counts for each column
    null_counts = df.select([count(when(isnull(c), c)).alias(c) for c in df.columns])
    
    # Total number of rows in the DataFrame
    total_rows = df.count()
    
    # Calculate null percentage for each column
    null_percentage = null_counts.select([(col(c) / total_rows * 100).alias(c) for c in df.columns])
    
    # Combine column names, null counts, and percentages
    null_counts_list = null_counts.collect()[0].asDict()
    null_percentage_list = null_percentage.collect()[0].asDict()
    
    # Create a list of tuples with column statistics
    columns_summary = [
        (col_name, null_counts_list[col_name], round(null_percentage_list[col_name], 2))
        for col_name in df.columns
    ]
    
    # Create a Spark DataFrame for the summary
    summary_df = spark.createDataFrame(columns_summary, ["Column", "Null Count", "Null Percentage"])
    
    return summary_df

def plot_column_distributions(spark_df, max_unique=50, sample_fraction=0.1, max_graphs=10, graphs_per_row=2):
    """
    Plots the distribution of columns in a PySpark DataFrame.

    Parameters:
    - spark_df: PySpark DataFrame
    - max_unique: Maximum number of unique values in a column to be considered for plotting
    - sample_fraction: Fraction of the DataFrame to sample for plotting
    - max_graphs: Maximum number of graphs to display
    - graphs_per_row: Number of graphs to display per row

    Returns:
    - None
    """
    # Sample the DataFrame
    sampled_df = spark_df.sample(withReplacement=False, fraction=sample_fraction)

    pandas_df = sampled_df.toPandas()

    columns_to_plot = [col for col in pandas_df.columns if 1 < pandas_df[col].nunique() <= max_unique]

    columns_to_plot = columns_to_plot[:max_graphs]

    num_rows = (len(columns_to_plot) + graphs_per_row - 1) // graphs_per_row
    
    fig, axes = plt.subplots(num_rows, graphs_per_row, figsize=(6 * graphs_per_row, 4 * num_rows))
    axes = axes.flatten()

    for i, col in enumerate(columns_to_plot):
        ax = axes[i]
        if pd.api.types.is_numeric_dtype(pandas_df[col]):
            sns.histplot(pandas_df[col], kde=False, ax=ax)
        else:
            sns.countplot(y=pandas_df[col], order=pandas_df[col].value_counts().index, ax=ax)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()