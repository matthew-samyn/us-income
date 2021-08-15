import pandas as pd
import numpy as np

def print_unique_values_from_columns(df, column):
    unique_values = df[column].unique()
    number_unique_values = df[column].nunique()
    print(f""" 
The {column}-column contains:
Number of unique values:{number_unique_values}
Values: {unique_values}""")

def normalizing_array(array):
    norm = np.linalg.norm(array)
    normalize_array = array / norm
    return normalize_array
