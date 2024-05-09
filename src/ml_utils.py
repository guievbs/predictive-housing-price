import pandas as pd
import numpy as np

def preprocess_data(data) -> pd.DataFrame:
    """
    Preprocesses the input data by applying logarithm transformation to selected columns,
    creating dummy variables for the 'ocean_proximity' column, and calculating bedroom
    ratio and household rooms.

    Parameters:
    data (DataFrame): Input data to be preprocessed.

    Returns:
    DataFrame: Preprocessed data.
    """
    cols = ['total_rooms', 'total_bedrooms', 'population', 'households']
    
    # Apply logarithm transformation to selected columns
    for col in cols:
        data[col] = np.log(data[col] + 1)
    
    # Create dummy variables for the 'ocean_proximity' column
    dummies = pd.get_dummies(data['ocean_proximity'], dtype=int, dummy_na=False)
    data = pd.concat([data, dummies], axis=1)
    
    # Remove the original 'ocean_proximity' column
    data.drop(['ocean_proximity'], axis=1, inplace=True)
    
    # Calculate the ratio of bedrooms to total rooms and household rooms
    data['bedroom_ratio'] = data['total_bedrooms'] / data['total_rooms']
    data['household_rooms'] = data['total_rooms'] / data['households']
    
    return data

def split_features_target(data):
    """
    Splits the input data into features (X) and target (y).

    Parameters:
    data (DataFrame): Input data.
    target_column (str): Name of the target column.

    Returns:
    DataFrame, Series: Features (X) and target (y).
    """
    X = data.drop(columns=['median_house_value'])
    y = data['median_house_value']
    return X, y


