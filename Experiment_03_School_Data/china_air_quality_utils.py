import numpy as np
import pandas as pd 
from scipy.stats import skew


# def process_df(df):
#     # The dataset contains PM (pollution) readings across different posts in the city.  We average them here
#     PM_vars = [f for f in df.columns if "PM" in f]

#     avg_pm = 0
#     for f in PM_vars: avg_pm += df[f]
#     avg_pm = avg_pm / len(PM_vars)

#     df = df.drop(PM_vars, axis=1)
#     df['avg_pm'] = avg_pm

#     # Create a date-time index
#     time_vars = ['year', 'month', 'day', 'hour']
#     time = pd.to_datetime(df[time_vars])
#     df.index = time

#     df['target'] = np.log1p(df['avg_pm'])

#     # Construct Features
#     drop_vars = ['year', 'month', 'day', 'hour', 'season']

#     cat_feats = ['month', 'day', 'hour', 'season', 'cbwd']
#     for f in cat_feats:
#         df[f] = pd.Categorical(df[f])

#     X = df.copy()
#     X = X.drop(drop_vars, axis=1)
#     X = X.drop(['target', 'avg_pm'], axis=1)

#     # Better variable names
#     X = X.rename(columns={'DEWP': 'DewPt', 
#                           'HUMI': 'Humidity', 
#                           'PRES': 'Press', 
#                           'TEMP': 'TempC', 
#                           'cbwd': 'WindDir', 
#                           'Iws': 'WindSp', 
                          
#                           'precipitation': 'PrecipHr', 
#                           'Iprec': 'PrecipCm'})

#     # Note that we drop the first dummy variable, because we'll be using OLS
#     X = pd.get_dummies(X, drop_first=True)
#     y = df['target']

#     Xy = pd.concat([X, y], axis=1)

#     numeric_feats = [f for f in X.columns if f not in time_vars and
#                      'WindDir' not in f and
#                      'season' not in f]

#     X['PrecipCm'] = X['PrecipCm'] - X['PrecipHr']

#     #log transform skewed numeric features:
#     skewed_feats = X[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
#     skewed_feats = skewed_feats[skewed_feats > 0.75]
#     skewed_feats = skewed_feats.index
#     #print(skewed_feats)

#     X[skewed_feats] = np.log1p(X[skewed_feats])

#     return df, X, y

import numpy as np
import pandas as pd 
from scipy.stats import skew

def process_df(df):
    # The dataset contains PM (pollution) readings across different posts in the city. We average them here
    PM_vars = [f for f in df.columns if "PM" in f]

    avg_pm = sum(df[f] for f in PM_vars) / len(PM_vars)
    df = df.drop(PM_vars, axis=1)
    df['avg_pm'] = avg_pm
    df['season'] = df['season'].astype(int)  # Keep as int, not float

    # Create a date-time index
    time_vars = ['year', 'month', 'day', 'hour']
    time = pd.to_datetime(df[time_vars])
    df.index = time

    df['target'] = np.log1p(df['avg_pm'])

    # Construct Features
    drop_vars = ['year', 'month', 'day', 'hour']
    cat_feats = ['cbwd']  # ⛔ Removed 'month', 'day', 'hour' and especially 'season'

    for f in cat_feats:
        df[f] = pd.Categorical(df[f])

    X = df.copy()
    X = X.drop(drop_vars, axis=1)
    X = X.drop(['target', 'avg_pm'], axis=1)

    # Rename columns
    X = X.rename(columns={'DEWP': 'DewPt', 
                          'HUMI': 'Humidity', 
                          'PRES': 'Press', 
                          'TEMP': 'TempC', 
                          'cbwd': 'WindDir', 
                          'Iws': 'WindSp', 
                          'season': 'Season',
                          'precipitation': 'PrecipHr', 
                          'Iprec': 'PrecipCm'})

    # One-hot encode only categorical variables (e.g. WindDir), NOT Season
    X = pd.get_dummies(X, columns=['WindDir'], drop_first=True)

    y = df['target']
    Xy = pd.concat([X, y], axis=1)

    # Define numeric features excluding one-hot and time vars
    numeric_feats = [f for f in X.columns if 'WindDir' not in f and f not in time_vars]

    # Engineering: Precipitation Difference
    X['PrecipCm'] = X['PrecipCm'] - X['PrecipHr']

    # Log transform skewed numeric features
    skewed_feats = X[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > 0.75].index
    X[skewed_feats] = np.log1p(X[skewed_feats])

    return df, X, y
