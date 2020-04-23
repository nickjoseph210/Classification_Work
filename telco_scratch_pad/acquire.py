# database connection

import pandas as pd
import numpy as np
import scipy as sp 

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

from env import host, user, password

def get_connection(db, user=user, host=host, password=password):
    return f"mysql+pymysql://{user}:{password}@{host}/{db}"

# getting titanic data as dataframe

def get_titanic_data():
    sql_query = "SELECT * FROM passengers"
    return pd.read_sql(sql_query, get_connection("titanic_db"))

# getting all the data from iris db

def get_iris_data():
    sql_query = """
    SELECT species_id,
    species_name,
    sepal_length, 
    sepal_width, 
    petal_length,
    petal_width
    FROM measurements
    JOIN species
    USING (species_id)
    """
    return pd.read_sql(sql_query, get_connection("iris_db"))

#  From Faith's Walkthrough - STUDY!
# def prep_titanic(df):
#     def prep_titanic(df):
#     # drop the deck column bc most values Null
#     drop_columns(df):
    
#     train, test = train_test_split(df, train_size=.75, stratify=titanic_df.survived, random_state=123)
    
#     # impute 2 NaNs in embark_town with most frequent value
#     train, test = impute_embark_town(train, test)
    
#     # impute 2 NaNs in embarked with most frequent value
#     train, test = impute_embarked(train, test)
    
#     # impute NaNs in age in train and test with the mean age in train
#     train, test = impute_age(train, test)
    
#     # use a minmax scaler on age and fare bc of differing measurement units
#     scaler, train, test = scale_columns(train, test)
    
#     # ohe embarked creating three new columns for C, Q, S representing embark towns
#     ohe, train, test = ohe_columns(train, test)
    
#     return scaler, ohe, train, test

def encode(train, test, col_name):
    """
    Encoder function that LabelEncodes and OneHots strings, turning them 
    into arrays, turns those arrays into a new df where column_names are 
    values, the index matches that of train / test, and then merges this new
    df with the existing train / test df.
    """
    encoded_values = sorted(list(train[col_name].unique()))

    # Integer Encoding
    int_encoder = LabelEncoder()
    train.encoded = int_encoder.fit_transform(train[col_name])
    test.encoded = int_encoder.transform(test[col_name])

    # Turns the integer encoding (LabelEncoder) values from 1-D to 2-D for ohe
    train_array = np.array(train.encoded).reshape(len(train.encoded), 1)
    test_array = np.array(test.encoded).reshape(len(test.encoded), 1)

    # One Hot Encoding
    ohe = OneHotEncoder(sparse=False, categories="auto")
    train_ohe = ohe.fit_transform(train_array)
    test_ohe = ohe.transform(test_array)

    # Turn the array of new values into a df with column names being 
    # the values and index matching that of train / test df then merging
    # this new df with the existing train / test df.
    train_encoded = pd.DataFrame(data=train_ohe, columns=encoded_values, index=train.index)

    train = train.join(train_encoded)

    test_encoded = pd.DataFrame(data=test_ohe, columns=encoded_values, index=test.index)

    test = test.join(test_encoded)
    
    return train, test

def impute(train, test, my_strategy, column_list):
    """
    Function to create the imputer object using whatever strategy I choose 
    ('mean,' 'most_frequent,' etc), then fits that imputed value to 
    the list of columns (column_list) in the training df.  
    """
    imputer = SimpleImputer(strategy=my_strategy)
    train[column_list] = imputer.fit_transform(train[column_list])
    test[column_list] = imputer.transform(test[column_list])
    return train, test_array

def get_telco_data():
    """
    Function to pull the needed data from the telco_churn database
    """
    sql_query = """SELECT customer_id, tenure, phone_service, 
    multiple_lines, internet_service_type_id, 
    streaming_tv, streaming_movies, monthly_charges,
    paperless_billing, payment_type_id, total_charges, churn
    FROM customers;"""
    return pd.read_sql(sql_query, get_connection("telco_churn"))