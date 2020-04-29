import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.impute
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

import env
import acquire

def fill_na(df):
    df.replace(to_replace = " ", value = np.nan, inplace = True)
    return df

def drop_na(df):
    df = df.dropna(axis = 0)
    return df

def phone_lines(df):
    phone_service = [0 if i == 'No' else 1 for i in df.phone_service]
    multiple_lines = [1 if i == 'Yes' else 0 for i in df.multiple_lines]
    df['phone_and_multi_line'] = [phone_service[i] + multiple_lines[i] for i in range(len(phone_service))]
    return df

def partner_dependents(df):
    partner_and_dependents = []
    
    for i in range(len(df.partner)):
        if df.partner[i] == 'No' and df.dependents[i] == 'No':
            partner_and_dependents.append(0)
        elif df.partner[i] == 'Yes' and df.dependents[i] == 'No':
            partner_and_dependents.append(1)
        elif df.partner[i] == "No" and df.dependents[i] == 'Yes':
            partner_and_dependents.append(2)
        elif df.partner[i] == 'Yes' and df.dependents[i] == 'Yes':
            partner_and_dependents.append(3)
    
    df['partner_and_dependents'] = partner_and_dependents
    return df

def drop_columns(df):
    return df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'partner', 'dependents', 'phone_service', 'tenure', 'multiple_lines'])

def X_label_encode(df):
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df.gender)
    df['online_security'] = le.fit_transform(df.online_security)
    df['online_backup'] = le.fit_transform(df.online_backup)
    df['device_protection'] = le.fit_transform(df.device_protection)
    df['tech_support'] = le.fit_transform(df.tech_support)
    df['streaming_tv'] = le.fit_transform(df.streaming_tv)
    df['streaming_movies'] = le.fit_transform(df.streaming_movies)
    df['paperless_billing'] = le.fit_transform(df.paperless_billing)
    df['churn'] = le.fit_transform(df.churn)
    return df

def y_label_encode(df):
    le = LabelEncoder()
    df['churn'] = le.fit_transform(df[["churn"]])
    return df

def one_hot_encoder(df):
    one_hot = OneHotEncoder(sparse = False)
    payment_encoded = one_hot.fit_transform(df[['payment_type']])
    payment_labels = list(df.payment_type.value_counts().sort_index().index)
    payment_encoded_df = pd.DataFrame(payment_encoded, columns = payment_labels, index = df.index)

    internet_encoded = one_hot.fit_transform(df[['internet_service_type']])
    internet_labels = list(df.internet_service_type.value_counts().sort_index().index)
    internet_encoded_df = pd.DataFrame(internet_encoded, columns = internet_labels, index = df.index)

    contract_encoded = one_hot.fit_transform(df[['contract_type']])
    contract_labels = list(df.contract_type.value_counts().sort_index().index)
    contract_encoded_df = pd.DataFrame(contract_encoded, columns = contract_labels, index = df.index)

    df = df.join([payment_encoded_df, internet_encoded_df, contract_encoded_df])

    return df

def drop_service_types(df):
    return df.drop(columns=['contract_type', 'internet_service_type', 'payment_type'])

def split_telco(df):
    '''
    Returns X and y for train, validate and test datasets
    '''
    # don't blow away our original data
    df = df.copy()
    
    df = fill_na(df)
    df = drop_na(df)
    df['total_charges'] = df['total_charges'].astype('float')
    df.set_index('customer_id', inplace=True)
    df['tenure_years'] = df.tenure/12
    df = phone_lines(df)
    df = partner_dependents(df)
    df = drop_columns(df)

    # Which features are we going to look at?
    # train = df
    # test = df

    # validate data split
    train, test = sklearn.model_selection.train_test_split(df, train_size=.80, random_state=123)
    train, validate = sklearn.model_selection.train_test_split(train, train_size=.80, random_state=123)

    # split into X and y
    X_train, y_train = train, train[["churn"]]
    X_validate, y_validate = validate, validate[["churn"]]
    X_test, y_test = test, test[["churn"]]
    
    X_train = X_label_encode(X_train)
    X_validate = X_label_encode(X_validate)
    X_test = X_label_encode(X_test)
    
    y_train = y_label_encode(y_train)
    y_validate = y_label_encode(y_validate)
    y_test = y_label_encode(y_test)

    X_train = one_hot_encoder(X_train)
    X_validate = one_hot_encoder(X_validate)
    X_test = one_hot_encoder(X_test)
    
    X_train = drop_service_types(X_train)
    X_validate = drop_service_types(X_validate)
    X_test = drop_service_types(X_test)

    return X_train, y_train, X_validate, y_validate, X_test, y_test