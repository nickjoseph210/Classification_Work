import pandas as pd
import numpy as np
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.impute
import acquire

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from env import host, user, password

def get_connection(db, user=user, host=host, password=password):
    return f"mysql+pymysql://{user}:{password}@{host}/{db}"

def get_telco_data():
    """
    Function to pull the needed data from the telco_churn database
    """
    sql_query = """SELECT * 
    FROM customers 
    JOIN contract_types 
    USING (contract_type_id)
    JOIN internet_service_types
    USING (internet_service_type_id)
    JOIN payment_types
    USING (payment_type_id)
    """
    return pd.read_sql(sql_query, get_connection("telco_churn"))

# Following code is an amalgamation of code from acquire on 'Classificaton - Data Prep' exercise, and Chase's master-level understanding of it all

def fill_na(df):
    """
    Replace spaces with NaN values
    """
    df.replace(to_replace = " ", value= np.nan, inplace = True)
    return df

def drop_na(df):
    """
    Drops rows with NaN values
    """
    df = df.dropna(axis=0)
    return df

def phone_lines(df):
    """
    Converts 'Yes' and 'No' to numerical values in df columns 'phone_service' and 'multiple_lines', then add 'phone_and_multi_line' column to df
    """
    phone_service = [0 if i == "No" else 1 for i in df.phone_service]
    multiple_lines = [1 if i == "Yes" else 0 for i in df.multiple_lines]
    df['phone_and_multi_line'] = [phone_service[i] + multiple_lines[i] for i in range(len(phone_service))]
    return df

def partner_dependents(df):
    """
    Converts 'Yes' and 'No' to numerical values from df columns 'partner' and 'dependents,' then creates a new df column 'partner_and_dependents'
    """
    partner_and_dependents = []# empty list to catch loop answers

    for i in range(len(df.partner)):
        if df.partner[i] == "No" and df.dependents[i] == "No":
            partner_and_dependents.append(0)
        elif df.partner[i] == "Yes" and df.dependents[i] == "No":
            partner_and_dependents.append(1)
        elif df.partner[i] == "No" and df.dependents[i] == "Yes":
            partner_and_dependents.append(2)
        elif df.partner[i] == "Yes" and df.dependents[i] == "Yes":
            partner_and_dependents.append(3)
    df["partner_and_dependents"] = partner_and_dependents
    return df

def drop_columns(df):
    """
    Drops the columns from which we aggregated the data in 'partner_dependents,' as well as columns we felt indeterminate in customer churn rate
    """
    return df.drop(columns=["payment_type_id", "internet_service_type_id", "contract_type_id", "partner", "dependents", "phone_service", "tenure", "multiple_lines"])

def X_label_encode(df):
    """
    LabelEncodes all the object values in the listed columns to numerical, assigning weights to them as they have been deemed possible predictors in customer churn rate
    """
    le = LabelEncoder()
    df["online_security"] = le.fit_transform(df.online_security)
    df["online_backup"] = le.fit_transform(df.online_backup)
    df["device_protection"] = le.fit_transform(df.device_protection)
    df["tech_support"] = le.fit_transform(df.tech_support)
    df["streaming_tv"] = le.fit_transform(df.streaming_tv)
    df["streaming_movies"] = le.fit_transform(df.streaming_movies)
    df["paperless_billing"] = le.fit_tranform(df.paperless_billing)
    df["gender"] = le.fit_transform(df.gender)
    return df

def y_label_encoder(df):
    """
    Encodes our target, 'churn', with LabelEncoder
    """
    le = LabelEncoder()
    df["churn"] = le.fit_transform(df.churn)
    return df

def one_hot_encoder(df):
    """
    Encodes objects into integers while minimizing their magnitude
    """
    one_hot = OneHotEncoder(categories = "auto", sparse = False) # 'auto'matically determines the categories from the trained data; do NOT want to return a 'sparse' matrix
    payment_encoded = one_hot.fit_transform(df[["payment_type"]]) # fit & transforms the column 'payment_type'
    payment_labels = list(np.array(df.payment_type.value_counts().index))
    payment_encoded_df = pd.DataFrame(payment_encoded, columns = payment_labels, index = df.index)

    internet_encoded = one_hot.fit_transform(df[["internet_service_type"]])
    internet_labels = list(df.internet_service_type.value_counts().sort_index().index)
    internet_encoded_df = pd.DataFrame(internet_encoded, columns = internet_labels, index = df.index)

    contract_encoded = one_hot.fit_transform(df[["contract_type"]])
    contract_labels = list(df.contract_type.value_counts().sort_index().index)
    conctract_encoded_df = pd.DataFrame(contract_encoded, columns = contract_labels, index = df.index)

    df = df.join([payment_encoded_df, internet_encoded_df, contract_encoded_df])
    
    return df

def drop_service_type(df):
    """
    drops the original columns we one-hot encoded - if we didn't do this, the df would grow another three columns (each encoded column becomes a column itself)
    """
    return df.drop(columns=["contract_type", "internet_service_type", "payment_type"])

def split_telco(df):
    """
    Function to return our X and y for train, validate, and test datasets
    """
    # This is to keep our original df as is.  W/o 'copy,' the original changes
    df = df.copy()

    # fill na's
    df = fill_na()

    # drop na's
    df = drop_na()

    # makes sure all the values in the column 'total_charges' are now in 'float' datatype
    df["total_charges"] = df["total_charges"].astype("float") 

    # sets the df index to customer_id.  This will relieve confusion throughout the process, as the unique customer id for this column is easily understood by technical and non-technical audiences
    df.set_index("customer_id", inplace = True) 

    # converts the values in 'tenure_years' from months to years
    df["tenure_years"] = df.tenure / 12

    df = phone_lines(df)
    df = partner_dependents(df)
    df = drop_columns(df)

    # to determine which features we're going to look at, train = df; test = df

    # data split
    train, test = sklearn.model_selection.test_train_split(df, trian_size=.80, random_state = 123)

    # data split validation
    train, validate = sklearn.model_selection.train_test_split(train, train_size = .80, random_state = 123)

    # split into X and y:
    X_train, y_train = train.drop(columns="churn"), train[["churn"]]
    X_validate, y_validate = validate.drop(columns="churn"), validate[["churn"]]
    X_test, y_test = test.drop(columns="churn"), test[["churn"]]

    # LabelEncoding X_train, X_validate, and X_test
    X_train = X_label_encode(X_train)
    X_validate = X_label_encode(X_validate)
    X_test = X_label_encode(X_test)

    # LabelEncoding y_train, y_validate, and y_test
    y_train = y_label_encode(y_train)
    y_validate = y_label_encode(y_validate)
    y_test = y_label_encode(y_test)

    # OneHotting X_train, X_validate, and X_test
    X_train = one_hot_encoder(X_train)
    X_validate = one_hot_encoder(X_validate)
    X_test = one_hot_encoder(X_test)

    # OneHotting y_train, y_validate, and y_test
    y_train = one_hot_encoder(y_train)
    y_validate = one_hot_encoder(y_validate)
    y_test = one_hot_encoder(y_test)

    # put this whole thing to work:
    return X_train, y_train, X_validate, y_validate, X_test, y_test








