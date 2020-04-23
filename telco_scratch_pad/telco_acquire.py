import pandas as pd
import numpy as np

from env import host, user, password

def get_connection(db, user=user, host=host, password=password):
    return f"mysql+pymysql://{user}:{password}@{host}/{db}"

def get_telco_basic():
    sql_query = "SELECT * FROM customers WHERE churn = 'Yes'"
    return pd.read_sql(sql_query, get_connection("telco_churn"))