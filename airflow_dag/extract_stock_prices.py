import yfinance as yf
from datetime import datetime
from datetime import timedelta
from datetime import date
import pytz
import numpy as np
import pandas as pd
from stockstats import StockDataFrame as sdf
from google.cloud import bigquery
from google.oauth2 import service_account
from airflow.decorators import dag, task
from airflow.utils.task_group import TaskGroup
from airflow.models import DAG


# Define DAG function
@task()
def extract_load(name, start_date, stats_list):
    data_date = str(datetime.now(pytz.timezone("Asia/Jakarta")).date().strftime('%Y-%m-%d'))
    
    data = yf.download(name , start = start_date , interval = '1d')
    stock_df = sdf.retype(data)
    data[stats_list]=stock_df[stats_list]
    data.reset_index(inplace=True)
    data["company_name"] = name
    data["jakarta_data_date"] = data_date
    data["jakarta_data_date"] = data["jakarta_data_date"].apply(pd.to_datetime)
    data["Date"] = data["Date"].apply(pd.to_datetime)
    data.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume',
       'stochrsi', 'macd', 'macds', 'macdh', 'mfi', 'company_name',
       'jakarta_data_date']

    credentials = service_account.Credentials.from_service_account_file(r"/home/credentials.json")
    project_id = 'latihan-345909'
    table_id = 'latihan-345909.stocks.stocks_table'
    client = bigquery.Client(credentials=credentials, project=project_id)

    sql = """
    SELECT *
    FROM latihan-345909.stocks.stocks_table
    WHERE jakarta_data_date = '{0}'
    """.format(data_date)

    bq_df = client.query(sql).to_dataframe()

    if bq_df.empty:
        job = client.load_table_from_dataframe(data, table_id)
        job.result()
        print("There are {0} rows added/changed".format(len(data)))
    else:
        changes = data[~data.apply(tuple, 1).isin(bq_df.apply(tuple, 1))]
        job = client.load_table_from_dataframe(changes, table_id)
        job.result()
        print("There are {0} rows added/changed".format(len(changes)))

# Declare Dag
with DAG(dag_id='extract_and_load_stocks',
         schedule_interval="0 0 * * *",
         start_date=datetime(2023, 7, 1),
         catchup=False,
         tags=['stocks_etl'])\
        as dag:

    extract_and_load = extract_load("GOOG", "2018-01-01", ['stochrsi', 'macd', 'mfi'])

    extract_and_load