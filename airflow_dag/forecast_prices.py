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
from sklearn.preprocessing import MinMaxScaler
import pickle
import pmdarima as pm
from tqdm.notebook import tnrange
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout , LSTM , Bidirectional

table_id_target = 'latihan-345909.stocks.prediction_table'
credentials = service_account.Credentials.from_service_account_file(r"/home/credentials.json")
project_id = 'latihan-345909'
table_id = 'latihan-345909.stocks.stocks_table'
client = bigquery.Client(credentials=credentials, project=project_id)

def extract_data(name, client):
    sql = """
        WITH latest_date AS (
        SELECT MAX(jakarta_data_date) as max_date
        FROM `latihan-345909.stocks.stocks_table`
        )

        SELECT *
        FROM latihan-345909.stocks.stocks_table
        WHERE jakarta_data_date = (SELECT max_date FROM latest_date)
            AND company_name = "{0}"
    """.format(name)

    bq_df = client.query(sql).to_dataframe()

    return bq_df

def load_object(name : str):
    pickle_in = open(f"{name}","rb")
    data = pickle.load(pickle_in)
    return data

def data_transformation(df, scaler):
  try:
    data = pd.DataFrame(np.squeeze(scaler.transform(df), axis=1), columns=df.columns, index=df.index)
  except:
    data = pd.DataFrame(np.squeeze(scaler.transform(df.values.reshape(-1, 1)), axis=1), columns=[df.name], index=df.index)
  return data


def sarima_forecast(model, df, forecast_date, scaler):
    # Forecast
    n_periods = (datetime.strptime(forecast_date, '%Y-%m-%d') - df.index[-1]).days
    fitted, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = pd.date_range(df.index[-1] + pd.DateOffset(days=1), periods = n_periods, freq='D')

    # make series for plotting purpose
    fitted_series = pd.Series(fitted.values, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    df_result = pd.concat([df, fitted_series, lower_series, upper_series], axis=1)
    df_result.columns = ["Actual", "Prediction", "Low", "High"]

    for column in df_result.columns:
      df_result[column] = scaler.inverse_transform(df_result[column].values.reshape(-1,1))


    return df_result


def create_lstm_model():
  model = Sequential()

  model.add(Bidirectional(LSTM(512 ,return_sequences=True , recurrent_dropout=0.1, input_shape=(20, 1))))
  model.add(LSTM(256 ,recurrent_dropout=0.1))
  model.add(Dropout(0.2))
  model.add(Dense(64 , activation='elu'))
  model.add(Dropout(0.2))
  model.add(Dense(32 , activation='elu'))
  model.add(Dense(1 , activation='linear'))

  optimizer = tf.keras.optimizers.SGD(learning_rate = 0.002)
  model.compile(loss='mse', optimizer=optimizer, metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])

  return model


def load_weight_model(data_transformed):
    test_data = data_transformed.iloc[-21:-1].values.reshape(1,20,1)
    target_data = data_transformed.iloc[-1].values.reshape(1,1)

    lstm_model = create_lstm_model()

    lstm_model.evaluate(test_data, target_data)

    lstm_model.load_weights("/home/dana123/airflow/forecast_model/best_weights_lstm_model.h5")

    return lstm_model

def PredictStockPriceLSTM(Model , df , ForecastDate , scaler, feature_length = 20):
    for i in range((datetime.strptime(ForecastDate, '%Y-%m-%d') - df.index[-1]).days):
      Features = df.iloc[-20:].values.reshape(-1, 1)
      Prediction = Model.predict(Features.reshape(1,20,1))
      df_forecast = pd.DataFrame(Prediction, index=[df.index[-1]+ timedelta(days=1)], columns=['close'])
      df = pd.concat([df, df_forecast])
    df = pd.DataFrame(np.squeeze(scaler.inverse_transform(df)),
                      index=df.index, columns=['close'])
    return df

def combine_model(data_transformed, lstm_model, sarima_model, forecast_date, scaler):
  df_low = pd.DataFrame(columns=['Low'])
  df_high = pd.DataFrame(columns=["High"])
  last_initial_date = data_transformed.index[-1]

  for i in range((datetime.strptime(forecast_date, '%Y-%m-%d') - data_transformed.index[-1]).days):
    df_lstm_temp = PredictStockPriceLSTM(lstm_model, data_transformed, (data_transformed.index[-1]+ timedelta(days=1)).strftime("%Y-%m-%d"), scaler)
    df_sarima_temp = sarima_forecast(sarima_model, data_transformed, (data_transformed.index[-1]+ timedelta(days=i+1)).strftime("%Y-%m-%d"), scaler)

    prediction = scaler.transform(df_lstm_temp.iloc[-1]["close"].reshape(1,-1)) * 0.3 + scaler.transform(df_sarima_temp.iloc[-1]["Prediction"].reshape(1,-1)) *0.7

    df_forecast = pd.DataFrame(prediction,
                              index=[data_transformed.index[-1]+ timedelta(days=1)], columns=['close'])


    df_low = pd.concat([df_low, pd.DataFrame(df_sarima_temp.iloc[-1]["Low"],
                                          index=[data_transformed.index[-1]+ timedelta(days=1)],
                                          columns=["Low"])
    ])
    df_high = pd.concat([df_high, pd.DataFrame(df_sarima_temp.iloc[-1]["High"],
                                          index=[data_transformed.index[-1]+ timedelta(days=1)],
                                          columns=["High"])
    ])

    data_transformed = pd.concat([data_transformed, df_forecast])

  initial_prediction_date = str(last_initial_date.date()+ timedelta(days=1))
  df_final = pd.DataFrame(np.squeeze(scaler.inverse_transform(data_transformed)),
                        index=data_transformed.index, columns=['close'])
  df_final_actual = df_final[:initial_prediction_date]
  df_final_prediction = df_final[initial_prediction_date:]

  return df_final_actual, df_final_prediction, df_low,  df_high


# Define DAG function
@task()
def forecast_price(name, forecast_date, client):
    data = extract_data(name, client)
    data.set_index('date', inplace=True)

    scaler = load_object("/home/dana123/airflow/forecast_model/scaler.pkl")
    sarima_model = load_object("/home/dana123/airflow/forecast_model/sarima_model.pkl")
    
    data_transformed = data_transformation(data["close"], scaler)
    
    lstm_model = load_weight_model(data_transformed)

    df_final_actual, df_final_prediction, df_low, df_high = combine_model(data_transformed, lstm_model, sarima_model, forecast_date, scaler)

    df_result = pd.concat([df_final_prediction, df_low, df_high], axis=1)
    df_result.columns = ["Prediction", "Low", "High"]
    df_result['jakarta_data_date'] = datetime.now(pytz.timezone("Asia/Jakarta")).date().strftime('%Y-%m-%d')
    df_result["jakarta_data_date"] = df_result["jakarta_data_date"].apply(pd.to_datetime)
    df_result["company_name"] = name
    df_result.reset_index(names=['forecast_date'], inplace=True)
    df_result["forecast_date"] = df_result["forecast_date"].apply(pd.to_datetime)

    columns_order = ['jakarta_data_date', 'company_name', 'Prediction', 'High', 'Low', 'forecast_date']
    df_result = df_result[columns_order]
    df_result.columns= df_result.columns.str.lower()

    job_config = bigquery.job.LoadJobConfig()
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
        
    job = client.load_table_from_dataframe(df_result, table_id_target, job_config=job_config)
    job.result()


# Declare Dag
with DAG(dag_id='forecast_stock_price',
         schedule_interval="0 0 15 * *",
         start_date=datetime(2023, 7, 1),
         catchup=False,
         tags=['stocks_etl'])\
        as dag:

    forecast_stock_price = forecast_price("GOOG", "2023-08-14", client)

    forecast_stock_price