# This app is a get API that accepts requests from postman.

# The purpose of the API is to make future predictions for
# a given crypto ticker symbol.

# Code is not encapsulated, but it passes pycodestyle check

# libraries
from sanic import response, Sanic, json
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from autots import AutoTS
import plotly.express as px
from yahooquery import Screener
from itertools import cycle, islice
import kaleido
import os
import time


# defining the API
app = Sanic("CodeToAPI")
HOST = "localhost"
PORT = 8000


# this code routes the request to the run program function
@app.route("/run_program", methods=["GET"])
# this function is for receiving and returning the request
def run_program(request):
    remove_old_image()
    return execution(request)


# loading all available crypto ticker symbols
def load_crypto_tickers():
    s = Screener()
    data = s.get_screeners("all_cryptocurrencies_us", count="250")

    # data is in the quotes key
    dicts = data["all_cryptocurrencies_us"]["quotes"]
    symbols = [d["symbol"] for d in dicts]
    return symbols


# downloading the stock data
def stock_data_download(past_data_timeline, crypto_ticker):
    # getting current and past dates
    curr_date = date.today()
    past_date = date.today() - timedelta(days=past_data_timeline)

    # converting dates to strings
    end_date = curr_date.strftime("%Y-%m-%d")
    start_date = past_date.strftime("%Y-%m-%d")

    data = yf.download(crypto_ticker.upper(), start_date, end_date)

    # return the data frame
    return data


def create_chart(total_data):
    # charting the prices
    chart = px.line(total_data, x="Date", y="Close", color="Time")

    # displaying the chart
    return chart


# this function accepts a variable to be converted as a list
# then converts it to a string or int
def convert_list_to_string_int(var, string_or_int):
    if string_or_int == "int":
        converted_var = int(" ".join([str(index) for index in var]))
    else:
        converted_var = " ".join([str(index) for index in var])
    return converted_var


# this function validates the request variables
def validate_request(request):
    # defining variables
    available_cryptos = load_crypto_tickers()
    past_days_data_range = [10, 3650]
    prediction_num_days_range = [1, 1825]

    # converting the user input to correct variable types
    past_days_data = convert_list_to_string_int(
        request.args["past_days_data"], "int")
    prediction_num_days = convert_list_to_string_int(
        request.args["prediction_num_days"], "int"
    )
    crypto_ticker_symbol = convert_list_to_string_int(
        request.args["crypto_ticker_symbol"], "string"
    )

    # validating past_days_data
    past_days_data_validation = validate_user_data(
        past_days_data, past_days_data_range, "past", ""
    )
    if past_days_data_validation is False:
        return {
            "validation_status": False,
            "invalid_variable": "past_days_data",
            "invalid_variable_value": past_days_data,
            "invalid_variable_range": past_days_data_range,
        }

    # validating prediction_num_days
    prediction_num_days_validation = validate_user_data(
        prediction_num_days, prediction_num_days_range,
        "future", past_days_data
    )

    if prediction_num_days_validation is False:
        return {
            "validation_status": False,
            "invalid_variable": "prediction_num_days",
            "invalid_variable_value": prediction_num_days,
            "invalid_variable_range": prediction_num_days_range,
        }

    # validating crypto_ticker_symbol
    crypto_ticker_symbol = crypto_ticker_symbol.upper()
    if crypto_ticker_symbol not in available_cryptos:
        return {
            "validation_status": False,
            "invalid_variable": "crypto_ticker_symbol",
            "invalid_variable_value": crypto_ticker_symbol,
            "invalid_variable_range": available_cryptos,
        }

    # all request variables are validated
    else:
        return {
            "validation_status": True,
            "past_days_data": past_days_data,
            "prediction_num_days": prediction_num_days,
            "crypto_ticker_symbol": crypto_ticker_symbol,
        }


# this function takes an input and checks if it lies in an
# acceptable range.
def validate_user_data(user_input, range,
                       past_or_fut, past_data_timeline):
    if user_input >= range[0] and user_input < range[1] \
            and past_or_fut == "past":
        return user_input
    elif user_input >= range[0] and user_input < range[1] \
            and past_or_fut == "future":
        if past_data_timeline // 2 > user_input:
            return user_input
        else:
            return False
    else:
        return False


# this function creates and prepares data frames
def prepping_data_frame(data_frame, time_tense, past_data_timeline):

    # resetting the index and renaming  index to Date for past prices DF
    data_frame = data_frame.reset_index(drop=False)
    data_frame.rename(columns={"index": "Date"}, inplace=True)

    # Time column used to plot past prices and future prices on graph
    time_column = [time_tense]

    if time_tense == "past":
        # below line creates 366 array with the word "past" +1
        time_column = list(islice(cycle(time_column), past_data_timeline))

        # Appending new column called "Time" to the data frame
        data_frame["Time"] = time_column

        # Adding a duplicate entry to DF for the last date: Time = 'future'
        # This is necessary for a line graph with no gaps in dates
        start_of_future = data_frame.loc[(past_data_timeline - 1)]
        start_of_future["Time"] = "future"
        data_frame = data_frame.append(start_of_future, ignore_index=True)

    else:
        # below line creates 366 array with the word "past"
        time_column = list(islice(cycle(time_column), past_data_timeline))

        # Appending new column called "Time" to the data frame
        data_frame["Time"] = time_column

    return data_frame


# this function runs the ML time series model
def run_prediction(data_frame, future_days_desired):
    # creating time series model
    model = AutoTS(
        forecast_length=future_days_desired,
        frequency="infer",
        ensemble="simplet",
        max_generations=1,
        no_negatives=True,
        # num_validations=3,
    )

    # entering our data parameters for the time series model
    model = model.fit(data_frame, date_col="Date",
                      value_col="Close", id_col=None)

    # forecasting the model
    prediction = model.predict()
    forecast = prediction.forecast

    return forecast


# this function gathers crypto price data from yahoo.com
def gather_past_and_future_data(past_data_timeline,
                                future_days_desired, crypto_ticker):
    # download the stock data
    # length 30
    past_data = stock_data_download(past_data_timeline, crypto_ticker)

    # prep past_data
    past_data = prepping_data_frame(past_data, "past", past_data_timeline)

    # get forecast
    future_data = run_prediction(past_data, future_days_desired)

    future_data = prepping_data_frame(future_data,
                                      "future", future_days_desired)

    total_data = past_data.append(future_data)

    # appending the prediction data frame to the past prices data frame
    return total_data


# this funciton creates a chart with given data
def write_chart_to_pdf(chart):
    curr_dir = os.getcwd()
    chart.write_image(curr_dir + "\chart.pdf")
    return curr_dir + "\chart.pdf"


def remove_old_image():
    os.remove(os.getcwd() + "\chart.pdf")


# this function serves as the execution layer of the code,
# middleman for executing logic to determine responses
def execution(request):
    # call function to validate request
    validation = validate_request(request)

    # request validation failed, return http 400 response
    if validation["validation_status"] is False:
        return http_response(
            "400",
            validation["invalid_variable"],
            validation["invalid_variable_value"],
            validation["invalid_variable_range"],
        )

    total_data = gather_past_and_future_data(
        validation["past_days_data"],
        validation["prediction_num_days"],
        validation["crypto_ticker_symbol"],
    )
    chart = create_chart(total_data)
    pdf_path = write_chart_to_pdf(chart)

    time.sleep(10)

    return response.file(pdf_path)

    # crypto_ticker = validate_crypto(crypto_ticker, crypto_dict)


# this function sets the http response for errors
def http_response(error_code, bad_data_variable,
                  bad_data_variable_value, range):
    # past_days_data and prediction_num_days error handling
    if error_code == "400" and bad_data_variable != "crypto_ticker_symbol":
        return json(
            {
                "message": "Invalid data. User sent: "
                           f"{bad_data_variable_value}, "
                           f"for {bad_data_variable}."
                f" Data range is {range[0]} => {range[1]}."
            },
            status=400,
        )

    # crypto_ticker_symbol error handling
    if error_code == "400" and bad_data_variable == "crypto_ticker_symbol":
        return json(
            {
                "message": "Invalid data. User sent: "
                           f"{bad_data_variable_value}, "
                           f"for {bad_data_variable}."
                f" Acceptable ticker symbols are: {range}."
            },
            status=400,
        )


# this code runs the app locally
if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=True)
