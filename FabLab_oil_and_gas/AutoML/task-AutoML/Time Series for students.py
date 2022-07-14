import glob
import os
import warnings
import datetime
import plotly.io as pio
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.composer.visualisation import ChainVisualiser
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from openpyxl import load_workbook
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_percentage_error

path_results_liq = "H://Ucheba/Poly_3_kurs/3_Kurs_MexMatMod/FabLab_oil_and_gas/AutoML/result_timeseries_liq_for_students"
path_results_oil = "H://Ucheba/Poly_3_kurs/3_Kurs_MexMatMod/FabLab_oil_and_gas/AutoML/result_timeseries_oil"
warnings.filterwarnings("ignore")

# функции преобразования времени и дат
def plot_results(actual_time_series, predicted_values, len_train_data, y_name='ql_m3_fact'):
    """
    Function for drawing plot with predictions

    :param actual_time_series: the entire array with one-dimensional data
    :param predicted_values: array with predicted values
    :param len_train_data: number of elements in the training sample
    :param y_name: name of the y axis
    """
    plt.figure(figsize=(28, 10), dpi=80)
    plt.plot(np.arange(0, len(actual_time_series)),
             actual_time_series, label='Actual values', c='green')
    plt.plot(np.arange(len_train_data, len_train_data + len(predicted_values)),
             predicted_values, label='Predicted', c='blue')
    #
    # plt.plot([len_train_data, len_train_data],
    #          [min(actual_time_series), max(actual_time_series)], c='black', linewidth=1)
    plt.ylabel(y_name, fontsize=15)
    plt.xlabel('Time index', fontsize=15)
    plt.legend(fontsize=15, loc='upper left')
    plt.grid()
    # save_name = os.path.split(csvfile)[1].split(sep='_')[1].split(sep='.')[0] #+ str(i)
    # plt.title('скважина  '+ save_name)
    # plt.savefig(os.path.join(path_results, save_name))
    # plt.clf()

    plt.show()
# path = 'C:/Users/Daniil/Downloads/chess_2560006508.csv'
# path = 'C:/Users/Daniil/Downloads/chess_2560604600.csv'
# path = 'C:/Users/Daniil/Downloads/chess_2560610300.csv'
# path = 'C:/Users/Daniil/Downloads/chess_2560616000.csv'
# path = 'C:/Users/Daniil/Downloads/chess_2560617200.csv'
def convert_day_date(x: str) -> datetime.date:
    return datetime.datetime.strptime(x, '%Y-%m-%d').date()
def train_period(path, ndays=90):
    df = pd.read_csv(path)
    df = df.drop(columns=['wc_fact', 'status', 'event',  'work_time'])
    df = df.dropna()
    def convert_day_time(x: str) -> datetime.date:
        return datetime.datetime.strptime(x, '%Y-%m-%d')

    for i in df.dt:
        df.dt[df.dt[df.dt == i].index] = convert_day_time(i)

    return df[df.dt < df.dt.values[-1] - datetime.timedelta(days=ndays)].reset_index(drop=True)
def validation_period(path, ndays=90):
    df = pd.read_csv(path)
    df = df.drop(columns=[ 'wc_fact', 'status', 'event',  'work_time'])
    df = df.dropna()
    def convert_day_time(x: str) -> datetime.date:
        return datetime.datetime.strptime(x, '%Y-%m-%d')

    for i in df.dt:
        df.dt[df.dt[df.dt == i].index] = convert_day_time(i)

    return df[df.dt > df.dt.values[-1] - datetime.timedelta(days=ndays)].reset_index(drop=True)

# csvfile = glob.glob(os.path.join("./packed_data_for_students", '*.csv'))
csvfile = 'H://Ucheba/Poly_3_kurs/3_Kurs_MexMatMod/FabLab_oil_and_gas/AutoML/task-AutoML/packed_data_for_students/chess_2.csv'


#расчёт дебита жидкости
df = pd.read_csv(csvfile)
df = df.drop(columns=['wc_fact', 'status', 'event', 'work_time'])
df = df.dropna()

forecast_length = 90
x_train = df.p.values[:-forecast_length]  # две фичи
x_test = df.p.values[:-forecast_length]

x_len = train_period(csvfile).ql_m3_fact.values

y_train = df.p.values[:-forecast_length]  # два таргета
y_test = df.ql_m3_fact.values[-forecast_length:]
print()

task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=forecast_length))
print(csvfile)

traffic = np.array(df['ql_m3_fact'])
traffic1 = np.array(df['p'])
input_data = InputData(idx=np.arange(0, len(traffic)),
                       features=traffic,
                       target=traffic1,
                       task=task,
                       data_type=DataTypesEnum.ts)
train_data = InputData(idx=np.arange(0, len(x_train)),
                       features=x_train,
                       target=y_train,
                       task=task,
                       data_type=DataTypesEnum.ts)

start_forecast = len(x_train)
end_forecast = start_forecast + forecast_length
idx_for_predict = np.arange(start_forecast, end_forecast)

test_data = InputData(idx=idx_for_predict,
                       features=x_test,
                       target=y_test,
                       task=task,
                       data_type=DataTypesEnum.ts)

train_input, predict_input = train_test_data_setup(input_data, split_ratio=0.7, shuffle_flag=False)
task_parameters = TsForecastingParams(forecast_length=forecast_length)

castom_params = params = {'max_depth': 4,
                          'max_arity': 3,
                          'pop_size': 50,
                          'num_of_generations': 50,
                          'learning_time': 1,
                          'preset': 'light_tun'}

model = Fedot(problem='ts_forecasting', learning_time=1, seed=42, verbose_level=1, task_params=task_parameters, composer_params=castom_params)
chain = model.fit(features=train_data)
chain.show()
forecast = model.predict(features=train_data)
forecast1 = model.predict(features=test_data)
# visualiser = ChainVisualiser()
# visualiser.visualise(chain)
# отрисовка дебита жидкости
figure = go.Figure(layout=go.Layout(
    font=dict(size=10),
    hovermode='x',
    template='seaborn',
))
fig = make_subplots(
    rows=2,
    cols=2,
    shared_xaxes=True,
    column_width=[0.7, 0.3],
    row_heights=[0.7, 0.3],
    vertical_spacing=0.02,
    horizontal_spacing=0.02,
    figure=figure,
)
m = 'markers'
ml = 'markers+lines'
mark = dict(size=3)
line = dict(width=1)
# здесь 3 строчки поменять на csvfile
date_to_pred2 = validation_period(csvfile).ql_m3_fact
date_to_pred = validation_period(csvfile).dt
date_to_pred1 = train_period(csvfile).dt
date_to_line = train_period(csvfile).dt.values[-1]
q_to_pred = pd.DataFrame(forecast1, columns=['q'])
q_to_pred1 = pd.DataFrame(forecast, columns=['q'])
len_train_data = len(traffic) - forecast_length

trace = go.Scatter(name='true', x=df.dt, y=traffic, mode=m, marker=mark, marker_color='rgb(0,0,255)')
fig.add_trace(trace, row=1, col=1)

trace = go.Scatter(name='pred', x=validation_period(csvfile).dt, y=forecast1, mode=ml, marker=mark, line=line,
                   marker_color='rgb(255,165,0)')
fig.add_trace(trace, row=1, col=1)

# trace = go.Scatter(name='pred', x=np.arange(0, len_train_data), y=forecast, mode=ml, marker=mark, line=line,
#                    marker_color='rgb(255,165,0)')
# fig.add_trace(trace, row=1, col=1)
#
trace = go.Scatter(name='pred', x=validation_period(csvfile).dt, y=forecast1, mode=ml, marker=mark, marker_color='rgb(255,165,0)')
fig.add_trace(trace, row=1, col=2)
#
trace = go.Scatter(name='true', x=validation_period(csvfile).dt, y=validation_period(csvfile).ql_m3_fact.values, mode=m, marker=mark, marker_color='rgb(0,0,255)')
fig.add_trace(trace, row=1, col=2)
#
trace = go.Scatter(name='predictor', x=df.dt, y=df.p, mode=m, marker=mark)
fig.add_trace(trace, row=2, col=1)

trace = go.Scatter(name='dev', x=validation_period(csvfile).dt, y=abs(y_test - forecast1) / y_test, mode=ml, marker=mark, line=line)
fig.add_trace(trace, row=2, col=2)

# trace = go.Figure(go.Image(fig1))
# fig.add_trace(trace, row=2, col=2)

fig.add_vline(row=1, col=1, x=train_period(csvfile).dt.values[-1], line_width=2, line_dash='dash')
fig.add_vline(row=2, col=1, x=train_period(csvfile).dt.values[-1], line_width=2, line_dash='dash')
figure.show()

save_name = os.path.split(csvfile)[1].split(sep='_')[1].split(sep='.')[0] + '_liq.png'
fig.update_layout(title_text=save_name, title_x=0.5)
file = (os.path.join(path_results_liq, save_name))
# pio.write_image(fig, file=file, format='png', width=1450, height=700, scale=2, engine='kaleido')




