import datetime
import glob
import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from openpyxl import load_workbook
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

task = Task(TaskTypesEnum.regression)
# path = "C:\\Users\Daniil\PycharmProjects\AutoM\save_results"
# path = 'C:/Users/Daniil/Downloads/chess_2560006508.csv'
# path = 'C:/Users/Daniil/Downloads/chess_2560604600.csv'
# path = 'C:/Users/Daniil/Downloads/chess_2560610300.csv'
# path = 'C:/Users/Daniil/Downloads/chess_2560616000.csv'
# path = 'C:/Users/Daniil/Downloads/chess_2560617200.csv'
path = 'H://Ucheba/Poly_3_kurs/3_Kurs_MexMatMod/FabLab_oil_and_gas/AutoML/task-AutoML/packed_data_for_students/chess_2.csv'

path_results_liq = 'H://Ucheba/Poly_3_kurs/3_Kurs_MexMatMod/FabLab_oil_and_gas/AutoML/result_regression_liq'
path_results_oil = 'H://Ucheba/Poly_3_kurs/3_Kurs_MexMatMod/FabLab_oil_and_gas/AutoML/result_regression_oil'

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


csvfiles = glob.glob(os.path.join("./packed_data_for_students", '*.csv'))



df = pd.read_csv(path)
feature_train = np.array([[i] for i in train_period(path).p.values])
target_train = train_period(path).ql_m3_fact.values

feature_validation = np.array([[i] for i in validation_period(path).p.values])
target_validation = validation_period(path).ql_m3_fact.values

print()
print(path)


input_data = InputData(idx=np.arange(0, len(feature_train)), features=feature_train,
                       target=target_train, task=task,
                       data_type=DataTypesEnum.table)

input_data1 = InputData(idx=np.arange(0, len(feature_validation)), features=feature_validation,
                       target=target_validation, task=task,
                       data_type=DataTypesEnum.table)


fedot_model = Fedot(problem='regression', learning_time=1,
                    seed=42, verbose_level=1)

pipeline = fedot_model.fit(features=input_data)
pipeline.show()

prediction = fedot_model.predict(features=input_data)
prediction1 = fedot_model.predict(features=input_data1)


mse_value = mean_squared_error(target_validation, prediction1)

print(f'Mean squared error - {mse_value:.4f}\n')




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
#здесь 3 строчки поменять на csvfile
date_to_pred2 = validation_period(path).ql_m3_fact
date_to_pred = validation_period(path).dt
date_to_pred1 = train_period(path).dt
date_to_line = train_period(path).dt.values[-1]
q_to_pred = pd.DataFrame(prediction1, columns=['q'])
q_to_pred1 = pd.DataFrame(prediction, columns=['q'])


trace = go.Scatter(name='true', x=df.dt, y=df.ql_m3_fact, mode=m, marker=mark,  marker_color='rgb(0,0,255)')
fig.add_trace(trace, row=1, col=1)

trace = go.Scatter(name='pred', x=date_to_pred1, y=q_to_pred1.q, mode=ml, marker=mark, line=line, marker_color='rgb(255,165,0)')
fig.add_trace(trace, row=1, col=1)

trace = go.Scatter(name='pred', x=date_to_pred, y=q_to_pred.q, mode=ml, marker=mark, line=line, marker_color='rgb(255,165,0)')
fig.add_trace(trace, row=1, col=1)

trace = go.Scatter(name='pred', x=date_to_pred, y=q_to_pred.q, mode=ml, marker=mark, marker_color='rgb(255,165,0)')
fig.add_trace(trace, row=1, col=2)

trace = go.Scatter(name='true', x=date_to_pred, y=date_to_pred2, mode=m, marker=mark,  marker_color='rgb(0,0,255)')
fig.add_trace(trace, row=1, col=2)

trace = go.Scatter(name='predictor', x=df.dt, y=df.p, mode=m, marker=mark)
fig.add_trace(trace, row=2, col=1)


trace = go.Scatter(name='dev', x=date_to_pred,
                   y=abs(validation_period(path).ql_m3_fact.values - prediction1) / validation_period(path).ql_m3_fact.values, mode=ml, marker=mark, line=line)
fig.add_trace(trace, row=2, col=2)

fig.add_vline(row=1, col=1, x=date_to_line, line_width=2, line_dash='dash')
fig.add_vline(row=2, col=1, x=date_to_line, line_width=2, line_dash='dash')
# figure.show()
save_name = os.path.split(path)[1].split(sep='_')[1].split(sep='.')[0] + '_liq.png'
fig.update_layout(title_text=save_name, title_x=0.5)
file = (os.path.join(path_results_liq, save_name))
# pio.write_image(fig, file=file, format='png', width=1450, height=700, scale=2, engine='kaleido')

figure.show()

