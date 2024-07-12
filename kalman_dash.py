import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
import dash
from dash import dcc, html
import plotly.graph_objs as go


# 데이터 파일 읽기 및 보간 적용 함수들 (위에서 제공한 코드 사용)
def read_imu_data(file_path):
    imu_data = pd.read_csv(file_path, sep='\s+')
    imu_data['dt'] = imu_data['Time'] - imu_data['Time'].shift(1)
    imu_data = imu_data.iloc[1:].reset_index(drop=True)
    return imu_data


def read_gps_data(file_path):
    gps_data = pd.read_csv(file_path)
    gps_data['dt'] = gps_data['Time'] - gps_data['Time'].shift(1)
    gps_data = gps_data.iloc[1:].reset_index(drop=True)
    return gps_data


def interpolate_imu_data(gps_time, imu_data):
    if len(imu_data) < 2:
        return None

    imu_data_times = imu_data['Time'].values
    idx = np.searchsorted(imu_data_times, gps_time, side='right')

    if idx == 0 or idx >= len(imu_data):
        return None

    imu_data_prev = imu_data.iloc[idx - 1]
    imu_data_next = imu_data.iloc[idx]

    t_prev = imu_data_prev['Time']
    t_next = imu_data_next['Time']
    ratio = (gps_time - t_prev) / (t_next - t_prev)

    interpolated_imu = {}
    interpolated_imu['Time'] = gps_time
    for key in imu_data_prev.keys():
        if key != 'Time' and key != 'dt':
            interpolated_imu[key] = imu_data_prev[key] + ratio * (imu_data_next[key] - imu_data_prev[key])

    return interpolated_imu


# 칼만 필터 적용 함수 (위에서 제공한 코드 사용)
def apply_kalman_filter(gps_data):
    kf = KalmanFilter(dim_x=9, dim_z=3)
    dt = 1.0  # 기본 시간 간격 (초)
    kf.F = np.array([[1, dt, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, dt, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, dt, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, dt, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, dt],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0, 0]])
    kf.x = np.zeros(9)
    kf.P *= 1000.
    kf.Q = np.eye(9) * 0.1
    kf.R = np.eye(3) * 5.

    filtered_data = []
    for i in range(len(gps_data)):
        dt = gps_data.iloc[i]['dt']
        kf.F[0, 1] = dt
        kf.F[3, 4] = dt
        kf.F[6, 7] = dt

        z = np.array([gps_data.iloc[i]['X'], gps_data.iloc[i]['Y'], gps_data.iloc[i]['Z']])
        kf.predict()
        kf.x[8] = -9.81  # Z 가속도 반영
        kf.update(z)

        filtered_data.append(kf.x.copy())
        print(f"시간 {gps_data.iloc[i]['Time']}의 상태 추정값: {kf.x}")

    filtered_df = pd.DataFrame(filtered_data, columns=['X', 'VX', 'Y', 'VY', 'Z', 'VZ', 'AX', 'AY', 'AZ'])
    return filtered_df


# 파일 경로 설정
imu_file_path = 'KittiEquivBiasedImu.txt'
gps_file_path = 'KittiGps_converted.txt'

# 데이터 읽기
imu_data = read_imu_data(imu_file_path)
gps_data = read_gps_data(gps_file_path)

# 보간된 IMU 데이터를 저장할 리스트
interpolated_imu_data = []

for i in range(len(gps_data)):
    gps_time = gps_data.iloc[i]['Time']
    interpolated_imu = interpolate_imu_data(gps_time, imu_data)
    if interpolated_imu is not None:
        interpolated_imu_data.append(interpolated_imu)

interpolated_imu_df = pd.DataFrame(interpolated_imu_data)
filtered_df = apply_kalman_filter(gps_data)
# Dash 앱 설정
app = dash.Dash(__name__)

frames = []
for i in range(len(gps_data)):
    frames.append(go.Frame(data=[
        go.Scatter3d(
            x=gps_data['X'][:i + 1],
            y=gps_data['Y'][:i + 1],
            z=gps_data['Z'][:i + 1],
            mode='markers+lines',
            name='Original GPS',
            marker=dict(size=2, color='blue')
        ),
        go.Scatter3d(
            x=filtered_df['X'][:i + 1],
            y=filtered_df['Y'][:i + 1],
            z=filtered_df['Z'][:i + 1],
            mode='markers+lines',
            name='Kalman Filtered',
            marker=dict(size=2, color='yellow')
        )
    ]))

app.layout = html.Div([
    dcc.Graph(
        id='3d-scatter-plot',
        figure={
            'data': [
                go.Scatter3d(
                    x=gps_data['X'],
                    y=gps_data['Y'],
                    z=gps_data['Z'],
                    mode='markers+lines',
                    name='Original GPS',
                    marker=dict(size=4, color='blue')
                ),
                go.Scatter3d(
                    x=filtered_df['X'],
                    y=filtered_df['Y'],
                    z=filtered_df['Z'],
                    mode='markers+lines',
                    name='Kalman Filtered',
                    marker=dict(size=4, color='yellow')
                )
            ],
            'layout': go.Layout(
                title='3D Visualization of GPS and Kalman Filtered Data',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=0.1)
                ),
                margin=dict(l=0, r=0, b=0, t=50),
                height=800,
                width=1000,
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True
                        }]
                    }]
                }]
            ),
            'frames': frames
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
