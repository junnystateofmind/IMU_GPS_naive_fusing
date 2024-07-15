import os
import pandas as pd
import utm
import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
import numpy as np
from filterpy.kalman import KalmanFilter
from argparse import ArgumentParser

# IMU 데이터 읽기 함수
def read_txt_files(directory):
    data = []
    for file in os.listdir(directory):
        if file.endswith('.txt') and file != 'timestamps.txt':
            with open(os.path.join(directory, file), 'r') as f:
                content = f.readlines()
                content = [line.strip().split() for line in content]
                data.append((file.split('.')[0], content))
    return data

# Extend Kalman Filter
dim_x = 15
dim_z = 3  # 추정 벡터는 3차원 (x, y, z) 위치

def F_matrix(dt):
    F = np.eye(dim_x)
    F[0, 3] = dt  # x 위치는 x 속도에 의존
    F[1, 4] = dt  # y 위치는 y 속도에 의존
    F[2, 5] = dt  # z 위치는 z 속도에 의존
    F[3, 6] = dt  # x 속도는 x 가속도에 의존
    F[4, 7] = dt  # y 속도는 y 가속도에 의존
    F[5, 8] = dt  # z 속도는 z 가속도에 의존
    F[9, 12] = dt  # 롤 각도는 롤 각속도에 의존
    F[10, 13] = dt  # 피치 각도는 피치 각속도에 의존
    F[11, 14] = dt  # 요 각도는 요 각속도에 의존
    return F

def apply_gravity(x):
    x[8] -= 9.81  # 중력 가속도 반영
    return x

H = np.zeros((dim_z, dim_x))
H[0, 0] = 1  # x 위치 관측
H[1, 1] = 1  # y 위치 관측
H[2, 2] = 1  # z 위치 관측

# 초기 상태 공분산 행렬 P
P = np.eye(dim_x) * 1000
# 프로세스 노이즈 행렬 Q
Q = np.eye(dim_x) * 0.1
# 측정 노이즈 행렬 R
R = np.eye(dim_z) * 5

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--testset", type=str, default='data', help="Path to the testset directory")
    args = parser.parse_args()

    directory = args.testset

    # 데이터 읽기
    data = read_txt_files(directory)

    # 빈 데이터프레임 생성
    df = pd.DataFrame()

    # 각 파일의 데이터를 데이터프레임으로 변환 후 병합
    for filename, content in data:
        temp_df = pd.DataFrame(content)
        temp_df['filename'] = filename
        df = pd.concat([df, temp_df], axis=0, ignore_index=True)

    # 열 이름 설정
    df.columns = ['lat', 'lon', 'alt', 'roll', 'pitch', 'yaw', 'vn', 've', 'vf', 'vl', 'vu', 'ax', 'ay', 'az', 'af',
                  'al', 'au', 'wx', 'wy', 'wz', 'wf', 'wl', 'wu', 'pos_accuracy', 'vel_accuracy', 'navstat', 'numsats',
                  'posmode', 'velmode', 'orimode', 'filename']

    # 숫자형으로 변환
    df[['lat', 'lon', 'alt', 'vn', 've', 'vu', 'ax', 'ay', 'az', 'roll', 'pitch', 'yaw', 'wx', 'wy', 'wz']] = df[
        ['lat', 'lon', 'alt', 'vn', 've', 'vu', 'ax', 'ay', 'az', 'roll', 'pitch', 'yaw', 'wx', 'wy', 'wz']].astype(
        float)

    # timestamp.txt를 읽고, 데이터프레임에 파일 이름 순서대로 추가
    timestamps_path = os.path.join(directory, 'timestamps.txt')
    if not os.path.exists(timestamps_path):
        raise FileNotFoundError(f"{timestamps_path} 파일을 찾을 수 없습니다.")

    timestamps = pd.read_csv(timestamps_path, header=None)

    # 타임스탬프를 데이터프레임에 추가
    df['timestamp'] = pd.to_datetime(np.tile(timestamps[0], len(df) // len(timestamps) + 1)[:len(df)])

    # 시간 간격 계산
    df['dt'] = df['timestamp'].diff().dt.total_seconds().fillna(0)

    # 데이터프레임 정렬
    df = df.sort_values(by=['filename', 'timestamp']).reset_index(drop=True)

    # UTM 좌표 변환 및 데이터프레임에 추가
    utm_list = [utm.from_latlon(lat, lon) for lat, lon in zip(df['lat'], df['lon'])]
    df['easting'] = [utm_coords[0] for utm_coords in utm_list]
    df['northing'] = [utm_coords[1] for utm_coords in utm_list]
    df['zone_number'] = [utm_coords[2] for utm_coords in utm_list]
    df['zone_letter'] = [utm_coords[3] for utm_coords in utm_list]

    # 초기 상태를 GPS 데이터의 첫 번째 위치로 설정
    initial_state = np.zeros(dim_x)
    initial_state[0] = df.iloc[0]['easting']  # 초기 x 위치
    initial_state[1] = df.iloc[0]['northing']  # 초기 y 위치
    initial_state[2] = df.iloc[0]['alt']  # 초기 z 위치
    initial_state[3] = df.iloc[0]['vn']  # 초기 x 속도
    initial_state[4] = df.iloc[0]['ve']  # 초기 y 속도
    initial_state[5] = df.iloc[0]['vu']  # 초기 z 속도
    initial_state[6] = df.iloc[0]['ax']  # 초기 x 가속도
    initial_state[7] = df.iloc[0]['ay']  # 초기 y 가속도
    initial_state[8] = df.iloc[0]['az']  # 초기 z 가속도
    initial_state[9] = df.iloc[0]['roll']  # 초기 롤 각도
    initial_state[10] = df.iloc[0]['pitch']  # 초기 피치 각도
    initial_state[11] = df.iloc[0]['yaw']  # 초기 요 각도
    initial_state[12] = df.iloc[0]['wx']  # 초기 롤 각속도
    initial_state[13] = df.iloc[0]['wy']  # 초기 피치 각속도
    initial_state[14] = df.iloc[0]['wz']  # 초기 요 각속도

    # 초기 상태 출력
    print("Initial state:", initial_state)

    # 칼만 필터 설정
    kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    kf.F = F_matrix(1.0)
    kf.H = H
    kf.P = P
    kf.Q = Q
    kf.R = R
    kf.x = initial_state

    # 필터 적용
    filtered_data = []

    for i in range(len(df)):
        dt = df.iloc[i]['dt']
        kf.F = F_matrix(dt)
        z = np.array([df.iloc[i]['easting'], df.iloc[i]['northing'], df.iloc[i]['alt']], dtype=float)
        kf.predict()
        # kf.x = apply_gravity(kf.x)
        kf.update(z)
        filtered_data.append(kf.x.copy())
        print(f"시간 {df.iloc[i]['timestamp']}의 상태 추정값: {kf.x}")

    filtered_df = pd.DataFrame(filtered_data,
                               columns=['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'roll', 'pitch', 'yaw', 'wx',
                                        'wy', 'wz'])

    # Dash 애플리케이션 설정
    app = dash.Dash(__name__)

    frames = [go.Frame(data=[
        go.Scatter3d(
            x=filtered_df['x'][:k + 1],
            y=filtered_df['y'][:k + 1],
            z=filtered_df['z'][:k + 1],
            mode='markers',
            marker=dict(size=2, color='red'),
            line=dict(width=1)
        ),
        go.Scatter3d(
            x=df['easting'][:k + 1],
            y=df['northing'][:k + 1],
            z=df['alt'][:k + 1],
            mode='markers',
            marker=dict(size=2, color='blue'),
            line=dict(width=1)
        )
    ]) for k in range(len(filtered_df))]

    app.layout = html.Div([
        dcc.Graph(
            id='3d-scatter-plot',
            figure={
                'data': [
                    go.Scatter3d(
                        x=df['easting'],
                        y=df['northing'],
                        z=df['alt'],
                        mode='markers+lines',
                        name='GPS',
                        marker=dict(size=2, color='blue'),
                        line=dict(width=1)
                    ),
                    go.Scatter3d(
                        x=filtered_df['x'],
                        y=filtered_df['y'],
                        z=filtered_df['z'],
                        mode='markers+lines',
                        name='Kalman Filtered',
                        marker=dict(size=2, color='red'),
                        line=dict(width=1)
                    ),
                    # index가 같은 지점을 선으로 연결
                    *[go.Scatter3d(
                        x=[df.iloc[k]['easting'], filtered_df.iloc[k]['x']],
                        y=[df.iloc[k]['northing'], filtered_df.iloc[k]['y']],
                        z=[df.iloc[k]['alt'], filtered_df.iloc[k]['z']],
                        mode='lines',
                        name='timeline',
                        line=dict(width=1, color='grey')
                    ) for k in range(len(df))],
                    # 최초 출발점 표시
                    go.Scatter3d(
                        x=[df.iloc[0]['easting']],
                        y=[df.iloc[0]['northing']],
                        z=[df.iloc[0]['alt']],
                        mode='markers',
                        name='Start',
                        marker=dict(size=5, color='black')
                    ),
                    # 10개마다 두 지점 검은색으로 마킹
                    go.Scatter3d(
                        x=df['easting'][::10],
                        y=df['northing'][::10],
                        z=df['alt'][::10],
                        mode='markers',
                        name='Checkpoints',
                        marker=dict(size=3, color='black')
                    ),
                    go.Scatter3d(
                        x=filtered_df['x'][::10],
                        y=filtered_df['y'][::10],
                        z=filtered_df['z'][::10],
                        mode='markers',
                        name='Checkpoints',
                        marker=dict(size=3, color='black')
                    ),
                    # 종점 마킹
                    go.Scatter3d(
                        x=[df.iloc[-1]['easting']],
                        y=[df.iloc[-1]['northing']],
                        z=[df.iloc[-1]['alt']],
                        mode='markers',
                        name='End',
                        marker=dict(size=5, color='green')
                    )
                ],
                'layout': go.Layout(
                    title='3D Visualization of GPS and Kalman Filtered Data',
                    scene=dict(
                        xaxis_title='Easting (m)',
                        yaxis_title='Northing (m)',
                        zaxis_title='Altitude (m)',
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

    app.run_server(debug=True, use_reloader=False)

