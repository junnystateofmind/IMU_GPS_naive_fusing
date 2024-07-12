import pandas as pd
import time
import threading
from queue import Queue

# IMU 데이터 읽기
def read_imu_data(file_path):
    imu_data = pd.read_csv(file_path, delim_whitespace=True)
    imu_data['dt'] = imu_data['Time'] - imu_data['Time'].shift(1)
    imu_data = imu_data.iloc[1:].reset_index(drop=True)  # 첫 번째 행을 제거
    return imu_data

# GPS 데이터 읽기
def read_gps_data(file_path):
    gps_data = pd.read_csv(file_path)
    gps_data['dt'] = gps_data['Time'] - gps_data['Time'].shift(1)
    gps_data = gps_data.iloc[1:].reset_index(drop=True)  # 첫 번째 행을 제거
    return gps_data

# 실시간 데이터 스트리밍 시뮬레이션
class DataStreamer(threading.Thread):
    def __init__(self, name, data, queue, interval_col):
        super().__init__()
        self.name = name
        self.data = data
        self.queue = queue
        self.interval_col = interval_col
        self.index = 0
        self.running = True

    def run(self):
        print(f"{self.name} streamer started.")
        while self.index < len(self.data) and self.running:
            start_time = time.perf_counter()
            self.queue.put(self.data.iloc[self.index])
            self.index += 1
            if self.index < len(self.data):
                elapsed_time = time.perf_counter() - start_time
                sleep_time = self.data.iloc[self.index][self.interval_col] - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
        print(f"{self.name} streamer finished.")

    def stop(self):
        self.running = False
