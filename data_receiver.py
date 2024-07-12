import time
from queue import Queue
from data_streamer import read_imu_data, read_gps_data, DataStreamer
import threading
import numpy as np

# 실시간 데이터 수신 및 처리
class DataReceiver(threading.Thread):
    def __init__(self, imu_queue, gps_queue):
        super().__init__()
        self.imu_queue = imu_queue
        self.gps_queue = gps_queue
        self.running = True
        self.imu_data_buffer = []

    def run(self):
        print("Data receiver started.")
        try:
            while self.running:
                while not self.imu_queue.empty():
                    imu_data = self.imu_queue.get()
                    self.imu_data_buffer.append(imu_data)
                    print(f"Buffered IMU data: {imu_data.to_dict()}")

                while not self.gps_queue.empty():
                    gps_data = self.gps_queue.get()
                    print(f"Processed GPS data: {gps_data.to_dict()}")
                    interpolated_imu = self.interpolate_imu_data(gps_data['Time'])
                    if interpolated_imu is not None:
                        print(f"Interpolated IMU data at GPS time {gps_data['Time']}: {interpolated_imu}")

                time.sleep(0.001)  # 메인 루프의 주기
        except KeyboardInterrupt:
            self.running = False
        print("Data receiver finished.")

    def interpolate_imu_data(self, gps_time):
        if len(self.imu_data_buffer) < 2:
            return None

        imu_data_times = [data['Time'] for data in self.imu_data_buffer]

        # 찾을 IMU 데이터의 인덱스
        idx = np.searchsorted(imu_data_times, gps_time, side='right')

        if idx == 0 or idx >= len(self.imu_data_buffer):
            return None

        # 보간할 두 IMU 데이터
        imu_data_prev = self.imu_data_buffer[idx - 1]
        imu_data_next = self.imu_data_buffer[idx]

        # 시간 비율 계산
        t_prev = imu_data_prev['Time']
        t_next = imu_data_next['Time']
        ratio = (gps_time - t_prev) / (t_next - t_prev)

        interpolated_imu = {}
        interpolated_imu['Time'] = gps_time
        for key in imu_data_prev.keys():
            if key != 'Time':
                interpolated_imu[key] = imu_data_prev[key] + ratio * (imu_data_next[key] - imu_data_prev[key])

        # 버퍼에서 사용된 IMU 데이터 제거 (idx 이전 데이터 제거)
        self.imu_data_buffer = self.imu_data_buffer[idx:]
        print(f"IMU buffer size after interpolation: {len(self.imu_data_buffer)}")

        return interpolated_imu

def main():
    # 파일 경로 설정
    imu_file_path = 'KittiEquivBiasedImu.txt'
    gps_file_path = 'KittiGps_converted.txt'

    # 데이터 읽기
    imu_data = read_imu_data(imu_file_path)
    gps_data = read_gps_data(gps_file_path)

    # 큐 생성
    imu_queue = Queue()
    gps_queue = Queue()

    # 데이터 스트리머 생성
    imu_streamer = DataStreamer('IMU', imu_data, imu_queue, 'dt')
    gps_streamer = DataStreamer('GPS', gps_data, gps_queue, 'dt')

    # 데이터 수신기 생성
    data_receiver = DataReceiver(imu_queue, gps_queue)

    # 데이터 스트리머 및 수신기 시작
    imu_streamer.start()
    gps_streamer.start()
    data_receiver.start()

    try:
        while imu_streamer.is_alive() or gps_streamer.is_alive():
            time.sleep(0.008)
    except KeyboardInterrupt:
        imu_streamer.stop()
        gps_streamer.stop()
        data_receiver.running = False
        imu_streamer.join()
        gps_streamer.join()
        data_receiver.join()

if __name__ == '__main__':
    main()
