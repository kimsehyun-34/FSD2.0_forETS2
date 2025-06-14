import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO

# YOLOv8 세그멘테이션 모델 로드
model = YOLO('models/TL/best_traffic_med_yolo_v8.pt')

# 화면 캡처 설정
monitor = {"top": 100, "left": 100, "width": 1280, "height": 720}  # 캡처 영역 설정
sct = mss()

def capture_screen():
    screenshot = sct.grab(monitor)
    img = np.array(screenshot)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

def segment_objects(frame):
    results = model.predict(frame, conf=0.5, device='cuda')  # 여기서 device 지정
    return results

def main():
    while True:
        # 화면 캡처
        frame = capture_screen()

        # 객체 세그멘테이션
        results = segment_objects(frame)

        # 세그멘테이션 결과 시각화
        annotated_frame = results[0].plot()  # 세그멘테이션된 객체를 시각화

        # 화면 출력 크기 조정
        resized_frame = cv2.resize(annotated_frame, (640, 360))  # 원하는 크기로 조정 (예: 640x360)

        # 화면 출력
        cv2.imshow("ETS2 Object Segmentation", resized_frame)

        # ESC 키로 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()