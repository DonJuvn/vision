"""
webcam_demo.py — запуск детекции СИЗ через веб-камеру
Запуск: python webcam_demo.py
Требования: pip install ultralytics opencv-python
"""

import cv2
import time
from ultralytics import YOLO
from pathlib import Path

# ── Настройки ────────────────────────────────────────────
MODEL_PATH  = "ppe_best.pt"   # путь к скачанной модели
CONF_THRESH = 0.35            # порог уверенности (0.0–1.0)
SOURCE      = 0               # 0 = веб-камера, или путь к видео: "video.mp4"
# ─────────────────────────────────────────────────────────

CLASS_COLORS = {
    "Head":   (60,  60,  255),   # красный — голова без каски (BGR)
    "Helmet": (60,  200, 60),    # зелёный — каска есть
    "Person": (255, 180, 60),    # синий — человек
}

ALERT_MSG = {
    True:  ("НАРУШЕНИЕ: есть без каски!", (0, 0, 220)),
    False: ("ВСЕ В КАСКАХ",               (0, 180, 0)),
}


def draw_box(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] Модель не найдена: {MODEL_PATH}")
        print("Скачай ppe_best.pt из Colab и положи рядом с этим файлом.")
        return

    print(f"Загружаем модель: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    class_names = model.names

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print("[ERROR] Камера не открылась")
        return

    print("Камера запущена. Нажми Q для выхода.")

    fps_list = []
    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=CONF_THRESH, verbose=False)

        counts = {name: 0 for name in class_names.values()}

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id   = int(box.cls[0])
                cls_name = class_names[cls_id]
                conf_val = float(box.conf[0])
                color    = CLASS_COLORS.get(cls_name, (200, 200, 0))
                counts[cls_name] += 1
                draw_box(frame, x1, y1, x2, y2,
                         f"{cls_name} {conf_val:.2f}", color)

        # Панель статуса
        violation = counts.get("Head", 0) > 0
        msg, msg_color = ALERT_MSG[violation]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (420, 80), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        cv2.putText(frame, msg, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, msg_color, 2)
        stats = (f"Helmet:{counts.get('Helmet',0)}  "
                 f"Head(no helmet):{counts.get('Head',0)}  "
                 f"Person:{counts.get('Person',0)}")
        cv2.putText(frame, stats, (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        # FPS
        fps_list.append(1 / (time.perf_counter() - t0 + 1e-9))
        if len(fps_list) > 30:
            fps_list.pop(0)
        cv2.putText(frame, f"FPS: {sum(fps_list)/len(fps_list):.1f}",
                    (frame.shape[1] - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("PPE Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Стрим завершён.")


if __name__ == "__main__":
    main()
