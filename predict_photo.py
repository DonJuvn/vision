"""
predict_photo.py — тестирование модели на фотографиях
Запуск: python predict_photo.py
Требования: pip install ultralytics opencv-python matplotlib
"""
import os
# ПРИНУДИТЕЛЬНО ОТКЛЮЧАЕМ GPU, чтобы обойти ошибку WinError 1114
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from ultralytics import YOLO

# ── Настройки ────────────────────────────────────────────
MODEL_PATH  = "ppe_best.pt"   # положи рядом с этим файлом
CONF_THRESH = 0.35
OUTPUT_DIR  = "results"       # папка для сохранения результатов
# ─────────────────────────────────────────────────────────

CLASS_COLORS = {
    "Head":   "#FF4444",   # красный — голова без каски
    "Helmet": "#44DD44",   # зелёный — каска есть
    "Person": "#44AAFF",   # синий — человек
}


def predict_image(img_path: str, model: YOLO, save=True):
    results = model.predict(img_path, conf=CONF_THRESH, verbose=False)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_rgb)

    counts = {}
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id   = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf_val = float(box.conf[0])
            color    = CLASS_COLORS.get(cls_name, "#FFFF00")
            counts[cls_name] = counts.get(cls_name, 0) + 1

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.5, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 6, f"{cls_name} {conf_val:.2f}",
                    color=color, fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))

    # Статус нарушения
    no_helmet = counts.get("Head", 0)
    if no_helmet > 0:
        status = f"🔴 НАРУШЕНИЕ: {no_helmet} чел. без каски!"
        status_color = "red"
    else:
        status = "🟢 ВСЕ В КАСКАХ"
        status_color = "lime"

    counts_str = "  |  ".join(f"{k}: {v}" for k, v in counts.items())
    ax.set_title(f"{status}\n{counts_str}", fontsize=13,
                 color=status_color, pad=10,
                 bbox=dict(facecolor="black", alpha=0.6))
    ax.axis("off")
    plt.tight_layout()

    if save:
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        out_path = Path(OUTPUT_DIR) / ("result_" + Path(img_path).name)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Сохранено: {out_path}")

    plt.show()
    return counts


def main():
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] Модель не найдена: {MODEL_PATH}")
        print("Положи ppe_best.pt в ту же папку что и этот файл.")
        return

    print(f"Загружаем модель: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"Классы: {list(model.names.values())}\n")

    # Если путь передан аргументом: python predict_photo.py photo.jpg
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        # Иначе обрабатываем все jpg/png в текущей папке
        paths = (list(Path(".").glob("*.jpg")) +
                 list(Path(".").glob("*.png")) +
                 list(Path(".").glob("*.jpeg")))
        paths = [str(p) for p in paths
                 if "result_" not in p.name and p.name != "ppe_best.pt"]

    if not paths:
        print("Не найдено изображений.")
        print("Использование:")
        print("  python predict_photo.py photo.jpg")
        print("  или положи .jpg/.png файлы в эту же папку и запусти скрипт.")
        return

    print(f"Найдено изображений: {len(paths)}\n")
    for img_path in paths:
        print(f"Обрабатываем: {img_path}")
        counts = predict_image(str(img_path), model)
        print(f"  Найдено: {counts}\n")

    print(f"Готово! Результаты сохранены в папку: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
