"""
predict_photo.py — тестирование модели на фотографиях
Запуск: python predict_photo.py
Требования: pip install ultralytics opencv-python

Этот файл может работать как самостоятельно (с matplotlib для отображения),
так и использовать predict_utils.py для headless-режима (например, для веба).
"""
import sys
from pathlib import Path
from ultralytics import YOLO

# Импортируем утилиты из predict_utils
from predict_utils import (
    load_model,
    predict_image as predict_image_utils,
    get_violation_status,
    validate_image_path
)

# ── Настройки ────────────────────────────────────────────
MODEL_PATH  = "ppe_best.pt"   # положи рядом с этим файлом
OUTPUT_DIR  = "results"       # папка для сохранения результатов
# ─────────────────────────────────────────────────────────

# Функция обратной совместимости - использует predict_utils
def predict_image(img_path: str, model: YOLO, save=True):
    """
    Совместимая с предыдущей версией функция.
    Использует predict_utils для детекции.
    """
    output_path = None
    if save:
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        output_path = str(Path(OUTPUT_DIR) / ("result_" + Path(img_path).name))

    result = predict_image_utils(img_path, model, output_path=output_path)

    if not result['success']:
        print(f"  [ERROR] {result['error']}")
        return {}

    counts = result['counts']
    violation, status_msg = get_violation_status(result)

    print(f"  Status: {status_msg}")
    if output_path:
        print(f"  Saved: {output_path}")

    return counts


def main():
    # Используем load_model из predict_utils с обработкой ошибок
    model = load_model(MODEL_PATH)
    if model is None:
        print(f"[ERROR] Модель не найдена: {MODEL_PATH}")
        print("Положи ppe_best.pt в ту же папку что и этот файл.")
        return

    print()  # пустая строка после загрузки модели

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

        # Валидация через predict_utils
        is_valid, error = validate_image_path(str(img_path))
        if not is_valid:
            print(f"  [ERROR] {error}")
            continue

        counts = predict_image(str(img_path), model)
        if counts:
            print(f"  Найдено: {counts}\n")
        else:
            print()  # пустая строка для разделения

    print(f"Готово! Результаты сохранены в папку: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
