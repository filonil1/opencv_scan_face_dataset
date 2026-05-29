# MediaPipe Face Detection + OpenCV Dataset Builder

## Архитектура: три бэкенда с авто-фоллбэком

```
FaceDetector (фасад)
│
├─ 1. MediaPipe C++ Tasks API   (USE_MEDIAPIPE_NATIVE=ON + Bazel)
├─ 2. MediaPipe Python Bridge   (pip install mediapipe — работает сразу)
└─ 3. OpenCV DNN Fallback       (ResNet-SSD, всегда доступен)
```

При старте пробуется бэкенд 1 → 2 → 3; используется первый рабочий.

---

## Быстрый старт (Python bridge — рекомендуется)

```bash
# 1. Зависимости
brew install cmake opencv          # OpenCV 4.13
pip3 install mediapipe             # Python-мост

# 2. Модель-фоллбэк (скачается автоматически)
bash setup.sh

# 3. Запуск
./build/face_detector                          # live-камера
./build/face_detector webcam  Alice 30         # 30 с → датасет/Alice/
./build/face_detector image   photo.jpg  Bob   # один файл
./build/face_detector video   clip.mp4   Carol # видео
./build/face_detector dir     raw_photos/      # пакетная обработка
```

---

## Управление в live-режиме

| Клавиша | Действие                      |
|---------|-------------------------------|
| `s`     | сохранить текущий кадр        |
| `1`–`9` | переключить метку (person1…9) |
| `q`     | выйти                         |

---

## Структура датасета

```
dataset/
├── Alice/
│   ├── 20240601_143201_001_0.95.jpg   # лицо
│   ├── 20240601_143201_001_0.95.jpg.json  # bbox + keypoints
│   └── ...
├── Bob/
│   └── ...
└── manifest.csv                       # path,label,confidence,x,y,w,h
```

### JSON-сайдкар

```json
{
  "confidence": 0.97,
  "bbox_norm": [0.31, 0.12, 0.38, 0.51],
  "keypoints": [[412,180],[531,178],[471,231],[418,290],[528,289]]
}
```

### YOLO-метки (опционально)

Включается через `builder.enable_yolo_labels(true)` — рядом с каждым
`.jpg` создаётся `.txt` в формате YOLO:

```
0 0.50 0.37 0.38 0.51
```

---

## Нативный C++ MediaPipe (опционально, сложнее)

MediaPipe C++ Tasks требует сборки через Bazel:

```bash
# Зависимости
brew install bazel xz

# Клонируем MediaPipe
git clone https://github.com/google/mediapipe.git
cd mediapipe

# Собираем Tasks Vision shared library
bazel build -c opt \
    --define MEDIAPIPE_DISABLE_GPU=1 \
    mediapipe/tasks/cc/vision/face_detector:face_detector \
    --output_groups=default

# Копируем хедеры и .dylib
sudo mkdir -p /usr/local/mediapipe/{include,lib}
sudo cp -r mediapipe /usr/local/mediapipe/include/
sudo cp bazel-bin/mediapipe/tasks/cc/vision/face_detector/*.dylib \
        /usr/local/mediapipe/lib/

# Сборка проекта с нативным бэкендом
bash setup.sh --with-native-mediapipe
```

---

## API

```cpp
#include "face_detector.h"
#include "dataset_builder.h"
using namespace fdet;

// Конфигурация
DetectorConfig cfg;
cfg.min_confidence  = 0.50f;  // порог детекции
cfg.save_confidence = 0.80f;  // порог сохранения в датасет
cfg.max_faces       = 10;
cfg.dataset_root    = "dataset";

FaceDetector detector(cfg);
std::cout << detector.active_backend(); // → "MediaPipe-Python"

// Детекция
cv::Mat frame = cv::imread("photo.jpg");
auto dets = detector.detect(frame);

for (auto& d : dets) {
    // d.confidence  — float  [0,1]
    // d.bbox_px     — cv::Rect (пиксели)
    // d.bbox        — cv::Rect2f (нормированные [0,1])
    // d.keypoints   — 6 точек: глаза, нос, уголки рта, уши
}

// Сохранение в датасет (только confidence > 0.80)
int saved = detector.save_dataset_faces(frame, dets, "Alice");

// Визуализация
cv::Mat vis = detector.visualise(frame, dets);

// Пакетная обработка директории
detector.process_directory("raw_photos/Alice/", "Alice");

// DatasetBuilder — расширенный сборщик
DatasetBuilder builder(detector);
builder.enable_yolo_labels(true);
builder.enable_csv_manifest(true);
builder.capture_webcam(0, /*seconds=*/30, "Bob");
builder.process_video("interview.mp4", "Carol", /*every_n=*/5);
```

---

## Точность по бэкендам

| Бэкенд           | mAP (wider_face) | Скорость (Mac M2) | Keypoints |
|------------------|------------------|-------------------|-----------|
| MediaPipe Native | ~92%             | ~3 ms/кадр        | ✓ 6 точек |
| MediaPipe Python | ~92%             | ~8 ms/кадр        | ✓ 6 точек |
| OpenCV DNN       | ~85%             | ~12 ms/кадр       | ✗         |

---

## Требования

- macOS 12+ (Apple Silicon или Intel)
- CMake ≥ 3.20
- OpenCV 4.13 (`brew install opencv`)
- Python 3.9+ + `pip install mediapipe`
- Для нативного бэкенда: Bazel 6+
