#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
step()  { echo -e "\n${CYAN}══ $* ══${NC}"; }

step "Checking project files"
MISSING=0
for f in \
    "CMakeLists.txt" \
    "src/face_detector.h" \
    "src/face_detector.cpp" \
    "src/dataset_builder.h" \
    "src/dataset_builder.cpp" \
    "src/person_registry.h" \
    "src/person_registry.cpp" \
    "src/dataset_manager.h" \
    "src/dataset_manager.cpp" \
    "src/face_recognizer.h" \
    "src/face_recognizer.cpp" \
    "src/live_window.h" \
    "src/live_window.cpp" \
    "src/main.cpp"
do
    if [[ -f "$SCRIPT_DIR/$f" ]]; then
        info "$f ✓"
    else
        error "$f  ← MISSING"
        MISSING=1
    fi
done
if [[ $MISSING -eq 1 ]]; then
    error "Some files are missing. Expected layout in $SCRIPT_DIR/"
    exit 1
fi

step "Homebrew dependencies"
for pkg in cmake opencv python3; do
    if ! brew list --formula "$pkg" &>/dev/null; then
        info "Installing $pkg …"
        brew install "$pkg"
    else
        info "$pkg already installed ✓"
    fi
done

step "Python virtual environment"
VENV_DIR="$SCRIPT_DIR/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating venv …"
    python3 -m venv "$VENV_DIR"
else
    info "venv already exists ✓"
fi
source "$VENV_DIR/bin/activate"
info "Activated: $(which python3)  ($(python3 --version))"
pip install --upgrade pip --quiet
if python3 -c "import mediapipe" 2>/dev/null; then
    info "mediapipe already installed in venv ✓"
else
    info "Installing mediapipe + opencv-python …"
    pip install mediapipe opencv-python
fi
ln -sf "$VENV_DIR/bin/python3" "$SCRIPT_DIR/python3_fdet"

step "DNN model download"
MODEL_DIR="$SCRIPT_DIR/models"
mkdir -p "$MODEL_DIR"
PROTO="$MODEL_DIR/deploy.prototxt"
CAFFE="$MODEL_DIR/res10_300x300_ssd_iter_140000_fp16.caffemodel"
[[ ! -f "$PROTO" ]] && {
    info "Downloading deploy.prototxt …"
    curl -fsSL "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt" -o "$PROTO" || warn "Download failed — place manually in models/"
} || info "deploy.prototxt ✓"
[[ ! -f "$CAFFE" ]] && {
    info "Downloading caffemodel …"
    curl -fsSL "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel" -o "$CAFFE" || warn "Download failed — place manually in models/"
} || info "caffemodel ✓"

step "MediaPipe model download"
TFLITE_MODEL="$MODEL_DIR/face_detector.tflite"
TFLITE_URL="https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
if [[ ! -f "$TFLITE_MODEL" ]]; then
    info "Downloading MediaPipe face detector model …"
    curl -fsSL "$TFLITE_URL" -o "$TFLITE_MODEL" || warn "Download failed — will be auto-downloaded at runtime"
else
    info "MediaPipe model already exists ✓"
fi

step "CMake configure"
BUILD_DIR="$SCRIPT_DIR/build"
[[ -f "$BUILD_DIR/CMakeCache.txt" ]] && {
    info "Removing stale CMakeCache.txt …"
    rm -rf "$BUILD_DIR/CMakeCache.txt" "$BUILD_DIR/CMakeFiles"
}
mkdir -p "$BUILD_DIR"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
CMAKE_EXIT=$?
if [[ $CMAKE_EXIT -ne 0 ]]; then
    error "cmake configure failed (exit $CMAKE_EXIT)"
    exit 1
fi

step "Build"
cmake --build "$BUILD_DIR" --parallel "$(sysctl -n hw.logicalcpu)"
BUILD_EXIT=$?
BIN="$BUILD_DIR/face_detector"
if [[ -f "$BIN" && $BUILD_EXIT -eq 0 ]]; then
    info "Build successful! → $BIN"
    echo ""
    echo "  $BIN add    Alice          # register person"
    echo "  $BIN list                  # show all persons"
    echo "  $BIN capture               # interactive webcam"
    echo "  $BIN record Alice 30       # auto-capture 30s"
    echo "  $BIN import ./photos/      # batch import"
    echo "  $BIN test                  # detector self-test"
else
    error "Build FAILED (cmake=$CMAKE_EXIT build=$BUILD_EXIT)"
    echo ""
    echo "Run for verbose output:"
    echo "  cmake --build $BUILD_DIR --verbose"
    exit 1
fi
