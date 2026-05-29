#include "face_detector.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#ifdef __APPLE__
  #include <mach-o/dyld.h>
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ═══════════════════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════════════════
namespace {

cv::Rect normalised_to_pixel(const cv::Rect2f& n, int w, int h)
{
    int x  = std::max(0, std::min((int)(n.x * w), w-1));
    int y  = std::max(0, std::min((int)(n.y * h), h-1));
    int bw = std::max(1, std::min((int)(n.width  * w), w-x));
    int bh = std::max(1, std::min((int)(n.height * h), h-y));
    return {x, y, bw, bh};
}

std::string ts_now()
{
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                   now.time_since_epoch()) % 1000;
    std::ostringstream ss;
    ss << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S")
       << "_" << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

// Find venv python3 relative to the running executable
std::string resolve_python()
{
    std::string exe_dir;
#ifdef __APPLE__
    char buf[4096]; uint32_t sz = sizeof(buf);
    if (_NSGetExecutablePath(buf, &sz) == 0)
        exe_dir = fs::path(buf).parent_path().string();
#endif
    if (!exe_dir.empty()) {
        for (auto c : {
            fs::path(exe_dir) / ".venv" / "bin" / "python3",
            fs::path(exe_dir).parent_path() / ".venv" / "bin" / "python3"
        }) if (fs::exists(c)) return c.string();
    }
    return "python3";
}

// Run a shell command and return stdout
std::string run_cmd(const std::string& cmd)
{
    std::string out;
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return out;
    char buf[4096];
    while (fgets(buf, sizeof(buf), p)) out += buf;
    pclose(p);
    return out;
}

} // anonymous namespace

namespace fdet {

// ═══════════════════════════════════════════════════════════════════════════
// MediaPipe Native stub (compiled out by default)
// ═══════════════════════════════════════════════════════════════════════════
struct MediaPipeNativeDetector::Impl {};
MediaPipeNativeDetector::MediaPipeNativeDetector(const DetectorConfig& cfg)
    : cfg_(cfg), impl_(std::make_unique<Impl>())
{ std::cout << "[MediaPipe-Native] Not compiled in — skipping.\n"; }
MediaPipeNativeDetector::~MediaPipeNativeDetector() = default;
std::vector<FaceDetection> MediaPipeNativeDetector::detect(const cv::Mat&) { return {}; }

// ═══════════════════════════════════════════════════════════════════════════
// Python bridge  (MediaPipe via subprocess)
// ═══════════════════════════════════════════════════════════════════════════

// НОВЫЙ Python-скрипт — использует MediaPipe Tasks API (0.10.35+)
static constexpr const char* kBridgeScript = R"PY(
#!/usr/bin/env python3
import sys, json, os
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import urllib.request
from pathlib import Path

# MediaPipe Tasks API (0.10.35+)
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
MODEL_PATH = None

def find_model_file():
    """Ищет модель в порядке приоритета: рядом с исполняемым файлом ../models, cwd/models, /tmp/fdet_models/"""
    exe_dir = os.path.dirname(sys.argv[0]) if hasattr(sys, 'frozen') else os.getcwd()
    candidates = [
        os.path.join(exe_dir, "../models/face_detector.tflite"),
        os.path.join(os.getcwd(), "models/face_detector.tflite"),
        "/tmp/fdet_models/face_detector.tflite"
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def ensure_model():
    global MODEL_PATH
    MODEL_PATH = find_model_file()
    if MODEL_PATH is not None:
        return True
    # Скачиваем
    try:
        os.makedirs("/tmp/fdet_models", exist_ok=True)
        download_path = "/tmp/fdet_models/face_detector.tflite"
        print(f"[Bridge] Downloading model from {MODEL_URL}", file=sys.stderr)
        urllib.request.urlretrieve(MODEL_URL, download_path)
        MODEL_PATH = download_path
        print(f"[Bridge] Model saved to {MODEL_PATH}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"[Bridge] Failed to download model: {e}", file=sys.stderr)
        return False

def detect_faces(image_path, min_conf, max_faces):
    if not ensure_model():
        return []

    img = cv2.imread(image_path)
    if img is None:
        print(f"[Bridge] Cannot read {image_path}", file=sys.stderr)
        return []
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Настройка детектора
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=min_conf
    )
    detector = mp_vision.FaceDetector.create_from_options(options)
    result = detector.detect(mp_image)

    detections = []
    for face in result.detections:
        # bounding_box в пикселях -> нормализация [0,1]
        bb = face.bounding_box
        x_norm = bb.origin_x / w
        y_norm = bb.origin_y / h
        w_norm = bb.width / w
        h_norm = bb.height / h
        # уверенность
        score = face.categories[0].score if face.categories else 0.0
        # keypoints уже нормализованы [0,1]
        keypoints = [{"x": kp.x, "y": kp.y} for kp in face.keypoints]
        detections.append({
            "confidence": score,
            "bbox": {
                "x": x_norm,
                "y": y_norm,
                "width": w_norm,
                "height": h_norm
            },
            "keypoints": keypoints
        })
    return detections

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--min-conf", type=float, default=0.25)
    p.add_argument("--max-faces", type=int, default=10)
    args = p.parse_args()
    result = detect_faces(args.image, args.min_conf, args.max_faces)
    # Выводим JSON в stdout (ровно одну строку)
    json.dump(result, sys.stdout, separators=(',', ':'))
)PY";

MediaPipePythonDetector::MediaPipePythonDetector(const DetectorConfig& cfg)
    : cfg_(cfg)
{
    python_script_path_ = "/tmp/fdet_bridge.py";
    { std::ofstream f(python_script_path_); f << kBridgeScript; }

    python_exe_ = resolve_python();
    std::cout << "[MediaPipe-Python] Interpreter: " << python_exe_ << "\n";

    ready_ = probe_python_mediapipe();
    if (ready_)
        std::cout << "[MediaPipe-Python] Initialised ✓\n";
    else
        std::cerr << "[MediaPipe-Python] mediapipe not found — run: bash setup.sh\n";
}

MediaPipePythonDetector::~MediaPipePythonDetector() = default;

bool MediaPipePythonDetector::probe_python_mediapipe()
{
    std::string cmd = python_exe_ + " -c \"import mediapipe\" 2>/dev/null";
    return std::system(cmd.c_str()) == 0;
}

std::vector<FaceDetection>
MediaPipePythonDetector::parse_json_result(const std::string& js,
                                            int W, int H)
{
    std::vector<FaceDetection> out;
    if (js.empty() || js.find('{') == std::string::npos) return out;

    // Simple extractor — avoids any external JSON lib
    auto get_f = [&](const std::string& key,
                     const std::string& s,
                     size_t from) -> std::pair<float,size_t> {
        std::string pat = "\"" + key + "\"";
        size_t p = s.find(pat, from);
        if (p == std::string::npos) return {0,std::string::npos};
        p = s.find(':', p + pat.size());
        if (p == std::string::npos) return {0,std::string::npos};
        while (++p < s.size() && (s[p]==' '||s[p]=='\t')) {}
        size_t e = p;
        while (e < s.size() && (std::isdigit(s[e])||s[e]=='.'||s[e]=='-')) ++e;
        if (e == p) return {0,std::string::npos};
        return {std::stof(s.substr(p,e-p)), e};
    };

    size_t cur = 0;
    while (true) {
        size_t s = js.find('{', cur);
        if (s == std::string::npos) break;
        int dep = 0; size_t e = s;
        for (; e < js.size(); ++e) {
            if (js[e]=='{') ++dep;
            else if (js[e]=='}') { --dep; if (!dep) break; }
        }
        std::string obj = js.substr(s, e-s+1);
        if (obj.find("\"confidence\"") != std::string::npos) {
            auto [conf,_] = get_f("confidence", obj, 0);
            if (conf >= cfg_.min_confidence) {
                FaceDetection fd;
                fd.confidence = conf;
                auto [bx,p1] = get_f("x",      obj, 0);
                auto [by,p2] = get_f("y",      obj, p1!=std::string::npos?p1:0);
                auto [bw,p3] = get_f("width",  obj, 0);
                auto [bh,p4] = get_f("height", obj, 0);
                fd.bbox    = {bx, by, bw, bh};
                fd.bbox_px = normalised_to_pixel(fd.bbox, W, H);

                // keypoints
                size_t ks = obj.find("\"keypoints\"");
                if (ks != std::string::npos) {
                    size_t as = obj.find('[', ks), ae = obj.find(']', as);
                    if (as != std::string::npos && ae != std::string::npos) {
                        std::string kstr = obj.substr(as, ae-as);
                        size_t kc = 0;
                        while (true) {
                            auto [kx,kp1] = get_f("x", kstr, kc);
                            if (kp1==std::string::npos) break;
                            auto [ky,kp2] = get_f("y", kstr, kp1);
                            if (kp2==std::string::npos) break;
                            fd.keypoints.emplace_back(kx*W, ky*H);
                            kc = kp2;
                        }
                    }
                }
                out.push_back(std::move(fd));
            }
        }
        cur = e + 1;
    }

    std::sort(out.begin(), out.end(),
        [](const FaceDetection& a, const FaceDetection& b){
            return a.confidence > b.confidence; });
    if (cfg_.max_faces > 0 && (int)out.size() > cfg_.max_faces)
        out.resize(cfg_.max_faces);
    return out;
}

std::vector<FaceDetection>
MediaPipePythonDetector::detect(const cv::Mat& frame)
{
    // Save frame → /tmp/fdet_TIMESTAMP.jpg
    std::string tmp = "/tmp/fdet_" + ts_now() + ".jpg";
    // Use 95% JPEG quality to avoid compression artifacts hurting detection
    cv::imwrite(tmp, frame, {cv::IMWRITE_JPEG_QUALITY, 95});

    std::string cmd = python_exe_ + " " + python_script_path_
                    + " \"" + tmp + "\""
                    + " --min-conf " + std::to_string(cfg_.min_confidence)
                    + " --max-faces " + std::to_string(cfg_.max_faces)
                    + " 2>/tmp/fdet_bridge_err.txt";

    std::string json_out = run_cmd(cmd);
    std::remove(tmp.c_str());

    if (json_out.empty() || json_out.find('[') == std::string::npos) {
        // Print last bridge error for debugging
        std::string err = run_cmd("tail -1 /tmp/fdet_bridge_err.txt 2>/dev/null");
        if (!err.empty() && err != "\n")
            std::cerr << "\r[bridge] " << err << std::flush;
        return {};
    }
    return parse_json_result(json_out, frame.cols, frame.rows);
}

// ═══════════════════════════════════════════════════════════════════════════
// OpenCV DNN fallback — ResNet-SSD
// ═══════════════════════════════════════════════════════════════════════════

OpenCVDNNDetector::OpenCVDNNDetector(const DetectorConfig& cfg) : cfg_(cfg)
{
    // Look for models/ relative to CWD, then exe dir
    auto try_load = [&](const std::string& base) -> bool {
        std::string proto = base + "/deploy.prototxt";
        std::string model = base + "/res10_300x300_ssd_iter_140000_fp16.caffemodel";
        if (!fs::exists(proto) || !fs::exists(model)) return false;
        net_ = cv::dnn::readNetFromCaffe(proto, model);
        return !net_.empty();
    };

    std::string exe_dir;
#ifdef __APPLE__
    char buf[4096]; uint32_t sz = sizeof(buf);
    if (_NSGetExecutablePath(buf, &sz) == 0)
        exe_dir = fs::path(buf).parent_path().parent_path().string();
#endif

    if (try_load("models") || try_load(exe_dir + "/models")) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        ready_ = true;
        std::cout << "[OpenCV-DNN] Initialised ✓\n";
    } else {
        std::cerr << "[OpenCV-DNN] models/ not found\n";
    }
}

std::vector<FaceDetection>
OpenCVDNNDetector::detect(const cv::Mat& frame)
{
    cv::Mat blob = cv::dnn::blobFromImage(
        frame, 1.0, {300,300}, {104.0,177.0,123.0}, false, false);
    net_.setInput(blob);
    cv::Mat out = net_.forward();

    const int W = frame.cols, H = frame.rows;
    std::vector<FaceDetection> res;
    float* d = (float*)out.data;
    int    N = out.size[2];
    for (int i = 0; i < N; ++i) {
        float conf = d[i*7+2];
        if (conf < cfg_.min_confidence) continue;
        FaceDetection fd;
        fd.confidence = conf;
        float x1=d[i*7+3], y1=d[i*7+4], x2=d[i*7+5], y2=d[i*7+6];
        fd.bbox    = {x1, y1, x2-x1, y2-y1};
        fd.bbox_px = normalised_to_pixel(fd.bbox, W, H);
        res.push_back(std::move(fd));
    }
    std::sort(res.begin(), res.end(),
        [](const FaceDetection& a, const FaceDetection& b){
            return a.confidence > b.confidence; });
    if (cfg_.max_faces > 0 && (int)res.size() > cfg_.max_faces)
        res.resize(cfg_.max_faces);
    return res;
}

// ═══════════════════════════════════════════════════════════════════════════
// FaceDetector façade
// ═══════════════════════════════════════════════════════════════════════════

FaceDetector::FaceDetector(const DetectorConfig& cfg) : cfg_(cfg)
{
    // 1 → MediaPipe Python,  2 → OpenCV DNN
    if (cfg_.use_mediapipe) {
        auto b = std::make_unique<MediaPipeNativeDetector>(cfg_);
        backends_.push_back(std::move(b));
    }
    if (cfg_.use_python_bridge) {
        auto b = std::make_unique<MediaPipePythonDetector>(cfg_);
        if (b->is_ready() && !active_) active_ = b.get();
        backends_.push_back(std::move(b));
    }
    if (cfg_.use_opencv_dnn) {
        auto b = std::make_unique<OpenCVDNNDetector>(cfg_);
        if (b->is_ready() && !active_) active_ = b.get();
        backends_.push_back(std::move(b));
    }
    if (!active_)
        std::cerr << "[FaceDetector] WARNING: no backend ready!\n";
}

std::string FaceDetector::active_backend() const
{
    return active_ ? active_->backend_name() : "none";
}

std::vector<FaceDetection> FaceDetector::detect(const cv::Mat& frame)
{
    if (!active_ || frame.empty()) return {};
    return active_->detect(frame);
}

cv::Mat FaceDetector::visualise(const cv::Mat& frame,
                                 const std::vector<FaceDetection>& dets) const
{
    cv::Mat out = frame.clone();
    for (auto& d : dets) {
        cv::Scalar col = (d.confidence >= cfg_.save_confidence)
                       ? cv::Scalar(50,220,50) : cv::Scalar(0,180,255);
        cv::rectangle(out, d.bbox_px, col, 2);

        std::ostringstream lbl;
        lbl << std::fixed << std::setprecision(0) << d.confidence*100 << "%";
        int base=0;
        cv::Size ts = cv::getTextSize(lbl.str(),
                                      cv::FONT_HERSHEY_SIMPLEX,0.5,1,&base);
        cv::Point tl(d.bbox_px.x, d.bbox_px.y-5);
        cv::rectangle(out, tl+cv::Point(0,base),
                      tl+cv::Point(ts.width,-ts.height-2), col, cv::FILLED);
        cv::putText(out, lbl.str(), tl,
                    cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,0),1,cv::LINE_AA);
        for (auto& kp : d.keypoints)
            cv::circle(out, kp, 3, cv::Scalar(255,100,30), cv::FILLED);
    }
    return out;
}

std::string FaceDetector::make_dataset_path(const std::string& label,
                                             int idx, float conf) const
{
    fs::path dir = fs::path(cfg_.dataset_root) / label;
    fs::create_directories(dir);
    std::ostringstream fn;
    fn << ts_now() << "_" << std::setfill('0') << std::setw(4) << idx
       << "_" << std::fixed << std::setprecision(2) << conf << ".jpg";
    return (dir / fn.str()).string();
}

int FaceDetector::save_dataset_faces(const cv::Mat& frame,
                                      const std::vector<FaceDetection>& dets,
                                      const std::string& label)
{
    int saved = 0;
    for (auto& d : dets) {
        if (d.confidence < cfg_.save_confidence) continue;
        if (d.bbox_px.area() <= 0) continue;
        cv::Rect roi = d.bbox_px;
        int px = (int)(roi.width*0.1f), py = (int)(roi.height*0.1f);
        roi.x      = std::max(0, roi.x-px);
        roi.y      = std::max(0, roi.y-py);
        roi.width  = std::min(frame.cols-roi.x, roi.width +2*px);
        roi.height = std::min(frame.rows-roi.y, roi.height+2*py);
        cv::Mat crop;
        cv::resize(frame(roi).clone(), crop, cfg_.model_input_size);
        std::string path = make_dataset_path(label, ++dataset_counter_, d.confidence);
        if (!cv::imwrite(path, crop)) continue;
        ++saved;
        if (cfg_.save_keypoints) {
            std::ofstream m(path+".json");
            m<<"{\n  \"confidence\":"<<d.confidence
             <<",\n  \"bbox_norm\":["<<d.bbox.x<<","<<d.bbox.y
             <<","<<d.bbox.width<<","<<d.bbox.height<<"],\n  \"keypoints\":[";
            for (size_t k=0;k<d.keypoints.size();++k){
                m<<"["<<d.keypoints[k].x<<","<<d.keypoints[k].y<<"]";
                if(k+1<d.keypoints.size())m<<",";
            }
            m<<"]\n}\n";
        }
    }
    return saved;
}

int FaceDetector::process_image(const std::string& path, const std::string& label)
{
    cv::Mat img = cv::imread(path);
    if (img.empty()) return 0;
    return save_dataset_faces(img, detect(img), label);
}

int FaceDetector::process_directory(const std::string& dir, const std::string& label)
{
    static const std::vector<std::string> exts =
        {".jpg",".jpeg",".png",".bmp",".tiff",".tif",".webp"};
    std::vector<fs::path> imgs;
    for (auto& e : fs::recursive_directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(),ext.end(),ext.begin(),::tolower);
        if (std::find(exts.begin(),exts.end(),ext)!=exts.end())
            imgs.push_back(e.path());
    }
    std::sort(imgs.begin(),imgs.end());
    int tot=0;
    for (int i=0;i<(int)imgs.size();++i){
        std::string lbl = label.empty()
            ? imgs[i].parent_path().filename().string() : label;
        tot += process_image(imgs[i].string(), lbl);
        if (progress_cb_) progress_cb_(i+1,(int)imgs.size(),imgs[i].filename().string());
    }
    return tot;
}

} // namespace fdet