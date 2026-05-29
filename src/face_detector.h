#pragma once

#include <opencv2/opencv.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#ifdef USE_MEDIAPIPE_NATIVE
  #include "mediapipe/tasks/cc/vision/face_detector/face_detector.h"
  #include "mediapipe/framework/formats/image_frame.h"
#endif

namespace fdet {

// ── Detection result ──────────────────────────────────────────────────────────

struct FaceDetection {
    cv::Rect2f  bbox;        // normalised [0,1]
    cv::Rect    bbox_px;     // pixel-space
    float       confidence{0};
    std::vector<cv::Point2f> keypoints;  // 6 MediaPipe keypoints

    bool valid() const { return confidence > 0.f && bbox.width > 0.f; }
};

// ── Configuration ─────────────────────────────────────────────────────────────

struct DetectorConfig {
    std::string model_path;
    float       min_confidence{0.5f};
    int         max_faces{10};

    bool use_mediapipe{true};
    bool use_python_bridge{true};
    bool use_opencv_dnn{true};

    float       save_confidence{0.80f};
    bool        save_keypoints{true};
    std::string dataset_root{"dataset"};
    cv::Size    model_input_size{128, 128};
};

// ── Abstract interface ────────────────────────────────────────────────────────

class IFaceDetector {
public:
    virtual ~IFaceDetector() = default;
    virtual std::vector<FaceDetection> detect(const cv::Mat& frame) = 0;
    virtual std::string backend_name() const = 0;
    virtual bool        is_ready()     const = 0;
};

// ── MediaPipe native C++ backend ──────────────────────────────────────────────

class MediaPipeNativeDetector : public IFaceDetector {
public:
    explicit MediaPipeNativeDetector(const DetectorConfig& cfg);
    ~MediaPipeNativeDetector() override;

    std::vector<FaceDetection> detect(const cv::Mat& frame) override;
    std::string backend_name() const override { return "MediaPipe-Native"; }
    bool        is_ready()     const override { return ready_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    DetectorConfig cfg_;
    bool ready_{false};
};

// ── Python bridge backend ────────────────────────────────────────────────────

class MediaPipePythonDetector : public IFaceDetector {
public:
    explicit MediaPipePythonDetector(const DetectorConfig& cfg);
    ~MediaPipePythonDetector() override;

    std::vector<FaceDetection> detect(const cv::Mat& frame) override;
    std::string backend_name() const override { return "MediaPipe-Python"; }
    bool        is_ready()     const override { return ready_; }

private:
    bool probe_python_mediapipe();
    std::vector<FaceDetection> parse_json_result(const std::string& json,
                                                  int img_w, int img_h);

    DetectorConfig cfg_;
    std::string    python_script_path_;
    std::string    python_exe_{"python3"};   // ← venv interpreter, resolved at init
    bool           ready_{false};
};

// ── OpenCV DNN fallback ───────────────────────────────────────────────────────

class OpenCVDNNDetector : public IFaceDetector {
public:
    explicit OpenCVDNNDetector(const DetectorConfig& cfg);
    ~OpenCVDNNDetector() override = default;

    std::vector<FaceDetection> detect(const cv::Mat& frame) override;
    std::string backend_name() const override { return "OpenCV-DNN"; }
    bool        is_ready()     const override { return ready_; }

private:
    cv::dnn::Net   net_;
    DetectorConfig cfg_;
    bool           ready_{false};
};

// ── Public façade ─────────────────────────────────────────────────────────────

class FaceDetector {
public:
    explicit FaceDetector(const DetectorConfig& cfg = {});
    ~FaceDetector() = default;

    FaceDetector(const FaceDetector&)            = delete;
    FaceDetector& operator=(const FaceDetector&) = delete;
    FaceDetector(FaceDetector&&)                 = default;
    FaceDetector& operator=(FaceDetector&&)      = default;

    std::vector<FaceDetection> detect(const cv::Mat& frame);

    cv::Mat visualise(const cv::Mat& frame,
                      const std::vector<FaceDetection>& dets) const;

    int save_dataset_faces(const cv::Mat& frame,
                           const std::vector<FaceDetection>& dets,
                           const std::string& label = "unknown");

    int process_image(const std::string& image_path,
                      const std::string& label = "unknown");

    int process_directory(const std::string& dir_path,
                          const std::string& label = "");

    std::string active_backend() const;

    DetectorConfig&       config()       { return cfg_; }
    const DetectorConfig& config() const { return cfg_; }

    using ProgressCb = std::function<void(int, int, const std::string&)>;
    void set_progress_callback(ProgressCb cb) { progress_cb_ = std::move(cb); }

private:
    void init_backends();
    std::string make_dataset_path(const std::string& label,
                                   int index, float conf) const;

    DetectorConfig cfg_;
    std::vector<std::unique_ptr<IFaceDetector>> backends_;
    IFaceDetector* active_{nullptr};
    ProgressCb     progress_cb_;
    int            dataset_counter_{0};
};

} // namespace fdet