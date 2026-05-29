#pragma once
#include "face_detector.h"

#include <fstream>        // std::ofstream
#include <functional>
#include <map>
#include <string>

namespace fdet {

struct DatasetStats {
    int frames_processed{0};
    int faces_detected{0};
    int faces_saved{0};
    int faces_skipped_low_conf{0};
    std::map<std::string, int> faces_per_label;
};

class DatasetBuilder {
public:
    explicit DatasetBuilder(FaceDetector& detector);

    DatasetStats capture_webcam(int cam_index, int seconds,
                                const std::string& label,
                                bool show_preview = true);

    DatasetStats process_video(const std::string& video_path,
                               const std::string& label,
                               int every_n_frames = 3,
                               bool show_preview  = false);

    DatasetStats process_directory(const std::string& dir_path,
                                   const std::string& label = "");

    void enable_yolo_labels(bool on = true)  { yolo_labels_   = on; }
    void enable_csv_manifest(bool on = true) { csv_manifest_  = on; }

    const DatasetStats& stats() const { return stats_; }
    void reset_stats()                { stats_ = {}; }

private:
    void print_stats() const;   // ← declared here
    void maybe_write_yolo(const std::string& img_path,
                           const FaceDetection& d, int img_w, int img_h);
    void maybe_write_csv(const std::string& img_path,
                          const std::string& label, const FaceDetection& d);

    FaceDetector&  detector_;
    DatasetStats   stats_;
    bool           yolo_labels_{false};
    bool           csv_manifest_{false};
    std::ofstream  manifest_file_;
};

} // namespace fdet