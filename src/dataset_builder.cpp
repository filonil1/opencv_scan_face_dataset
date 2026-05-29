#include "dataset_builder.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>

namespace fs  = std::filesystem;
namespace chr = std::chrono;
using chr::steady_clock;

namespace fdet {

DatasetBuilder::DatasetBuilder(FaceDetector& detector)
    : detector_(detector) {}

// ─── Webcam capture ───────────────────────────────────────────────────────────

DatasetStats DatasetBuilder::capture_webcam(int cam_index, int seconds,
                                             const std::string& label,
                                             bool show_preview)
{
    cv::VideoCapture cap(cam_index);
    if (!cap.isOpened()) {
        std::cerr << "[DatasetBuilder] Cannot open camera " << cam_index << "\n";
        return stats_;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    // Warm up: skip first N frames while camera exposure settles
    std::cout << "[DatasetBuilder] Warming up camera";
    cv::Mat dummy;
    for (int i = 0; i < 20; ++i) { cap >> dummy; std::cout << "." << std::flush; }
    std::cout << " ready!\n";

    auto t_end = steady_clock::now() + chr::seconds(seconds);
    cv::Mat frame;

    std::cout << "[DatasetBuilder] Capturing " << seconds
              << "s from webcam " << cam_index
              << " → label '" << label << "'\n"
              << "  Press 'q' to stop early.\n";

    while (steady_clock::now() < t_end) {
        cap >> frame;
        if (frame.empty()) continue;

        auto dets = detector_.detect(frame);
        stats_.frames_processed++;
        stats_.faces_detected += static_cast<int>(dets.size());

        int saved = detector_.save_dataset_faces(frame, dets, label);
        stats_.faces_saved += saved;
        stats_.faces_skipped_low_conf +=
            static_cast<int>(dets.size()) - saved;
        stats_.faces_per_label[label] += saved;

        if (show_preview) {
            cv::Mat vis = detector_.visualise(frame, dets);

            // Countdown — use chr::duration_cast explicitly
            auto remaining = t_end - steady_clock::now();
            int secs_left = static_cast<int>(
                chr::duration_cast<chr::seconds>(remaining).count());

            cv::putText(vis,
                "Saving: " + label +
                "  [" + std::to_string(secs_left) + "s]  saved:" +
                std::to_string(stats_.faces_saved),
                {8, vis.rows - 10},
                cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(50, 255, 120), 1, cv::LINE_AA);
            cv::imshow("FaceDetector — Dataset Capture", vis);
            if (cv::waitKey(1) == 'q') break;
        }
    }
    if (show_preview) cv::destroyAllWindows();
    cap.release();
    print_stats();
    return stats_;
}

// ─── Video file ───────────────────────────────────────────────────────────────

DatasetStats DatasetBuilder::process_video(const std::string& video_path,
                                            const std::string& label,
                                            int every_n_frames,
                                            bool show_preview)
{
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "[DatasetBuilder] Cannot open: " << video_path << "\n";
        return stats_;
    }

    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << "[DatasetBuilder] Processing video: " << video_path
              << "  (" << total_frames << " frames)\n";

    cv::Mat frame;
    int frame_idx = 0;

    while (cap.read(frame)) {
        ++frame_idx;
        if (frame_idx % every_n_frames != 0) continue;

        stats_.frames_processed++;
        auto dets = detector_.detect(frame);
        stats_.faces_detected += static_cast<int>(dets.size());

        int saved = detector_.save_dataset_faces(frame, dets, label);
        stats_.faces_saved += saved;
        stats_.faces_per_label[label] += saved;

        if (show_preview) {
            cv::Mat vis = detector_.visualise(frame, dets);
            cv::putText(vis,
                "Frame " + std::to_string(frame_idx) +
                "/" + std::to_string(total_frames),
                {8, vis.rows - 10}, cv::FONT_HERSHEY_SIMPLEX,
                0.55, cv::Scalar(50, 255, 120), 1, cv::LINE_AA);
            cv::imshow("FaceDetector — Video", vis);
            if (cv::waitKey(1) == 'q') break;
        }

        if (frame_idx % 100 == 0)
            std::cout << "  Frame " << frame_idx << "/" << total_frames
                      << "  saved: " << stats_.faces_saved << "\n";
    }
    if (show_preview) cv::destroyAllWindows();
    cap.release();
    print_stats();
    return stats_;
}

// ─── Directory of images ──────────────────────────────────────────────────────

DatasetStats DatasetBuilder::process_directory(const std::string& dir_path,
                                                const std::string& label)
{
    detector_.set_progress_callback(
        [this](int done, int total, const std::string& file) {
            std::cout << "\r  [" << done << "/" << total << "] " << file
                      << "   saved total: " << stats_.faces_saved << "   "
                      << std::flush;
        });

    int saved = detector_.process_directory(dir_path, label);
    stats_.faces_saved += saved;
    stats_.faces_per_label[label.empty() ? "auto" : label] += saved;

    std::cout << "\n";
    print_stats();
    return stats_;
}

// ─── YOLO helper ─────────────────────────────────────────────────────────────

void DatasetBuilder::maybe_write_yolo(const std::string& img_path,
                                       const FaceDetection& d,
                                       int /*img_w*/, int /*img_h*/)
{
    if (!yolo_labels_) return;
    fs::path lbl_path = fs::path(img_path).replace_extension(".txt");
    std::ofstream f(lbl_path, std::ios::app);
    float cx = d.bbox.x + d.bbox.width  / 2.f;
    float cy = d.bbox.y + d.bbox.height / 2.f;
    f << "0 " << cx << " " << cy << " "
               << d.bbox.width << " " << d.bbox.height << "\n";
}

// ─── CSV helper ───────────────────────────────────────────────────────────────

void DatasetBuilder::maybe_write_csv(const std::string& img_path,
                                      const std::string& label,
                                      const FaceDetection& d)
{
    if (!csv_manifest_) return;
    if (!manifest_file_.is_open()) {
        fs::path mp = fs::path(detector_.config().dataset_root) / "manifest.csv";
        manifest_file_.open(mp, std::ios::app);
        if (manifest_file_.tellp() == 0)
            manifest_file_ << "path,label,confidence,x,y,w,h\n";
    }
    manifest_file_ << img_path << "," << label << ","
                   << std::fixed << std::setprecision(4) << d.confidence << ","
                   << d.bbox.x << "," << d.bbox.y << ","
                   << d.bbox.width << "," << d.bbox.height << "\n";
}

// ─── Print stats ─────────────────────────────────────────────────────────────

void DatasetBuilder::print_stats() const
{
    std::cout << "\n══ Dataset stats ═══════════════════════════\n"
              << "  Frames processed  : " << stats_.frames_processed << "\n"
              << "  Faces detected    : " << stats_.faces_detected   << "\n"
              << "  Faces saved       : " << stats_.faces_saved      << "\n"
              << "  Skipped (low conf): " << stats_.faces_skipped_low_conf << "\n"
              << "  Per label:\n";
    for (const auto& [lbl, cnt] : stats_.faces_per_label)
        std::cout << "    " << lbl << ": " << cnt << "\n";
    std::cout << "════════════════════════════════════════════\n";
}

} // namespace fdet