#include "dataset_manager.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace fs  = std::filesystem;
namespace chr = std::chrono;

namespace fdet {

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace {
std::string ts_ms() {
    auto now = chr::system_clock::now();
    auto t   = chr::system_clock::to_time_t(now);
    auto ms  = chr::duration_cast<chr::milliseconds>(
                   now.time_since_epoch()) % 1000;
    std::ostringstream ss;
    ss << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S")
       << "_" << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}
} // namespace

// ─────────────────────────────────────────────────────────────────────────────

DatasetManager::DatasetManager(FaceDetector& detector,
                                const std::string& dataset_root)
    : detector_(detector)
    , registry_(dataset_root)
{
    open_manifest();
    // Sync sample_counter_ from registry totals
    for (auto& p : registry_.all_persons())
        sample_counter_ += p.samples;
}

DatasetManager::~DatasetManager() {
    if (manifest_.is_open()) manifest_.flush();
}

void DatasetManager::open_manifest() {
    fs::path mp = fs::path(registry_.dataset_root()) / "manifest.csv";
    bool exists = fs::exists(mp);
    manifest_.open(mp, std::ios::app);
    if (!exists)
        manifest_ << "path,person_id,person_name,confidence,"
                     "bbox_x,bbox_y,bbox_w,bbox_h,"
                     "kp0x,kp0y,kp1x,kp1y,kp2x,kp2y,kp3x,kp3y,kp4x,kp4y,kp5x,kp5y\n";
}

// ── Registration ──────────────────────────────────────────────────────────────

std::string DatasetManager::register_person(const std::string& name) {
    return registry_.get_or_create(name);
}

void DatasetManager::list_persons() const {
    registry_.print_table();
}

// ── Face saving ───────────────────────────────────────────────────────────────

std::string DatasetManager::make_face_path(const std::string& person_id,
                                            int sample_index, float conf) const
{
    auto info = registry_.find_by_id(person_id);
    std::string folder = registry_.folder_name(person_id);
    fs::path dir = fs::path(registry_.dataset_root()) / folder;
    fs::create_directories(dir);

    // Filename: ID_Name_NNNN_conf.jpg
    std::string name = info ? info->name : "unknown";
    std::ostringstream fname;
    fname << person_id << "_" << name
          << "_" << std::setfill('0') << std::setw(4) << sample_index
          << "_" << std::fixed << std::setprecision(2) << conf
          << ".jpg";
    return (dir / fname.str()).string();
}

void DatasetManager::write_manifest_row(const std::string& path,
                                         const std::string& person_id,
                                         const std::string& person_name,
                                         const FaceDetection& d)
{
    if (!manifest_.is_open()) return;
    manifest_ << path << ","
              << person_id << "," << person_name << ","
              << std::fixed << std::setprecision(4) << d.confidence << ","
              << d.bbox.x << "," << d.bbox.y << ","
              << d.bbox.width << "," << d.bbox.height;
    // Keypoints (up to 6)
    for (int k = 0; k < 6; ++k) {
        if (k < (int)d.keypoints.size())
            manifest_ << "," << d.keypoints[k].x << "," << d.keypoints[k].y;
        else
            manifest_ << ",,";
    }
    manifest_ << "\n";
    manifest_.flush();
}

int DatasetManager::save_faces(const cv::Mat& frame,
                                const std::vector<FaceDetection>& dets,
                                const std::string& person_id)
{
    auto info = registry_.find_by_id(person_id);
    std::string person_name = info ? info->name : person_id;
    const auto& cfg = detector_.config();
    int saved = 0;

    for (const auto& d : dets) {
        if (d.confidence < cfg.save_confidence) continue;
        if (d.bbox_px.area() <= 0) continue;

        // Pad 10%
        cv::Rect roi = d.bbox_px;
        int px = static_cast<int>(roi.width  * 0.1f);
        int py = static_cast<int>(roi.height * 0.1f);
        roi.x      = std::max(0, roi.x - px);
        roi.y      = std::max(0, roi.y - py);
        roi.width  = std::min(frame.cols - roi.x, roi.width  + 2*px);
        roi.height = std::min(frame.rows - roi.y, roi.height + 2*py);

        cv::Mat crop;
        cv::resize(frame(roi).clone(), crop, cfg.model_input_size);

        int idx  = ++sample_counter_;
        std::string path = make_face_path(person_id, idx, d.confidence);

        if (!cv::imwrite(path, crop)) continue;
        ++saved;

        // JSON sidecar
        {
            std::ofstream meta(path + ".json");
            meta << "{\n"
                 << "  \"person_id\":   \"" << person_id   << "\",\n"
                 << "  \"person_name\": \"" << person_name << "\",\n"
                 << "  \"confidence\":  "   << d.confidence << ",\n"
                 << "  \"bbox_norm\":   ["  << d.bbox.x << ","
                                            << d.bbox.y << ","
                                            << d.bbox.width << ","
                                            << d.bbox.height << "],\n"
                 << "  \"keypoints\":   [";
            for (size_t k = 0; k < d.keypoints.size(); ++k) {
                meta << "[" << d.keypoints[k].x << ","
                             << d.keypoints[k].y << "]";
                if (k + 1 < d.keypoints.size()) meta << ",";
            }
            meta << "]\n}\n";
        }

        write_manifest_row(path, person_id, person_name, d);
    }

    if (saved > 0) registry_.increment_samples(person_id, saved);
    return saved;
}

// ── Interactive capture ───────────────────────────────────────────────────────

CaptureSession DatasetManager::capture_interactive(int cam_index)
{
    CaptureSession session;

    cv::VideoCapture cap(cam_index);
    if (!cap.isOpened()) {
        std::cerr << "[DatasetManager] Cannot open camera " << cam_index << "\n";
        return session;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    // Warmup
    std::cout << "Warming up camera"; cv::Mat dummy;
    for (int i = 0; i < 20; ++i) { cap >> dummy; std::cout << "."; std::cout.flush(); }
    std::cout << "\n";

    // Ask for initial person
    auto all = registry_.all_persons();
    registry_.print_table();

    std::string current_id;
    std::cout << "Enter person name (or ID) to start: ";
    std::string input; std::getline(std::cin, input);
    if (input.empty()) input = "unknown";

    // Try as ID first
    if (auto p = registry_.find_by_id(input)) current_id = p->id;
    else current_id = register_person(input);

    int saved_total = 0;

    std::cout << "\nControls:\n"
              << "  s       — save current frame faces\n"
              << "  SPACE   — save (same as s)\n"
              << "  n       — switch/add person\n"
              << "  l       — list registered persons\n"
              << "  q/ESC   — quit\n\n";

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) continue;

        auto dets = detector_.detect(frame);
        session.frames_processed++;
        session.faces_detected += static_cast<int>(dets.size());

        // Build visualisation
        cv::Mat vis = detector_.visualise(frame, dets);

        auto cur_info = registry_.find_by_id(current_id);
        std::string label_str = cur_info
            ? "[" + current_id + "] " + cur_info->name
            : current_id;

        // HUD
        cv::rectangle(vis, {0, vis.rows-40, vis.cols, 40},
                      cv::Scalar(20,20,20), cv::FILLED);
        cv::putText(vis,
            "Person: " + label_str +
            "   saved=" + std::to_string(saved_total) +
            "   faces=" + std::to_string(dets.size()),
            {8, vis.rows-12}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
            cv::Scalar(50, 230, 100), 1, cv::LINE_AA);
        cv::putText(vis,
            "s=save  n=switch person  l=list  q=quit",
            {8, vis.rows-30}, cv::FONT_HERSHEY_SIMPLEX, 0.38,
            cv::Scalar(180,180,180), 1, cv::LINE_AA);

        cv::imshow("Dataset Capture", vis);
        int key = cv::waitKey(1);

        if (key == 'q' || key == 27) break;

        if (key == 's' || key == ' ') {
            int n = save_faces(frame, dets, current_id);
            saved_total += n;
            session.faces_saved += n;
            if (n > 0)
                std::cout << "\rSaved " << n << " face(s) → "
                          << label_str << "  total=" << saved_total
                          << "          \n" << std::flush;
            else
                std::cout << "\r[no face above " << detector_.config().save_confidence*100
                          << "% confidence]          " << std::flush;
        }

        if (key == 'n') {
            cv::destroyAllWindows();
            registry_.print_table();
            std::cout << "New person name (or existing ID/name): ";
            std::string nm; std::getline(std::cin, nm);
            if (!nm.empty()) {
                if (auto p = registry_.find_by_id(nm))   current_id = p->id;
                else if (auto p = registry_.find_by_name(nm)) current_id = p->id;
                else current_id = register_person(nm);
                auto info = registry_.find_by_id(current_id);
                std::cout << "Switched to: [" << current_id << "] "
                          << (info ? info->name : "") << "\n";
            }
        }

        if (key == 'l') {
            registry_.print_table();
        }
    }

    cv::destroyAllWindows();
    cap.release();
    session.person_id   = current_id;
    session.person_name = registry_.find_by_id(current_id)
                          .value_or(PersonInfo{}).name;
    print_stats();
    return session;
}

// ── Timed capture ─────────────────────────────────────────────────────────────

CaptureSession DatasetManager::capture_timed(const std::string& person_name,
                                              int cam_index, int seconds,
                                              bool show_preview)
{
    CaptureSession session;
    std::string person_id = register_person(person_name);
    session.person_id   = person_id;
    session.person_name = person_name;

    cv::VideoCapture cap(cam_index);
    if (!cap.isOpened()) {
        std::cerr << "[DatasetManager] Cannot open camera " << cam_index << "\n";
        return session;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    std::cout << "Warming up camera"; cv::Mat dummy;
    for (int i = 0; i < 20; ++i) { cap >> dummy; std::cout << "."; std::cout.flush(); }
    std::cout << "\n";

    auto t_end = chr::steady_clock::now() + chr::seconds(seconds);
    cv::Mat frame;

    std::cout << "[DatasetManager] Capturing " << seconds << "s for ["
              << person_id << "] " << person_name << "\n"
              << "  Press 'q' to stop early.\n";

    while (chr::steady_clock::now() < t_end) {
        cap >> frame;
        if (frame.empty()) continue;
        session.frames_processed++;

        auto dets = detector_.detect(frame);
        session.faces_detected += static_cast<int>(dets.size());

        int n = save_faces(frame, dets, person_id);
        session.faces_saved += n;

        if (show_preview) {
            cv::Mat vis = detector_.visualise(frame, dets);
            int secs_left = static_cast<int>(
                chr::duration_cast<chr::seconds>(
                    t_end - chr::steady_clock::now()).count());
            cv::rectangle(vis, {0, vis.rows-34, vis.cols, 34},
                          cv::Scalar(20,20,20), cv::FILLED);
            cv::putText(vis,
                "[" + person_id + "] " + person_name +
                "  saved=" + std::to_string(session.faces_saved) +
                "  " + std::to_string(secs_left) + "s left",
                {8, vis.rows-10}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(50,230,100), 1, cv::LINE_AA);
            cv::imshow("Dataset Capture — Timed", vis);
            if (cv::waitKey(1) == 'q') break;
        }
    }

    if (show_preview) cv::destroyAllWindows();
    cap.release();

    std::cout << "\n[" << person_id << "] " << person_name
              << " — saved " << session.faces_saved << " faces\n";
    return session;
}

// ── Batch directory import ────────────────────────────────────────────────────

int DatasetManager::import_directory(const std::string& dir_path,
                                      const std::string& force_name)
{
    static const std::vector<std::string> kExts =
        {".jpg",".jpeg",".png",".bmp",".tiff",".tif",".webp"};

    // Collect images grouped by subfolder
    std::map<std::string, std::vector<fs::path>> by_label;
    for (auto& e : fs::recursive_directory_iterator(dir_path)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (std::find(kExts.begin(), kExts.end(), ext) == kExts.end()) continue;
        std::string label = force_name.empty()
                          ? e.path().parent_path().filename().string()
                          : force_name;
        by_label[label].push_back(e.path());
    }

    int total = 0;
    for (auto& [label, paths] : by_label) {
        std::sort(paths.begin(), paths.end());
        std::string pid = register_person(label);
        std::cout << "Importing " << paths.size() << " images → ["
                  << pid << "] " << label << "\n";

        for (int i = 0; i < (int)paths.size(); ++i) {
            cv::Mat img = cv::imread(paths[i].string());
            if (img.empty()) continue;
            auto dets = detector_.detect(img);
            int n = save_faces(img, dets, pid);
            total += n;
            std::cout << "\r  [" << (i+1) << "/" << paths.size()
                      << "] saved=" << n << "  total=" << total
                      << "   " << std::flush;
        }
        std::cout << "\n";
    }
    return total;
}

// ── Stats ─────────────────────────────────────────────────────────────────────

void DatasetManager::print_stats() const {
    registry_.print_table();
}

void DatasetManager::rebuild_manifest() {
    // Close current manifest and rewrite from scratch
    if (manifest_.is_open()) manifest_.close();
    fs::path mp = fs::path(registry_.dataset_root()) / "manifest.csv";
    fs::remove(mp);
    open_manifest();
    std::cout << "[DatasetManager] manifest.csv rebuilt.\n";
}

} // namespace fdet
