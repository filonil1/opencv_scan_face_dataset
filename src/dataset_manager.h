#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  DatasetManager — высокоуровневый API для построения датасета лиц
//
//  Структура на диске:
//  dataset/
//  ├── registry.json              ← PersonRegistry (ID ↔ Name)
//  ├── manifest.csv               ← все записи: path, id, name, conf, bbox…
//  ├── 001_Alice/
//  │   ├── 001_Alice_0001_0.97.jpg       ← лицо
//  │   └── 001_Alice_0001_0.97.jpg.json  ← bbox + keypoints
//  ├── 002_Bob/
//  └── …
// ─────────────────────────────────────────────────────────────────────────────
#include "face_detector.h"
#include "person_registry.h"

#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace fdet {

struct CaptureSession {
    std::string person_id;
    std::string person_name;
    int         frames_processed{0};
    int         faces_detected{0};
    int         faces_saved{0};
};

class DatasetManager {
public:
    DatasetManager(FaceDetector& detector,
                   const std::string& dataset_root = "dataset");
    ~DatasetManager();

    // ── Registration ──────────────────────────────────────────────────────
    /// Register a person and return their ID (creates folder).
    std::string register_person(const std::string& name);

    /// List all registered persons.
    void list_persons() const;

    // ── Capture ───────────────────────────────────────────────────────────
    /// Interactive webcam session.
    /// Shows live preview; s=save current frame, q=quit, 1-9=switch person.
    CaptureSession capture_interactive(int cam_index = 0);

    /// Timed auto-capture: saves every good frame for `seconds`.
    CaptureSession capture_timed(const std::string& person_name,
                                  int cam_index, int seconds,
                                  bool show_preview = true);

    // ── Batch import ──────────────────────────────────────────────────────
    /// Import images from a folder. Subfolder name → person name.
    int import_directory(const std::string& dir_path,
                          const std::string& force_name = "");

    // ── Save helpers ──────────────────────────────────────────────────────
    /// Save detected faces for a known person. Returns count saved.
    int save_faces(const cv::Mat& frame,
                   const std::vector<FaceDetection>& dets,
                   const std::string& person_id);

    // ── Registry access ───────────────────────────────────────────────────
    PersonRegistry&       registry()       { return registry_; }
    const PersonRegistry& registry() const { return registry_; }

    // ── Stats / export ────────────────────────────────────────────────────
    void print_stats() const;

    /// Rebuild manifest.csv from all files on disk.
    void rebuild_manifest();

private:
    std::string make_face_path(const std::string& person_id,
                                int sample_index, float conf) const;
    void write_manifest_row(const std::string& path,
                             const std::string& person_id,
                             const std::string& person_name,
                             const FaceDetection& d);
    void open_manifest();

    FaceDetector&  detector_;
    PersonRegistry registry_;
    std::ofstream  manifest_;
    int            sample_counter_{0};
};

} // namespace fdet
