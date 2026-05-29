#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  LiveWindow — real-time camera view with face detection + recognition overlay
// ─────────────────────────────────────────────────────────────────────────────
#include "face_detector.h"
#include "face_recognizer.h"
#include "dataset_manager.h"
#include <string>

namespace fdet {

class LiveWindow {
public:
    LiveWindow(FaceDetector&    detector,
               FaceRecognizer&  recognizer,
               DatasetManager&  dm);

    // Open camera window. Shows:
    //   - Bounding box per face
    //   - [ID] Name  conf%  (green = recognised, orange = unknown)
    //   - Detection confidence
    //   - Keypoints (dots)
    //   - HUD: person count, FPS, current capture label
    //
    // Controls:
    //   s / SPACE  — save current faces for `current_person`
    //   n          — switch / add person
    //   t          — retrain recognizer from current dataset
    //   l          — list persons
    //   q / ESC    — quit
    void run(int cam_index = 0);

private:
    void draw_face(cv::Mat& vis,
                   const FaceDetection& det,
                   const RecognitionResult& rec) const;
    void draw_hud(cv::Mat& vis,
                  const std::string& person_label,
                  int face_count,
                  int saved_total,
                  double fps) const;

    FaceDetector&   detector_;
    FaceRecognizer& recognizer_;
    DatasetManager& dm_;
};

} // namespace fdet