#include "live_window.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace chr = std::chrono;
namespace fs  = std::filesystem;

namespace fdet {

LiveWindow::LiveWindow(FaceDetector&   det,
                       FaceRecognizer& rec,
                       DatasetManager& dm)
    : detector_(det), recognizer_(rec), dm_(dm) {}

// ── Draw one face ─────────────────────────────────────────────────────────────
void LiveWindow::draw_face(cv::Mat& vis,
                            const FaceDetection& det,
                            const RecognitionResult& rec) const
{
    const cv::Rect& b = det.bbox_px;
    cv::Scalar col = rec.recognized()
        ? cv::Scalar(50, 220, 50)   // green  = known
        : cv::Scalar(0, 160, 255);  // orange = unknown

    // Box
    cv::rectangle(vis, b, col, 2);

    // Corner ticks
    int tk = std::min(b.width, b.height) / 5;
    auto L = [&](cv::Point a, cv::Point c){ cv::line(vis,a,c,col,3); };
    L(b.tl(), b.tl()+cv::Point(tk,0));  L(b.tl(), b.tl()+cv::Point(0,tk));
    cv::Point tr(b.x+b.width,b.y);
    L(tr, tr+cv::Point(-tk,0)); L(tr, tr+cv::Point(0,tk));
    cv::Point bl(b.x,b.y+b.height);
    L(bl, bl+cv::Point(tk,0));  L(bl, bl+cv::Point(0,-tk));
    L(b.br(), b.br()+cv::Point(-tk,0)); L(b.br(), b.br()+cv::Point(0,-tk));

    // Keypoints
    for (auto& kp : det.keypoints)
        cv::circle(vis, kp, 3, cv::Scalar(255,100,30), cv::FILLED);

    // Label pill
    std::string top_lbl, bot_lbl;
    if (rec.recognized()) {
        top_lbl = "[" + rec.person_id + "] " + rec.person_name;
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(0)
           << std::max(0.0, 100.0 - rec.confidence) << "% match";
        bot_lbl = ss.str();
    } else {
        top_lbl = "Unknown";
        // Выводим в интерфейс текущую минимальную дистанцию для калибровки порога
        std::ostringstream ss;
        ss << "det=" << (int)(det.confidence*100) << "% dist=" << std::fixed << std::setprecision(1) << rec.confidence;
        bot_lbl = ss.str();
    }

    int font = cv::FONT_HERSHEY_SIMPLEX, base = 0;
    cv::Size s1 = cv::getTextSize(top_lbl, font, 0.58, 1, &base);
    cv::Size s2 = cv::getTextSize(bot_lbl, font, 0.38, 1, &base);
    int pw = std::max(s1.width, s2.width) + 14;
    int ph = s1.height + s2.height + 16;
    int px = b.x, py = std::max(0, b.y - ph - 4);
    cv::rectangle(vis, {px,py}, {px+pw,py+ph}, cv::Scalar(20,20,20), cv::FILLED);
    cv::rectangle(vis, {px,py}, {px+pw,py+ph}, col, 1);
    cv::putText(vis, top_lbl, {px+7, py+s1.height+4}, font, 0.58, col, 1, cv::LINE_AA);
    cv::putText(vis, bot_lbl, {px+7, py+s1.height+s2.height+12},
                font, 0.38, cv::Scalar(170,170,170), 1, cv::LINE_AA);
}

// ── HUD bar ───────────────────────────────────────────────────────────────────
void LiveWindow::draw_hud(cv::Mat& vis,
                           const std::string& person_label,
                           int face_count, int saved_total,
                           double fps) const
{
    int bh = 56;
    cv::rectangle(vis, {0,vis.rows-bh}, {vis.cols,vis.rows},
                  cv::Scalar(12,12,12), cv::FILLED);

    // Top row: person + counters
    cv::putText(vis, "Saving as: " + person_label,
                {10, vis.rows-bh+20},
                cv::FONT_HERSHEY_SIMPLEX, 0.58,
                cv::Scalar(50,230,100), 1, cv::LINE_AA);

    // Bottom row: stats + controls
    std::ostringstream bot;
    bot << std::fixed << std::setprecision(1) << fps
        << " fps   faces=" << face_count
        << "   saved=" << saved_total
        << "     SPACE/s=save  n=person  t=train  q=quit";
    cv::putText(vis, bot.str(), {10, vis.rows-10},
                cv::FONT_HERSHEY_SIMPLEX, 0.38,
                cv::Scalar(130,130,130), 1, cv::LINE_AA);

    // Detection status indicator
    std::string status = face_count > 0
        ? "● DETECTING"
        : "○ no face (move closer / better light)";
    cv::Scalar sc = face_count > 0 ? cv::Scalar(50,220,50) : cv::Scalar(80,80,255);
    int base=0;
    cv::Size ts = cv::getTextSize(status, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &base);
    cv::putText(vis, status,
                {vis.cols - ts.width - 10, vis.rows-bh+20},
                cv::FONT_HERSHEY_SIMPLEX, 0.45, sc, 1, cv::LINE_AA);
}

// ── Main run loop ─────────────────────────────────────────────────────────────
void LiveWindow::run(int cam_index)
{
    cv::VideoCapture cap(cam_index);
    if (!cap.isOpened()) {
        std::cerr << "[LiveWindow] Cannot open camera " << cam_index << "\n";
        return;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    // Warmup
    std::cout << "Warming up camera";
    cv::Mat dummy;
    for (int i = 0; i < 20; ++i) { cap >> dummy; std::cout << "."; std::cout.flush(); }
    std::cout << "\n";

    dm_.list_persons();
    std::cout << "\nEnter your name (or ID) — Enter for 'unknown': ";
    std::string input; std::getline(std::cin, input);
    if (input.empty()) input = "unknown";

    std::string current_id;
    if (auto p = dm_.registry().find_by_id(input))        current_id = p->id;
    else if (auto p = dm_.registry().find_by_name(input)) current_id = p->id;
    else current_id = dm_.register_person(input);

    // Train recognizer if data exists
    if (!recognizer_.is_trained()) {
        std::cout << "[Recognizer] Training from dataset …\n";
        recognizer_.train(dm_.registry().dataset_root(), {});
    }

    std::cout << "\nControls: SPACE/s=save  n=switch person  t=retrain  l=list  q=quit\n\n";

    int    saved_total = 0;
    double fps = 0; int fc = 0;
    auto   t0  = chr::steady_clock::now();

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) continue;

        ++fc;
        auto now = chr::steady_clock::now();
        double el = chr::duration_cast<chr::milliseconds>(now-t0).count()/1000.0;
        if (el >= 0.5) { fps = fc/el; fc = 0; t0 = now; }

        // ── Детекция лиц ─────────────────────────────────────────────────────
        auto dets = detector_.detect(frame);

        // ── Распознавание каждого лица ──────────────────────────────────────
        std::vector<RecognitionResult> recs(dets.size());
        for (size_t i = 0; i < dets.size(); ++i) {
            if (dets[i].bbox_px.area() > 0) {
                recs[i] = recognizer_.recognize(frame(dets[i].bbox_px));
                
                // Гарантированный вывод в консоль для дебага
                std::cout << "[RECOGNIZE DIAL] Face " << i 
                          << " | Computed Dist: " << recs[i].confidence
                          << " | Recognized: " << (recs[i].recognized() ? "YES" : "NO") 
                          << std::endl;
            }
        }

        // ── Отрисовка ───────────────────────────────────────────────────────
        cv::Mat vis = frame.clone();
        for (size_t i = 0; i < dets.size(); ++i)
            draw_face(vis, dets[i], recs[i]);

        auto cur_info = dm_.registry().find_by_id(current_id);
        std::string plabel = cur_info
            ? "[" + current_id + "] " + cur_info->name : current_id;
        draw_hud(vis, plabel, (int)dets.size(), saved_total, fps);

        cv::imshow("FaceDetector  |  " + detector_.active_backend(), vis);
        int key = cv::waitKey(1);

        if (key=='q' || key==27) break;

        if (key==' ' || key=='s') {
            int n = dm_.save_faces(frame, dets, current_id);
            saved_total += n;
            if (n > 0)
                std::cout << "\rSaved " << n << " face(s) → " << plabel
                          << "  total=" << saved_total << "    \n";
            else
                std::cout << "\rNo face saved (conf<"
                          << (int)(detector_.config().save_confidence*100)
                          << "% or not detected)   ";
            std::cout.flush();
        }

        if (key=='n') {
            cv::destroyAllWindows();
            dm_.list_persons();
            std::cout << "Name or ID: ";
            std::string nm; std::getline(std::cin, nm);
            if (!nm.empty()) {
                if (auto p = dm_.registry().find_by_id(nm))        current_id = p->id;
                else if (auto p = dm_.registry().find_by_name(nm)) current_id = p->id;
                else current_id = dm_.register_person(nm);
            }
        }

        if (key=='t') {
            std::cout << "\nRetraining …\n";
            int n = recognizer_.train(dm_.registry().dataset_root(), {});
            if (n > 0) recognizer_.save_model("dataset/model.yml");
            std::cout << "Done — " << n << " samples\n";
        }

        if (key=='l') dm_.list_persons();
    }

    cv::destroyAllWindows();
    cap.release();
}

} // namespace fdet