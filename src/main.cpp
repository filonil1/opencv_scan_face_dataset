#include "face_detector.h"
#include "face_recognizer.h"
#include "dataset_manager.h"
#include "live_window.h"

#include <opencv2/highgui.hpp>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace fdet;

static DetectorConfig make_config() {
    DetectorConfig cfg;
    cfg.min_confidence   = 0.25f;   // LOW — detect even in bad lighting
    cfg.save_confidence  = 0.70f;   // save faces above 70%
    cfg.max_faces        = 10;
    cfg.dataset_root     = "dataset";
    cfg.model_input_size = {128, 128};
    cfg.use_python_bridge = false;   // <-- добавить
    cfg.use_opencv_dnn = true;       // <-- добавить
    cfg.use_mediapipe = false;       // <-- добавить (опционально)

    return cfg;
}

// ─── MODE 1: Register (self-photo) ────────────────────────────────────────────
static void mode_register(FaceDetector& det, FaceRecognizer& rec, DatasetManager& dm)
{
    std::cout << "\n╔══════════════════════════════════════╗\n"
              << "║  MODE 1 — Register yourself          ║\n"
              << "╚══════════════════════════════════════╝\n\n";
    dm.list_persons();

    std::cout << "Your name: ";
    std::string name; std::getline(std::cin, name);
    if (name.empty()) { std::cout << "Name cannot be empty.\n"; return; }

    std::string pid = dm.register_person(name);
    std::cout << "\n✓ ID assigned: [" << pid << "] " << name << "\n\n"
              << "Look at the camera and press SPACE to take photos.\n"
              << "Recommended: 30+ photos (different angles).\n"
              << "Press Q when done.\n\n";

    cv::VideoCapture cap(0);
    std::cerr << "Opening camera..." << std::endl;
    if (!cap.isOpened()) { std::cerr << "Cannot open camera\n"; return; }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    std::cout << "Warming up";
    cv::Mat dummy;
    for (int i = 0; i < 20; ++i) { cap >> dummy; std::cout<<"."; std::cout.flush(); }
    std::cout << "\n\n";

    const int TARGET = 30;
    int saved = 0;
    cv::Mat frame;

    while (true) {
        cap >> frame;
        if (frame.empty()) continue;

        auto dets = det.detect(frame);
        cv::Mat vis = frame.clone();

        // Draw boxes
        for (auto& d : dets) {
            bool good = d.confidence >= det.config().save_confidence;
            cv::Scalar col = good ? cv::Scalar(50,220,50) : cv::Scalar(0,160,255);
            cv::rectangle(vis, d.bbox_px, col, 2);
            for (auto& kp : d.keypoints)
                cv::circle(vis, kp, 3, cv::Scalar(255,100,30), cv::FILLED);

            // conf label
            std::ostringstream ls;
            ls << std::fixed << std::setprecision(0) << d.confidence*100 << "%";
            cv::putText(vis, ls.str(),
                        {d.bbox_px.x+4, d.bbox_px.y-6},
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv::LINE_AA);
        }

        // Progress bar
        float pct = std::min(1.0f, saved/(float)TARGET);
        int bar_y = vis.rows - 58;
        cv::rectangle(vis, {0,bar_y}, {vis.cols,bar_y+6}, cv::Scalar(40,40,40), cv::FILLED);
        cv::rectangle(vis, {0,bar_y}, {(int)(vis.cols*pct),bar_y+6},
                      saved>=TARGET ? cv::Scalar(50,220,50) : cv::Scalar(0,150,255),
                      cv::FILLED);

        // HUD
        cv::rectangle(vis, {0,vis.rows-52}, {vis.cols,vis.rows},
                      cv::Scalar(12,12,12), cv::FILLED);

        std::string status_line = dets.empty()
            ? "○ No face detected — move closer!"
            : "● Face detected  conf=" + std::to_string((int)(dets[0].confidence*100)) + "%";
        cv::Scalar sc = dets.empty() ? cv::Scalar(80,80,255) : cv::Scalar(50,220,50);
        cv::putText(vis, status_line, {10, vis.rows-34},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, sc, 1, cv::LINE_AA);

        cv::putText(vis,
            "[" + pid + "] " + name +
            "   photos: " + std::to_string(saved) + "/" + std::to_string(TARGET) +
            (saved>=TARGET ? "  ✓ READY" : ""),
            {10, vis.rows-12},
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(50,230,100), 1, cv::LINE_AA);

        cv::putText(vis, "SPACE=capture  Q=done",
                    {vis.cols-200, vis.rows-12},
                    cv::FONT_HERSHEY_SIMPLEX, 0.38,
                    cv::Scalar(140,140,140), 1, cv::LINE_AA);

        cv::imshow("Register: " + name + "  [" + pid + "]", vis);
        int key = cv::waitKey(1);
        if (key=='q'||key==27) break;

        if (key==' '||key=='s') {
            int n = dm.save_faces(frame, dets, pid);
            saved += n;
            if (n > 0)
                std::cout << "\rPhoto " << saved << "/" << TARGET
                          << (saved>=TARGET ? " ✓ enough! press Q when ready" : "")
                          << "    " << std::flush;
            else
                std::cout << "\rNo face above threshold — come closer!   " << std::flush;
        }
    }
    cv::destroyAllWindows(); cap.release();

    std::cout << "\n\nDone. [" << pid << "] " << name
              << " — " << saved << " photos saved.\n";
    if (saved >= 5) {
        std::cout << "Training recognizer …\n";
        int n = rec.train(dm.registry().dataset_root(), {});
        if (n > 0) {
            rec.save_model("dataset/model.yml");
            std::cout << "Model saved → dataset/model.yml\n";
        }
        std::cout << "✓ Ready for auto-recognition (choose mode 2)\n\n";
    } else {
        std::cout << "Need ≥5 photos to train. Run register again.\n\n";
    }
}

// ─── MODE 2: Auto recognition ─────────────────────────────────────────────────
static void mode_auto(FaceDetector& det, FaceRecognizer& rec, DatasetManager& dm)
{
    std::cerr << "=== mode_auto started ===" << std::endl;
    std::cout << "\n╔══════════════════════════════════════╗\n"
              << "║  MODE 2 — Auto Recognition           ║\n"
              << "╚══════════════════════════════════════╝\n\n";

    if (!rec.is_trained()) {
        std::cout << "Training …\n";
        int n = rec.train(dm.registry().dataset_root(), {});
        if (n == 0) {
            std::cout << "No data! Use mode 1 to register first.\n\n";
            return;
        }
    }
    std::cout << "Ready: " << rec.num_persons() << " person(s), "
              << rec.num_samples() << " samples\n\n"
              << "t=retrain  l=list  q=quit\n\n";
    std::cerr << "Opening camera..." << std::endl;
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) { std::cerr << "Cannot open camera\n"; return; }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    std::cout << "Warming up";
    cv::Mat dummy;
    for (int i=0;i<20;++i){cap>>dummy;std::cout<<".";std::cout.flush();}
    std::cout << " go!\n\n";

    double fps=0; int fc=0;
    auto t0 = std::chrono::steady_clock::now();
    cv::Mat frame;

    while (true) {
        cap >> frame;
        if (frame.empty()) continue;
        ++fc;
        auto now = std::chrono::steady_clock::now();
        double el = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now-t0).count()/1000.0;
        if (el>=0.5){fps=fc/el;fc=0;t0=now;}

        auto dets = det.detect(frame);
        cv::Mat vis = frame.clone();

        for (auto& d : dets) {
            RecognitionResult r;
            if (d.bbox_px.area()>0) r = rec.recognize(frame(d.bbox_px));

            cv::Scalar col = r.recognized()
                ? cv::Scalar(50,220,50) : cv::Scalar(0,160,255);
            cv::rectangle(vis, d.bbox_px, col, 2);

            // Corner ticks
            int tk = std::min(d.bbox_px.width, d.bbox_px.height)/5;
            auto b = d.bbox_px;
            auto L = [&](cv::Point a,cv::Point c){cv::line(vis,a,c,col,3);};
            L(b.tl(),b.tl()+cv::Point(tk,0));  L(b.tl(),b.tl()+cv::Point(0,tk));
            cv::Point tr(b.x+b.width,b.y);
            L(tr,tr+cv::Point(-tk,0)); L(tr,tr+cv::Point(0,tk));
            cv::Point bl(b.x,b.y+b.height);
            L(bl,bl+cv::Point(tk,0)); L(bl,bl+cv::Point(0,-tk));
            L(b.br(),b.br()+cv::Point(-tk,0)); L(b.br(),b.br()+cv::Point(0,-tk));

            for (auto& kp : d.keypoints)
                cv::circle(vis, kp, 3, cv::Scalar(255,100,30), cv::FILLED);

            // Pill
            std::string top = r.recognized()
                ? "[" + r.person_id + "] " + r.person_name : "Unknown";
            std::ostringstream bot_ss;
            if (r.recognized())
                bot_ss << std::fixed << std::setprecision(0)
                       << std::max(0.0,100.0-r.confidence) << "% match";
            else
                bot_ss << "det=" << (int)(d.confidence*100) << "%";
            std::string bot = bot_ss.str();

            int font=cv::FONT_HERSHEY_SIMPLEX, base=0;
            cv::Size s1=cv::getTextSize(top,font,0.6,1,&base);
            cv::Size s2=cv::getTextSize(bot,font,0.4,1,&base);
            int pw=std::max(s1.width,s2.width)+14, ph=s1.height+s2.height+16;
            int px=b.x, py=std::max(0,b.y-ph-4);
            cv::rectangle(vis,{px,py},{px+pw,py+ph},cv::Scalar(20,20,20),cv::FILLED);
            cv::rectangle(vis,{px,py},{px+pw,py+ph},col,1);
            cv::putText(vis,top,{px+7,py+s1.height+4},font,0.6,col,1,cv::LINE_AA);
            cv::putText(vis,bot,{px+7,py+s1.height+s2.height+12},
                        font,0.4,cv::Scalar(170,170,170),1,cv::LINE_AA);
        }

        // HUD
        cv::rectangle(vis,{0,vis.rows-48},{vis.cols,vis.rows},
                      cv::Scalar(12,12,12),cv::FILLED);
        std::string det_status = dets.empty()
            ? "○ No face — move closer / improve lighting"
            : "● " + std::to_string(dets.size()) + " face(s) detected";
        cv::Scalar sc = dets.empty() ? cv::Scalar(80,80,255) : cv::Scalar(50,220,50);
        cv::putText(vis, det_status, {10,vis.rows-28},
                    cv::FONT_HERSHEY_SIMPLEX,0.5,sc,1,cv::LINE_AA);
        std::ostringstream hud;
        hud<<std::fixed<<std::setprecision(1)<<fps<<" fps   "
           <<rec.num_persons()<<" person(s) in DB   t=retrain  l=list  q=quit";
        cv::putText(vis,hud.str(),{10,vis.rows-10},
                    cv::FONT_HERSHEY_SIMPLEX,0.38,cv::Scalar(120,120,120),1,cv::LINE_AA);

        cv::imshow("Auto Recognition  |  " + det.active_backend(), vis);
        int key = cv::waitKey(1);
        if (key=='q'||key==27) break;
        if (key=='t'){
            std::cout<<"\nRetraining …\n";
            int n=rec.train(dm.registry().dataset_root(),{});
            if(n>0)rec.save_model("dataset/model.yml");
            std::cout<<"Done — "<<n<<" samples\n";
        }
        if (key=='l') dm.list_persons();
    }
    cv::destroyAllWindows(); cap.release();
}

// ─── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    auto cfg = make_config();
    FaceDetector   det(cfg);
    FaceRecognizer rec;
    DatasetManager dm(det, cfg.dataset_root);

    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║         Face Dataset Builder             ║\n"
              << "╚══════════════════════════════════════════╝\n"
              << "  Backend : " << det.active_backend() << "\n"
              << "  Dataset : " << cfg.dataset_root << "/\n"
              << "  Detect threshold : " << (int)(cfg.min_confidence*100) << "%\n"
              << "  Save threshold   : " << (int)(cfg.save_confidence*100) << "%\n\n";

    if (fs::exists("dataset/model.yml")) {
        rec.load_model("dataset/model.yml");
        std::cout << "Loaded saved model\n";
    }

    // CLI shortcut
    if (argc >= 2) {
        std::string cmd = argv[1];
        if      (cmd=="register") mode_register(det,rec,dm);
        else if (cmd=="auto")     mode_auto(det,rec,dm);
        else if (cmd=="list")     dm.list_persons();
        else if (cmd=="train") {
            int n=rec.train(cfg.dataset_root,{});
            if(n>0)rec.save_model("dataset/model.yml");
        }
        return 0;
    }

    // Interactive menu
    while (true) {
        std::cout << "\n┌──────────────────────────────────────┐\n"
                  << "│  1 — Register (take selfie → get ID) │\n"
                  << "│  2 — Auto recognition (live camera)  │\n"
                  << "│  3 — List persons                    │\n"
                  << "│  4 — Retrain model                   │\n"
                  << "│  0 — Exit                            │\n"
                  << "└──────────────────────────────────────┘\n"
                  << "Choose: ";
        std::string c; std::getline(std::cin, c);
        if      (c=="1") mode_register(det,rec,dm);
        else if (c=="2") mode_auto(det,rec,dm);
        else if (c=="3") dm.list_persons();
        else if (c=="4") {
            int n=rec.train(cfg.dataset_root,{});
            if(n>0){rec.save_model("dataset/model.yml");
                    std::cout<<"Model saved\n";}
        }
        else if (c=="0") break;
    }
    return 0;
}