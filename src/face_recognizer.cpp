#include "face_recognizer.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>

namespace fs = std::filesystem;

namespace fdet {

static const int CELL = 16;   // cell size in pixels (128/8 = 16)
static const int GRID = 8;    // grid dimension

cv::Mat FaceRecognizer::compute_lbp(const cv::Mat& gray_in) const
{
    cv::Mat gray;
    cv::resize(gray_in, gray, {128, 128});

    // LBP image
    cv::Mat lbp(128, 128, CV_8U, cv::Scalar(0));
    for (int r = 1; r < 127; ++r) {
        for (int c = 1; c < 127; ++c) {
            uchar ctr = gray.at<uchar>(r, c);
            uchar code = 0;
            code |= (gray.at<uchar>(r-1,c-1) >= ctr) << 7;
            code |= (gray.at<uchar>(r-1,c  ) >= ctr) << 6;
            code |= (gray.at<uchar>(r-1,c+1) >= ctr) << 5;
            code |= (gray.at<uchar>(r  ,c+1) >= ctr) << 4;
            code |= (gray.at<uchar>(r+1,c+1) >= ctr) << 3;
            code |= (gray.at<uchar>(r+1,c  ) >= ctr) << 2;
            code |= (gray.at<uchar>(r+1,c-1) >= ctr) << 1;
            code |= (gray.at<uchar>(r  ,c-1) >= ctr) << 0;
            lbp.at<uchar>(r, c) = code;
        }
    }

    // Build feature vector: concatenate cell histograms (256 bins each)
    int n_cells = GRID * GRID;
    cv::Mat feat(1, n_cells * 256, CV_32F, cv::Scalar(0));
    for (int gr = 0; gr < GRID; ++gr) {
        for (int gc = 0; gc < GRID; ++gc) {
            cv::Rect roi(gc*CELL, gr*CELL, CELL, CELL);
            cv::Mat cell = lbp(roi);
            cv::Mat hist;
            int channels[] = {0};
            int histSize[] = {256};
            float range[]  = {0, 256};
            const float* ranges[] = {range};
            cv::calcHist(&cell, 1, channels, cv::Mat(), hist, 1, histSize, ranges);
            
            // Normalise cell histogram
            hist /= (float)(CELL * CELL);
            
            // Copy into feature row
            hist.reshape(1, 1).copyTo(
                feat(cv::Rect((gr*GRID+gc)*256, 0, 256, 1)));
        }
    }
    return feat; // 1 × (64*256) = 1 × 16384
}

int FaceRecognizer::train(const std::string& dataset_root,
                           const std::map<std::string,std::string>& id_to_name)
{
    static const std::vector<std::string> kExts = {".jpg",".jpeg",".png",".bmp"};

    persons_.clear();
    id_to_idx_.clear();
    label_to_id_.clear();
    label_to_name_.clear();
    total_samples_ = 0;

    if (!fs::exists(dataset_root)) {
        std::cerr << "[Recognizer] Dataset root not found: " << dataset_root << "\n";
        trained_ = false;
        return 0;
    }

    std::vector<fs::path> subdirs;
    for (auto& e : fs::directory_iterator(dataset_root)) {
        if (e.is_directory()) subdirs.push_back(e.path());
    }
    std::sort(subdirs.begin(), subdirs.end());

    int idx = 0;
    for (auto& dir : subdirs) {
        std::string dname = dir.filename().string();
        if (dname.rfind('.', 0) == 0) continue; // skip hidden

        std::string person_id = dname.substr(0, dname.find('_'));
        std::string person_name = dname.size() > person_id.size() + 1
                                ? dname.substr(person_id.size() + 1)
                                : person_id;
        if (auto it = id_to_name.find(person_id); it != id_to_name.end())
            person_name = it->second;

        PersonModel pm;
        pm.id   = person_id;
        pm.name = person_name;

        for (auto& f : fs::directory_iterator(dir)) {
            if (!f.is_regular_file()) continue;
            std::string ext = f.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (std::find(kExts.begin(), kExts.end(), ext) == kExts.end()) continue;

            cv::Mat img = cv::imread(f.path().string(), cv::IMREAD_GRAYSCALE);
            if (img.empty()) continue;
            
            // Предобработка картинок датасета для консистентности базы
            cv::equalizeHist(img, img);
            pm.histograms.push_back(compute_lbp(img));
        }

        if (pm.histograms.empty()) continue;

        id_to_idx_[person_id]  = idx;
        label_to_id_[idx]      = person_id;
        label_to_name_[idx]    = person_name;
        total_samples_        += (int)pm.histograms.size();
        persons_.push_back(std::move(pm));

        std::cout << "  [" << person_id << "] " << person_name
                  << " — " << persons_.back().histograms.size() << " samples\n";
        ++idx;
    }

    trained_ = !persons_.empty();
    if (trained_)
        std::cout << "[Recognizer] Trained: "
                  << persons_.size() << " person(s), "
                  << total_samples_  << " samples\n";
    else
        std::cerr << "[Recognizer] No samples found.\n";

    return total_samples_;
}

RecognitionResult FaceRecognizer::recognize(const cv::Mat& face_crop) const
{
    RecognitionResult res;
    if (!trained_ || face_crop.empty()) return res;

    cv::Mat gray;
    if (face_crop.channels() == 3)
        cv::cvtColor(face_crop, gray, cv::COLOR_BGR2GRAY);
    else
        gray = face_crop.clone();

    cv::equalizeHist(gray, gray);

    cv::Mat query = compute_lbp(gray);

    double best_dist = 1e18;
    int    best_idx  = -1;

    for (int i = 0; i < (int)persons_.size(); ++i) {
        double person_dist = 1e18;
        for (auto& h : persons_[i].histograms) {
            double d = cv::compareHist(query, h, cv::HISTCMP_CHISQR_ALT);
            if (d < person_dist) person_dist = d;
        }
        if (person_dist < best_dist) {
            best_dist = person_dist;
            best_idx  = i;
        }
    }

    if (best_idx < 0) return res;

    res.confidence   = best_dist ;
    res.person_id    = label_to_id_.at(best_idx);
    res.person_name  = label_to_name_.at(best_idx);

    // Вывод отладки для калибровки threshold
    std::cout << "[Debug] Best raw dist: " << best_dist << " | Confidence: " << res.confidence << "\n";

    return res;
}

bool FaceRecognizer::save_model(const std::string& path) const
{
    if (!trained_) return false;
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;

    fs << "num_persons" << (int)persons_.size();
    for (int i = 0; i < (int)persons_.size(); ++i) {
        std::string key = "person_" + std::to_string(i);
        fs << key + "_id"   << persons_[i].id;
        fs << key + "_name" << persons_[i].name;
        fs << key + "_n"    << (int)persons_[i].histograms.size();
        for (int j = 0; j < (int)persons_[i].histograms.size(); ++j)
            fs << key + "_h" + std::to_string(j) << persons_[i].histograms[j];
    }
    return true;
}

bool FaceRecognizer::load_model(const std::string& path)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;

    persons_.clear(); id_to_idx_.clear();
    label_to_id_.clear(); label_to_name_.clear();
    total_samples_ = 0;

    int np = (int)fs["num_persons"];
    for (int i = 0; i < np; ++i) {
        std::string key = "person_" + std::to_string(i);
        PersonModel pm;
        fs[key+"_id"]   >> pm.id;
        fs[key+"_name"] >> pm.name;
        int n = (int)fs[key+"_n"];
        for (int j = 0; j < n; ++j) {
            cv::Mat h; fs[key+"_h"+std::to_string(j)] >> h;
            pm.histograms.push_back(h);
        }
        id_to_idx_[pm.id]   = i;
        label_to_id_[i]     = pm.id;
        label_to_name_[i]   = pm.name;
        total_samples_     += n;
        persons_.push_back(std::move(pm));
    }
    trained_ = !persons_.empty();
    if (trained_)
        std::cout << "[Recognizer] Loaded model: "
                  << persons_.size() << " person(s)\n";
    return trained_;
}

} // namespace fdet