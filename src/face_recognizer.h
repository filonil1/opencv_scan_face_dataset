#ifndef FACE_RECOGNIZER_H
#define FACE_RECOGNIZER_H

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <map>

namespace fdet {

struct RecognitionResult {
    std::string person_id;
    std::string person_name;
    double      confidence{9999.0}; // Минимальное расстояние хи-квадрат
    
    bool recognized() const { return confidence < threshold; }
    
    // ВРЕМЕННО подняли порог до 120.0, чтобы локализовать проблему.
    // Сделай калибровку на основе вывода в консоль!
    static constexpr double threshold = 110.1; 
};

struct PersonModel {
    std::string id;
    std::string name;
    std::vector<cv::Mat> histograms;
};

class FaceRecognizer {
public:
    FaceRecognizer() = default;

    int train(const std::string& dataset_root,
              const std::map<std::string,std::string>& id_to_name);

    RecognitionResult recognize(const cv::Mat& face_crop) const;

    bool save_model(const std::string& path) const;
    bool load_model(const std::string& path);

    bool is_trained() const { return trained_; }
    
    // ИСПРАВЛЕНО: Геттеры переименованы/добавлены для полной совместимости с main.cpp
    int  num_persons() const { return (int)persons_.size(); }
    int  num_samples() const { return total_samples_; }
    int  total_samples() const { return total_samples_; } // Оставляем для обратной совместимости

private:
    // Метод const, полностью соответствует реализации в face_recognizer.cpp
    cv::Mat compute_lbp(const cv::Mat& gray_in) const;

    bool                             trained_{false};
    int                              total_samples_{0};
    std::vector<PersonModel>         persons_;
    std::map<std::string, int>       id_to_idx_;
    std::map<int, std::string>       label_to_id_;
    std::map<int, std::string>       label_to_name_;
};

} // namespace fdet

#endif // FACE_RECOGNIZER_H