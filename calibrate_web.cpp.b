#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <thread>
#include <atomic>
#include <mutex>
#include "httplib.h"

using namespace cv;
using namespace std;
#include <fstream>
#include <ctime>
#include <iomanip>
#include <numeric>

// Функция получения текущего времени
string getCurrentTime() {
    time_t now = time(nullptr);
    tm* ltm = localtime(&now);
    ostringstream oss;
    oss << put_time(ltm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

void saveMonoCalibrationYML(const string& cam_id, const Mat& K, const Mat& D,
                             int frames, Size img_size, double rms_error,
                             const vector<double>& per_view_errors, const string& output_dir) {
    string filepath = output_dir + "/cam_" + cam_id + "_calibration.yml";
    FileStorage fs(filepath, FileStorage::WRITE);
    
    fs << "camera_matrix" << K;
    fs << "distortion_coefficients" << D;
    fs << "image_width" << img_size.width;
    fs << "image_height" << img_size.height;
    fs << "reprojection_error" << rms_error;
    fs << "frames_used" << frames;
    fs << "calibration_time" << getCurrentTime();
    fs << "per_view_errors" << per_view_errors;
    
    fs.release();
    cout << "Saved: " << filepath << endl;
}

void saveStereoCalibrationYML(const string& cam1_id, const string& cam2_id,
                               const Mat& R, const Mat& T, const Mat& E, const Mat& F,
                               double rms, int frames, const string& output_dir) {
    string filepath = output_dir + "/stereo_" + cam1_id + "_" + cam2_id + "_calibration.yml";
    FileStorage fs(filepath, FileStorage::WRITE);
    
    fs << "camera_1_id" << cam1_id;
    fs << "camera_2_id" << cam2_id;
    fs << "rotation_matrix" << R;
    fs << "translation_vector" << T;
    fs << "essential_matrix" << E;
    fs << "fundamental_matrix" << F;
    fs << "stereo_rms_error" << rms;
    fs << "frames_used" << frames;
    fs << "calibration_time" << getCurrentTime();
    
    fs.release();
    cout << "Saved: " << filepath << endl;
}

void saveCalibrationResultsJSON(const vector<string>& new_cam_ids, const string& output_dir) {
    string filepath = output_dir + "/calibration_results.json";
    
    // Структура для хранения результатов
    struct MonoCalib {
        string id;
        string file;
        string time;
        int frames;
        int width;
        int height;
        double error;
    };
    
    map<string, MonoCalib> existing_calibs;  // ключ = camera_id
    
    // Прочитать существующий JSON если есть
    ifstream existing_file(filepath);
    if (existing_file.is_open()) {
        cout << "Reading existing calibration_results.json..." << endl;
        string line;
        bool in_mono = false;
        MonoCalib current;
        
        while (getline(existing_file, line)) {
            if (line.find("\"mono_calibrations\"") != string::npos) {
                in_mono = true;
                continue;
            }
            
            if (!in_mono) continue;
            
            if (line.find("\"camera_id\":") != string::npos) {
                size_t start = line.find("\"", line.find(":") + 1) + 1;
                size_t end = line.find("\"", start);
                current.id = line.substr(start, end - start);
            }
            if (line.find("\"calibration_file\":") != string::npos) {
                size_t start = line.find("\"", line.find(":") + 1) + 1;
                size_t end = line.find("\"", start);
                current.file = line.substr(start, end - start);
            }
            if (line.find("\"calibration_time\":") != string::npos) {
                size_t start = line.find("\"", line.find(":") + 1) + 1;
                size_t end = line.find("\"", start);
                current.time = line.substr(start, end - start);
            }
            if (line.find("\"frames_used\":") != string::npos) {
                size_t pos = line.find(":") + 1;
                string val = line.substr(pos);
                val.erase(remove_if(val.begin(), val.end(), [](char c) { return !isdigit(c); }), val.end());
                if (!val.empty()) current.frames = stoi(val);
            }
            if (line.find("\"image_width\":") != string::npos) {
                size_t pos = line.find(":") + 1;
                string val = line.substr(pos);
                val.erase(remove_if(val.begin(), val.end(), [](char c) { return !isdigit(c); }), val.end());
                if (!val.empty()) current.width = stoi(val);
            }
            if (line.find("\"image_height\":") != string::npos) {
                size_t pos = line.find(":") + 1;
                string val = line.substr(pos);
                val.erase(remove_if(val.begin(), val.end(), [](char c) { return !isdigit(c); }), val.end());
                if (!val.empty()) current.height = stoi(val);
            }
            if (line.find("\"reprojection_error\":") != string::npos) {
                size_t pos = line.find(":") + 1;
                size_t comma = line.find_first_of(",}", pos);
                string val = line.substr(pos, comma - pos);
                val.erase(remove_if(val.begin(), val.end(), ::isspace), val.end());
                if (!val.empty()) current.error = stod(val);
                
                // Конец записи камеры
                if (!current.id.empty()) {
                    existing_calibs[current.id] = current;
                    cout << "  Found existing: ID=" << current.id << endl;
                    current = MonoCalib();
                }
            }
            
            if (line.find("\"stereo_calibrations\"") != string::npos) {
                break;
            }
        }
        existing_file.close();
    }
    
    // Добавить/обновить новые камеры
    for (const auto& cam_id : new_cam_ids) {
        string yml_path = output_dir + "/cam_" + cam_id + "_calibration.yml";
        FileStorage fs(yml_path, FileStorage::READ);
        
        if (!fs.isOpened()) continue;
        
        MonoCalib calib;
        calib.id = cam_id;
        calib.file = "cam_" + cam_id + "_calibration.yml";
        
        fs["frames_used"] >> calib.frames;
        fs["image_width"] >> calib.width;
        fs["image_height"] >> calib.height;
        fs["reprojection_error"] >> calib.error;
        fs["calibration_time"] >> calib.time;
        fs.release();
        
        existing_calibs[cam_id] = calib;  // Добавить или обновить
        cout << "  Added/Updated: ID=" << cam_id << endl;
    }
    
    // Записать обратно
    ofstream file(filepath);
    file << "{\n";
    file << "  \"mono_calibrations\": [\n";
    
    size_t idx = 0;
    for (const auto& pair : existing_calibs) {
        const auto& c = pair.second;
        file << "    {\n";
        file << "      \"calibration_file\": \"" << c.file << "\",\n";
        file << "      \"calibration_time\": \"" << c.time << "\",\n";
        file << "      \"camera_id\": \"" << c.id << "\",\n";
        file << "      \"frames_used\": " << c.frames << ",\n";
        file << "      \"image_height\": " << c.height << ",\n";
        file << "      \"image_width\": " << c.width << ",\n";
        file << "      \"mode\": \"\",\n";
        file << "      \"reprojection_error\": " << c.error << ",\n";
        file << "      \"success\": true\n";
        file << "    }";
        if (idx < existing_calibs.size() - 1) file << ",";
        file << "\n";
        idx++;
    }
    
    file << "  ],\n";
    file << "  \"stereo_calibrations\": []\n";
    file << "}\n";
    
    file.close();
    cout << "Saved: " << filepath << " (total " << existing_calibs.size() << " cameras)" << endl;
}

// Структура камеры из config.json
struct CameraConfig {
    string id;
    string device;
    int width;
    int height;
    int fps;
};

vector<CameraConfig> loadCamerasFromConfig(const string& config_path) {
    vector<CameraConfig> cameras;
    ifstream file(config_path);
    
    if (!file.is_open()) {
        cerr << "Failed to open " << config_path << endl;
        return cameras;
    }
    
    string content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    file.close();
    
    // Найти все "mode": "calibration" (с любыми пробелами)
    size_t pos = 0;
    while (true) {
        pos = content.find("calibration", pos);
        if (pos == string::npos) break;
        
        // Проверить что это mode: calibration (не просто слово в комментарии)
        size_t mode_pos = content.rfind("\"mode\"", pos);
        if (mode_pos == string::npos || pos - mode_pos > 50) {
            pos++;
            continue;
        }
        
        cout << "[DEBUG] Found calibration at pos " << pos << endl;
        
        // Найти начало объекта камеры (идём назад до открывающей {)
        int depth = 0;
        size_t obj_start = pos;
        for (int i = pos; i >= 0; i--) {
            if (content[i] == '}') depth++;
            if (content[i] == '{') {
                depth--;
                if (depth < 0) {
                    obj_start = i;
                    break;
                }
            }
        }
        
        // Найти конец объекта
        depth = 0;
        size_t obj_end = obj_start;
        for (size_t i = obj_start; i < content.size(); i++) {
            if (content[i] == '{') depth++;
            if (content[i] == '}') {
                depth--;
                if (depth == 0) {
                    obj_end = i;
                    break;
                }
            }
        }
        
        string obj = content.substr(obj_start, obj_end - obj_start);
        cout << "[DEBUG] Camera object length: " << obj.size() << endl;
        
        CameraConfig cam;
        
        // ID
        size_t id_pos = obj.find("\"id\"");
        if (id_pos != string::npos) {
            size_t start = obj.find("\"", id_pos + 5) + 1;
            size_t end = obj.find("\"", start);
            cam.id = obj.substr(start, end - start);
        }
        
        // Device
        size_t dev_pos = obj.find("\"device\"");
        if (dev_pos != string::npos) {
            size_t start = obj.find("\"", dev_pos + 9) + 1;
            size_t end = obj.find("\"", start);
            cam.device = obj.substr(start, end - start);
        }
        
        // W, H, FPS в preferred блоке
        size_t pref_pos = obj.find("\"preferred\"");
        if (pref_pos != string::npos) {
            size_t w_pos = obj.find("\"w\"", pref_pos);
            if (w_pos != string::npos) {
                size_t colon = obj.find(":", w_pos);
                size_t comma = obj.find_first_of(",}", colon);
                string val = obj.substr(colon + 1, comma - colon - 1);
                val.erase(remove_if(val.begin(), val.end(), ::isspace), val.end());
                if (!val.empty()) cam.width = stoi(val);
            }
            
            size_t h_pos = obj.find("\"h\"", pref_pos);
            if (h_pos != string::npos) {
                size_t colon = obj.find(":", h_pos);
                size_t comma = obj.find_first_of(",}", colon);
                string val = obj.substr(colon + 1, comma - colon - 1);
                val.erase(remove_if(val.begin(), val.end(), ::isspace), val.end());
                if (!val.empty()) cam.height = stoi(val);
            }
            
            size_t fps_pos = obj.find("\"fps\"", pref_pos);
            if (fps_pos != string::npos) {
                size_t colon = obj.find(":", fps_pos);
                size_t comma = obj.find_first_of(",}", colon);
                string val = obj.substr(colon + 1, comma - colon - 1);
                val.erase(remove_if(val.begin(), val.end(), ::isspace), val.end());
                if (!val.empty()) cam.fps = stoi(val);
            }
        }
        
        if (!cam.id.empty() && !cam.device.empty()) {
            cameras.push_back(cam);
            cout << "Found: ID=" << cam.id << " dev=" << cam.device 
                 << " " << cam.width << "x" << cam.height << endl;
        }
        
        pos = obj_end + 1;
    }
    
    return cameras;
}


// ============================================================================
// Конфигурация
// ============================================================================

struct CalibConfig {
    int pattern_cols = 15;
    int pattern_rows = 7;
    float square_size = 22.0f;
    int min_frames = 20;        // было 100
    int max_frames = 100;        // было 200
    
    // Ослабленные пороги для быстрой калибровки
    float max_center_diff_horizontal = 0.40f;  // было 0.30
    float max_center_diff_vertical = 0.40f;    // было 0.30
    float max_tilt_diff = 15.0f;               // было 10.0
    float min_coverage = 0.25f;
    float max_coverage = 0.75f;
    float min_distance_between_frames = 0.03f; // было 0.05 (меньше = больше похожих кадров)
};

enum class HintType {
    SEARCHING,
    TOO_CLOSE,
    TOO_FAR,
    TOO_FLAT,
    TOO_TILTED,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_UP,
    MOVE_DOWN,
    HOLD_STILL,
    CAPTURED,
    TOO_SIMILAR,
    COMPLETE,
    STEREO_NOT_BOTH,
    STEREO_TILT_DIFF,
    STEREO_CENTER_DIFF
};

string hintToString(HintType hint, float value = 0) {
    switch (hint) {
        case HintType::SEARCHING: return "Searching for chessboard...";
        case HintType::TOO_CLOSE: return "Move board AWAY";
        case HintType::TOO_FAR: return "Move board CLOSER";
        case HintType::TOO_FLAT: return "TILT board more";
        case HintType::TOO_TILTED: return "Board too tilted";
        case HintType::MOVE_LEFT: return "Move LEFT";
        case HintType::MOVE_RIGHT: return "Move RIGHT";
        case HintType::MOVE_UP: return "Move UP";
        case HintType::MOVE_DOWN: return "Move DOWN";
        case HintType::HOLD_STILL: return "HOLD STILL... " + to_string((int)value) + "s";
        case HintType::CAPTURED: return "CAPTURED!";
        case HintType::TOO_SIMILAR: return "Move to NEW position";
        case HintType::COMPLETE: return "Calibration COMPLETE!";
        case HintType::STEREO_NOT_BOTH: return "Board visible on ONE camera only";
        case HintType::STEREO_TILT_DIFF: return "Cameras see different tilt";
        case HintType::STEREO_CENTER_DIFF: return "Board in different positions";
        default: return "";
    }
}

// ============================================================================
// Стерео-калибратор
// ============================================================================

class StereoCalibrator {
public:
    StereoCalibrator(const string& dev1, const string& dev2, const CalibConfig& cfg)
        : device1(dev1), device2(dev2), config(cfg) {
        
        is_running = false;
        is_calibrating = false;
        hold_start_time = 0;
        hold_duration = 1.0;
            

        // Объектные точки
        for (int i = 0; i < config.pattern_rows; i++) {
            for (int j = 0; j < config.pattern_cols; j++) {
                objp.push_back(Point3f(j * config.square_size, i * config.square_size, 0));
            }
        }
    }


    
bool startCameras() {
    // Извлечь номер из /dev/videoX
    int cam1_index = -1, cam2_index = -1;
    
    if (device1.find("/dev/video") != string::npos) {
        cam1_index = stoi(device1.substr(device1.find("video") + 5));
    }
    if (device2.find("/dev/video") != string::npos) {
        cam2_index = stoi(device2.substr(device2.find("video") + 5));
    }
    
    if (cam1_index < 0 || cam2_index < 0) {
        cerr << "Invalid device paths: " << device1 << ", " << device2 << endl;
        return false;
    }
    
    cout << "Opening cameras: index " << cam1_index << " and " << cam2_index << endl;
    
    cap1.open(cam1_index, CAP_V4L2);
    cap2.open(cam2_index, CAP_V4L2);
    
    if (!cap1.isOpened() || !cap2.isOpened()) {
        cerr << "Failed to open cameras" << endl;
        return false;
    }

    cap1.open(device1, CAP_V4L2);
    cap2.open(device2, CAP_V4L2);
    
    if (!cap1.isOpened() || !cap2.isOpened()) {
        return false;
    }
    
    // ДОБАВИТЬ: Увеличить буфер
    cap1.set(CAP_PROP_BUFFERSIZE, 3);
    cap2.set(CAP_PROP_BUFFERSIZE, 3);
    
    cap1.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
    cap2.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
    
    // Попробуйте установить одинаковое разрешение для обеих
    cap1.set(CAP_PROP_FRAME_WIDTH, 800);
    cap1.set(CAP_PROP_FRAME_HEIGHT, 600);
    cap2.set(CAP_PROP_FRAME_WIDTH, 800);
    cap2.set(CAP_PROP_FRAME_HEIGHT, 600);
    
    // Если cam2 не поддерживает 800x600, попробуйте 640x480 для обеих
    if ((int)cap2.get(CAP_PROP_FRAME_WIDTH) != 800) {
        cout << "Cam2 doesn't support 800x600, using 640x480 for both" << endl;
        cap1.set(CAP_PROP_FRAME_WIDTH, 640);
        cap1.set(CAP_PROP_FRAME_HEIGHT, 480);
        cap2.set(CAP_PROP_FRAME_WIDTH, 640);
        cap2.set(CAP_PROP_FRAME_HEIGHT, 480);
    }
    
    cap1.set(CAP_PROP_FPS, 30);
    cap2.set(CAP_PROP_FPS, 30);
    
    cout << "Cam1: " << cap1.get(CAP_PROP_FRAME_WIDTH) << "x" << cap1.get(CAP_PROP_FRAME_HEIGHT) << endl;
    cout << "Cam2: " << cap2.get(CAP_PROP_FRAME_WIDTH) << "x" << cap2.get(CAP_PROP_FRAME_HEIGHT) << endl;
    

    actual_img_size = Size(
        (int)cap1.get(CAP_PROP_FRAME_WIDTH),
        (int)cap1.get(CAP_PROP_FRAME_HEIGHT)
    );

    this_thread::sleep_for(chrono::milliseconds(1000));
    
    // Очистить буфер
    Mat dummy;
    for (int i = 0; i < 10; i++) {
        cap1.grab();
        cap2.grab();
    }
    
    is_running = true;
    return true;
}

    void stopCameras() {
        is_running = false;
        cap1.release();
        cap2.release();
    }
    
    void startCalibration() {
        lock_guard<mutex> lock(data_mutex);
        is_calibrating = true;
        object_points.clear();
        image_points_1.clear();
        image_points_2.clear();
        hold_start_time = 0;
        cout << "Calibration started" << endl;
    }

bool getFrame(Mat& combined, string& hint_text, int& progress_current, int& progress_max) {
    if (!is_running) return false;
    
    Mat frame1, frame2;
    bool ok1 = false, ok2 = false;
    
    // Попытки захвата с retry
    for (int attempt = 0; attempt < 3; attempt++) {
        ok1 = cap1.read(frame1);
        ok2 = cap2.read(frame2);
        
        if (ok1 && ok2 && !frame1.empty() && !frame2.empty()) {
            break;
        }
        
        if (attempt < 2) {
            this_thread::sleep_for(chrono::milliseconds(10));
        }
    }
    
    if (!ok1 || !ok2 || frame1.empty() || frame2.empty()) {
        // Не логировать каждый кадр, только каждую секунду
        static time_t last_error = 0;
        if (time(nullptr) - last_error > 1) {
            cerr << "Failed to capture frames (ok1=" << ok1 << " ok2=" << ok2 << ")" << endl;
            last_error = time(nullptr);
        }
        
        frame1 = Mat::zeros(600, 800, CV_8UC3);
        frame2 = Mat::zeros(600, 800, CV_8UC3);
        putText(frame1, "Cam1: Waiting...", Point(250, 300), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,255), 2);
        putText(frame2, "Cam2: Waiting...", Point(250, 300), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,255), 2);
        hconcat(frame1, frame2, combined);
        hint_text = "Waiting for cameras...";
        return true;
    }    
    // ИСПРАВЛЕНИЕ: Приведение к одному размеру И типу
    if (frame1.size() != frame2.size()) {
        resize(frame2, frame2, frame1.size());
    }
    
    // КРИТИЧНО: Проверка типа данных
    if (frame1.type() != frame2.type()) {
        if (frame1.channels() != frame2.channels()) {
            if (frame1.channels() == 1) cvtColor(frame1, frame1, COLOR_GRAY2BGR);
            if (frame2.channels() == 1) cvtColor(frame2, frame2, COLOR_GRAY2BGR);
        }
    }
    
    // Теперь безопасно делать hconcat
    HintType hint = HintType::SEARCHING;
    float hint_value = 0;
    
    if (is_calibrating) {
        vector<Point2f> corners1, corners2;
        bool found1 = findChessboardCorners(frame1, Size(config.pattern_cols, config.pattern_rows), corners1,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
        bool found2 = findChessboardCorners(frame2, Size(config.pattern_cols, config.pattern_rows), corners2,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);

    // ДОБАВИТЬ:
    static int frame_counter = 0;
    frame_counter++;
    if (frame_counter % 30 == 0) { // Каждую секунду
        cout << "[DEBUG] Frame #" << frame_counter 
             << " found1=" << found1 
             << " found2=" << found2 << endl;
    }

        if (found1 && found2) {
        cout << "[DEBUG] Both cameras see chessboard!" << endl;

            Mat gray1, gray2;
            cvtColor(frame1, gray1, COLOR_BGR2GRAY);
            cvtColor(frame2, gray2, COLOR_BGR2GRAY);
            
            cornerSubPix(gray1, corners1, Size(11,11), Size(-1,-1), 
                TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001));
            cornerSubPix(gray2, corners2, Size(11,11), Size(-1,-1),
                TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001));
            
            drawChessboardCorners(frame1, Size(config.pattern_cols, config.pattern_rows), corners1, true);
            drawChessboardCorners(frame2, Size(config.pattern_cols, config.pattern_rows), corners2, true);

            hint = checkStereoQuality(corners1, corners2, frame1.size(), hint_value);

        cout << "[DEBUG] Quality check result: " << (int)hint << " hint_value=" << hint_value << endl;

            if (hint == HintType::CAPTURED) {
                lock_guard<mutex> lock(data_mutex);
                object_points.push_back(objp);
                image_points_1.push_back(corners1);
                image_points_2.push_back(corners2);
                hold_start_time = 0;
                
                cout << "Frame captured: " << image_points_1.size() << "/" << config.max_frames << endl;
                
                if (image_points_1.size() >= config.max_frames) {
                    hint = HintType::COMPLETE;
                    is_calibrating = false;
                }
            }
        } else if (found1 || found2) {
            hint = HintType::STEREO_NOT_BOTH;
        }
    }
    
    // Комбинирование
    try {
        hconcat(frame1, frame2, combined);
    } catch (const cv::Exception& e) {
        cerr << "hconcat failed: " << e.what() << endl;
        cerr << "frame1: " << frame1.size() << " type=" << frame1.type() << " channels=" << frame1.channels() << endl;
        cerr << "frame2: " << frame2.size() << " type=" << frame2.type() << " channels=" << frame2.channels() << endl;
        return false;
    }
    
    // Разделитель
    line(combined, Point(frame1.cols, 0), Point(frame1.cols, combined.rows), Scalar(255,255,255), 2);
    
    // Подсказка
    hint_text = hintToString(hint, hint_value);
    Scalar color = (hint == HintType::CAPTURED) ? Scalar(0,255,0) : Scalar(0,165,255);
    putText(combined, hint_text, Point(10, 40), FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
    
    // Прогресс
    if (is_calibrating) {
        lock_guard<mutex> lock(data_mutex);
        progress_current = image_points_1.size();
        progress_max = config.max_frames;
        string progress_str = to_string(progress_current) + "/" + to_string(progress_max);
        putText(combined, progress_str, Point(10, 80), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255,255,0), 2);
    }
    
    return true;
}
    bool calibrate(string& error_msg, const string& cam1_id, const string& cam2_id) {
    lock_guard<mutex> lock(data_mutex);
    
    if (image_points_1.size() < config.min_frames) {
        error_msg = "Not enough frames: " + to_string(image_points_1.size());
        return false;
    }
    
//    Size img_size(640, 480); // TODO: использовать реальное разрешение
    // Использовать реальный размер вместо захардкоженного
    Size img_size = actual_img_size;
    cout << "Using image size: " << img_size.width << "x" << img_size.height << endl;

    Mat K1, D1, K2, D2, R, T, E, F;
    
    double rms = stereoCalibrate(
        object_points, image_points_1, image_points_2,
        K1, D1, K2, D2, img_size, R, T, E, F, 0,
        TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 100, 1e-5)
    );
    
    cout << "Stereo calibration RMS error: " << rms << endl;
    
    // Вычислить per-view ошибки
//    vector<double> errors1, errors2;
//    for (size_t i = 0; i < object_points.size(); i++) {
//        vector<Point2f> proj1, proj2;
//        projectPoints(object_points[i], Mat::zeros(3,1,CV_64F), Mat::zeros(3,1,CV_64F), K1, D1, proj1);
//        projectPoints(object_points[i], Mat::zeros(3,1,CV_64F), Mat::zeros(3,1,CV_64F), K2, D2, proj2);
        
//        double err1 = norm(image_points_1[i], proj1, NORM_L2) / proj1.size();
//        double err2 = norm(image_points_2[i], proj2, NORM_L2) / proj2.size();
//        errors1.push_back(err1);
//        errors2.push_back(err2);
//    }
      vector<double> errors1, errors2;
      for (size_t i = 0; i < object_points.size(); i++) {
          // Найти позу для каждого кадра
          Mat rvec1, tvec1, rvec2, tvec2;
          solvePnP(object_points[i], image_points_1[i], K1, D1, rvec1, tvec1);
          solvePnP(object_points[i], image_points_2[i], K2, D2, rvec2, tvec2);

          // Спроецировать обратно
          vector<Point2f> proj1, proj2;
          projectPoints(object_points[i], rvec1, tvec1, K1, D1, proj1);
          projectPoints(object_points[i], rvec2, tvec2, K2, D2, proj2);

          // Вычислить среднюю ошибку для кадра
          double err1 = 0, err2 = 0;
          for (size_t j = 0; j < proj1.size(); j++) {
          err1 += norm(image_points_1[i][j] - proj1[j]);
          err2 += norm(image_points_2[i][j] - proj2[j]);
          }
          errors1.push_back(err1 / proj1.size());
          errors2.push_back(err2 / proj2.size());
          }


    double rms1 = accumulate(errors1.begin(), errors1.end(), 0.0) / errors1.size();
    double rms2 = accumulate(errors2.begin(), errors2.end(), 0.0) / errors2.size();
    
    // Создать папку results
    system("mkdir -p results");
    
    // Сохранить моно-калибровки
    saveMonoCalibrationYML(cam1_id, K1, D1, image_points_1.size(), img_size, rms1, errors1, "results");
    saveMonoCalibrationYML(cam2_id, K2, D2, image_points_2.size(), img_size, rms2, errors2, "results");
    
    // Сохранить стерео
    saveStereoCalibrationYML(cam1_id, cam2_id, R, T, E, F, rms, image_points_1.size(), "results");
    
    // Сохранить JSON индекс
    vector<string> cam_ids = {cam1_id, cam2_id};
    saveCalibrationResultsJSON(cam_ids, "results");
    
    error_msg = "Stereo RMS: " + to_string(rms);
    return true;
}
    
    atomic<bool> is_running;
    atomic<bool> is_calibrating;
    Size actual_img_size;  // ← ДОБАВИТЬ

    // ДОБАВИТЬ: Сделать публичными для доступа из main()
    mutex data_mutex;
    vector<vector<Point2f>> image_points_1;
    vector<vector<Point2f>> image_points_2;

private:
    HintType checkStereoQuality(const vector<Point2f>& c1, const vector<Point2f>& c2, 
                                 Size img_size, float& hint_value) {

    // ДОБАВИТЬ В НАЧАЛО:
    cout << "[QUALITY] Checking stereo quality..." << endl;

        // Центры
        Point2f center1 = accumulate(c1.begin(), c1.end(), Point2f(0,0)) / (float)c1.size();
        Point2f center2 = accumulate(c2.begin(), c2.end(), Point2f(0,0)) / (float)c2.size();
        
        Point2f center1_norm(center1.x / img_size.width, center1.y / img_size.height);
        Point2f center2_norm(center2.x / img_size.width, center2.y / img_size.height);
        
        float diff_x = abs(center1_norm.x - center2_norm.x);
        float diff_y = abs(center1_norm.y - center2_norm.y);

    cout << "[QUALITY] Center diff: x=" << diff_x << " (max " << config.max_center_diff_horizontal 
         << ") y=" << diff_y << " (max " << config.max_center_diff_vertical << ")" << endl;
        
        if (diff_x > config.max_center_diff_horizontal) {
        cout << "[QUALITY] REJECTED: horizontal center diff too large" << endl;
            hold_start_time = 0;
            return HintType::STEREO_CENTER_DIFF;
        }
        if (diff_y > config.max_center_diff_vertical) {
            hold_start_time = 0;
            return HintType::STEREO_CENTER_DIFF;
        }
        
        // Наклон
        float tilt1 = atan2(c1[config.pattern_cols-1].y - c1[0].y, 
                           c1[config.pattern_cols-1].x - c1[0].x) * 180.0 / CV_PI;
        float tilt2 = atan2(c2[config.pattern_cols-1].y - c2[0].y,
                           c2[config.pattern_cols-1].x - c2[0].x) * 180.0 / CV_PI;
        
        if (abs(tilt1 - tilt2) > config.max_tilt_diff) {
            hold_start_time = 0;
            return HintType::STEREO_TILT_DIFF;
        }
        
        // Похожесть с предыдущим
        if (!image_points_1.empty()) {
            float diff = 0;
            for (size_t i = 0; i < c1.size(); i++) {
                diff += norm(c1[i] - image_points_1.back()[i]);
            }
            diff /= c1.size();
            float normalized_diff = diff / sqrt(img_size.width*img_size.width + img_size.height*img_size.height);
            
            if (normalized_diff < config.min_distance_between_frames) {
                hold_start_time = 0;
                return HintType::TOO_SIMILAR;
            }
        }
        
        // Удержание
        if (hold_start_time == 0) {
            hold_start_time = time(nullptr);
        }
        
        time_t elapsed = time(nullptr) - hold_start_time;
        if (elapsed < hold_duration) {
            hint_value = hold_duration - elapsed;
            return HintType::HOLD_STILL;
        }

    cout << "Quality check: hint=" << (int)HintType::CAPTURED
         << " diff_x=" << diff_x 
         << " diff_y=" << diff_y 
         << " tilt_diff=" << abs(tilt1-tilt2) << endl;

        return HintType::CAPTURED;
    }
    
    string device1, device2;
    CalibConfig config;
    VideoCapture cap1, cap2;
    
    vector<vector<Point3f>> object_points;
    vector<Point3f> objp;
    
    time_t hold_start_time;
    double hold_duration;
    
};

// ============================================================================
// HTTP Server
// ============================================================================

const char* HTML_PAGE = 
"<!DOCTYPE html>\n"
"<html>\n"
"<head>\n"
"    <title>Stereo Calibration</title>\n"
"    <meta charset=\"utf-8\">\n"
"    <style>\n"
"        body { font-family: Arial; background: #0d1117; color: #c9d1d9; padding: 20px; }\n"
"        .container { max-width: 1600px; margin: 0 auto; }\n"
"        button { padding: 12px 24px; margin: 5px; border: none; border-radius: 6px; cursor: pointer; font-size: 15px; }\n"
"        .btn-primary { background: #238636; color: white; }\n"
"        .btn-danger { background: #da3633; color: white; }\n"
"        select, input { padding: 10px; margin: 5px; border-radius: 6px; background: #21262d; color: #c9d1d9; border: 1px solid #30363d; width: 200px; }\n"
"        label { display: inline-block; width: 250px; margin: 5px 0; }\n"
"        #video { width: 100%; border-radius: 6px; margin: 20px 0; }\n"
"        .status { background: #21262d; padding: 15px; border-radius: 6px; margin: 10px 0; }\n"
"        .params { background: #161b22; padding: 15px; border-radius: 6px; margin: 10px 0; }\n"
"    </style>\n"
"</head>\n"
"<body>\n"
"<script>\n"
"    let cameras = [];\n"
"    \n"
"    fetch('/cameras').then(r => r.json()).then(data => {\n"
"        cameras = data;\n"
"        const sel1 = document.getElementById('cam1');\n"
"        const sel2 = document.getElementById('cam2');\n"
"        cameras.forEach((cam, idx) => {\n"
"            sel1.add(new Option(`ID=${cam.id} ${cam.device} ${cam.width}x${cam.height}`, idx));\n"
"            sel2.add(new Option(`ID=${cam.id} ${cam.device} ${cam.width}x${cam.height}`, idx));\n"
"        });\n"
"        if (cameras.length > 1) sel2.selectedIndex = 1;\n"
"    });\n"
"    \n"
"    function startCameras() {\n"
"        const cam1 = document.getElementById('cam1').value;\n"
"        const cam2 = document.getElementById('cam2').value;\n"
"        const cols = document.getElementById('board_cols').value;\n"
"        const rows = document.getElementById('board_rows').value;\n"
"        const square = document.getElementById('square_size').value;\n"
"        const frames = document.getElementById('max_frames').value;\n"
"        fetch(`/start?cam1=${cam1}&cam2=${cam2}&cols=${cols}&rows=${rows}&square=${square}&frames=${frames}`)\n"
"            .then(r => r.text()).then(txt => alert(txt));\n"
"    }\n"
"    \n"
"    function compute() {\n"
"        console.log('Starting calibration computation...');\n"
"        fetch('/compute')\n"
"            .then(r => r.text())\n"
"            .then(txt => {\n"
"                console.log('Result:', txt);\n"
"                alert(txt);\n"
"            })\n"
"            .catch(err => {\n"
"                console.error('Error:', err);\n"
"                alert('Error: ' + err);\n"
"            });\n"
"    }\n"
"    \n"
"    setInterval(() => {\n"
"        fetch('/status').then(r => r.json()).then(data => {\n"
"            document.getElementById('status').textContent = \n"
"                `Status: ${data.hint} | Frames: ${data.current}/${data.max}`;\n"
"        });\n"
"    }, 1000);\n"
"</script>\n"
"    <div class=\"container\">\n"
"        <h1>Stereo Camera Calibration</h1>\n"
"        \n"
"        <div class=\"params\">\n"
"            <h3>Camera Selection</h3>\n"
"            <label>Camera 1:</label><select id=\"cam1\"></select><br>\n"
"            <label>Camera 2:</label><select id=\"cam2\"></select><br>\n"
"            \n"
"            <h3>Calibration Board Parameters</h3>\n"
"            <label>Columns (inner corners):</label><input type=\"number\" id=\"board_cols\" value=\"15\" min=\"3\" max=\"30\"><br>\n"
"            <label>Rows (inner corners):</label><input type=\"number\" id=\"board_rows\" value=\"7\" min=\"3\" max=\"30\"><br>\n"
"            <label>Square size (mm):</label><input type=\"number\" id=\"square_size\" value=\"22\" min=\"5\" max=\"200\" step=\"0.1\"><br>\n"
"            \n"
"            <h3>Capture Settings</h3>\n"
"            <label>Target frames:</label><input type=\"number\" id=\"max_frames\" value=\"50\" min=\"20\" max=\"300\"><br>\n"
"        </div>\n"
"        \n"
"        <div>\n"
"            <button class=\"btn-primary\" onclick=\"startCameras()\">Start Cameras</button>\n"
"            <button class=\"btn-danger\" onclick=\"fetch('/stop')\">Stop</button>\n"
"            <button class=\"btn-primary\" onclick=\"fetch('/calibrate')\">Start Calibration</button>\n"
"            <button class=\"btn-primary\" onclick=\"compute()\">Compute</button>\n"
"        </div>\n"
"        \n"
"        <img id=\"video\" src=\"/video\">\n"
"        <div class=\"status\" id=\"status\">Status: Ready</div>\n"
"    </div>\n"
"</body>\n"
"</html>\n";


int main(int argc, char** argv) {
    // Путь к config.json
    string config_path = "config.json";
    if (argc > 1) {
        config_path = argv[1];
    }

    cout << "Loading config from: " << config_path << endl;
    // Загрузить камеры из конфига
    vector<CameraConfig> available_cameras = loadCamerasFromConfig(config_path);
    cout << "Loaded " << available_cameras.size() << " cameras" << endl;

    if (available_cameras.empty()) {
        cerr << "No cameras with mode='calibration' found in " << config_path << endl;
        return 1;
    }
    
    cout << "Found " << available_cameras.size() << " calibration cameras:" << endl;
    for (const auto& cam : available_cameras) {
        cout << "  ID=" << cam.id << " device=" << cam.device 
             << " " << cam.width << "x" << cam.height << "@" << cam.fps << "fps" << endl;
    }
    
    // По умолчанию первые две камеры
    if (available_cameras.size() < 2) {
        cerr << "Need at least 2 calibration cameras for stereo" << endl;
        return 1;
    }
    
    CalibConfig config;
    
    // Глобальные переменные для выбора камер
    int selected_cam1_idx = 0;
    int selected_cam2_idx = 1;
    StereoCalibrator* calibrator = nullptr;
    
    httplib::Server svr;
    
    // HTML page
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(HTML_PAGE, "text/html");
    });
    
    // Получить список камер
    svr.Get("/cameras", [&](const httplib::Request&, httplib::Response& res) {
        ostringstream json;
        json << "[";
        for (size_t i = 0; i < available_cameras.size(); i++) {
            const auto& cam = available_cameras[i];
            json << "{\"id\":\"" << cam.id << "\","
                 << "\"device\":\"" << cam.device << "\","
                 << "\"width\":" << cam.width << ","
                 << "\"height\":" << cam.height << ","
                 << "\"fps\":" << cam.fps << "}";
            if (i < available_cameras.size() - 1) json << ",";
        }
        json << "]";
        res.set_content(json.str(), "application/json");
    });
    
    // Start cameras
    svr.Get("/start", [&](const httplib::Request& req, httplib::Response& res) {
    if (req.has_param("cam1")) selected_cam1_idx = stoi(req.get_param_value("cam1"));
    if (req.has_param("cam2")) selected_cam2_idx = stoi(req.get_param_value("cam2"));
    
    // Получить параметры доски
    if (req.has_param("cols")) config.pattern_cols = stoi(req.get_param_value("cols"));
    if (req.has_param("rows")) config.pattern_rows = stoi(req.get_param_value("rows"));
    if (req.has_param("square")) config.square_size = stof(req.get_param_value("square"));
    if (req.has_param("frames")) config.max_frames = stoi(req.get_param_value("frames"));
    
    cout << "Board: " << config.pattern_cols << "x" << config.pattern_rows 
         << " square=" << config.square_size << "mm frames=" << config.max_frames << endl;
    
    if (selected_cam1_idx >= available_cameras.size() || 
        selected_cam2_idx >= available_cameras.size() ||
        selected_cam1_idx == selected_cam2_idx) {
        res.set_content("INVALID_SELECTION", "text/plain");
        return;
    }

    // ИЗМЕНИТЬ: Только если калибратора нет или камеры другие
    bool need_recreate = !calibrator;
    if (calibrator) {
        // Проверить что камеры те же
        const auto& cam1 = available_cameras[selected_cam1_idx];
        const auto& cam2 = available_cameras[selected_cam2_idx];
        // Добавьте проверку если хотите запомнить какие камеры открыты
        // Пока просто предупредим
        cout << "WARNING: Calibrator already exists, cameras already started" << endl;
        res.set_content("ALREADY_STARTED", "text/plain");
        return;
    }

    const auto& cam1 = available_cameras[selected_cam1_idx];
    const auto& cam2 = available_cameras[selected_cam2_idx];
    
    calibrator = new StereoCalibrator(cam1.device, cam2.device, config);
    
    bool ok = calibrator->startCameras();
    res.set_content(ok ? "OK" : "FAILED", "text/plain");
    });

    // Stop
    svr.Get("/stop", [&](const httplib::Request&, httplib::Response& res) {
        if (calibrator) calibrator->stopCameras();
        res.set_content("OK", "text/plain");
    });
    
    // Start calibration
    svr.Get("/calibrate", [&](const httplib::Request&, httplib::Response& res) {
        if (calibrator) {
            calibrator->startCalibration();
            res.set_content("OK", "text/plain");
        } else {
            res.set_content("NO_CAMERAS", "text/plain");
        }
    });
    
    svr.Get("/compute", [&](const httplib::Request&, httplib::Response& res) {
    if (!calibrator) {
        res.set_content("NO_CAMERAS", "text/plain");
        return;
    }
    
    string error_msg;
    const auto& cam1 = available_cameras[selected_cam1_idx];
    const auto& cam2 = available_cameras[selected_cam2_idx];
    
    bool ok = calibrator->calibrate(error_msg, cam1.id, cam2.id);
    res.set_content(ok ? "SUCCESS: " + error_msg : "FAILED: " + error_msg, "text/plain");
    });
    
    // Status
    svr.Get("/status", [&](const httplib::Request&, httplib::Response& res) {
        if (!calibrator) {
            res.set_content("{\"current\":0,\"max\":0,\"hint\":\"No cameras\"}", "application/json");
            return;
        }
        
        int current = 0, max_frames = config.max_frames;
        {
            lock_guard<mutex> lock(calibrator->data_mutex);
            current = calibrator->image_points_1.size();
        }
        
        string json = "{\"current\":" + to_string(current) + 
                      ",\"max\":" + to_string(max_frames) + 
                      ",\"hint\":\"Calibrating\"}";
        res.set_content(json, "application/json");
    });
    
    // Video stream
    svr.Get("/video", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content_provider(
            "multipart/x-mixed-replace; boundary=frame",
            [&](size_t, httplib::DataSink& sink) {
                while (calibrator && calibrator->is_running) {
                    Mat combined;
                    string hint;
                    int progress_cur, progress_max;
                    
                    if (!calibrator->getFrame(combined, hint, progress_cur, progress_max)) {
                        break;
                    }
                    
                    vector<uchar> buf;
                    imencode(".jpg", combined, buf);
                    
                    ostringstream oss;
                    oss << "--frame\r\n";
                    oss << "Content-Type: image/jpeg\r\n";
                    oss << "Content-Length: " << buf.size() << "\r\n\r\n";
                    
                    string header = oss.str();
                    sink.write(header.c_str(), header.size());
                    sink.write((const char*)buf.data(), buf.size());
                    sink.write("\r\n", 2);
                    
                    this_thread::sleep_for(chrono::milliseconds(33));
                }
                return true;
            }
        );
    });
    
    cout << "Server starting on http://0.0.0.0:5000" << endl;
    svr.listen("0.0.0.0", 5000);
    
    if (calibrator) delete calibrator;
    return 0;
}