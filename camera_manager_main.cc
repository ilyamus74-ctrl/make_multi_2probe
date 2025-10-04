#include "camera_manager.h"
#include <csignal>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <atomic>
#include <sys/stat.h>
#include <thread>
#include <chrono>
#include <stdexcept>
#include <cstdio>
#include <unordered_map>
#include <map>
#include <mutex>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <ctime>
#include "calibration_watcher.h"
#include <unordered_map>
#include <future>  // ← ДОБАВЬТЕ ЭТУ СТРОКУ


#include "httplib.h"
#include "nlohmann/json.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include "calibration/session.h"
#include "camera_roles.h"
#include "global_tracker.h"
#include "camera_scheme.h"



static CameraManager g_mgr;
static CameraSchemeManager g_scheme_manager;
static GlobalTracker g_global_tracker(&g_scheme_manager, &g_mgr);
static bool g_use_global_tracking = false;


static constexpr const char *kManagerDebugLogPath =
    "/tmp/camera_manager_debug.log";
static std::ofstream g_manager_debug;
static std::mutex g_manager_debug_mutex;
static std::atomic<bool> g_manager_debug_enabled{false};
static std::unordered_map<std::string, std::chrono::steady_clock::time_point>
    g_last_poll_time;

static std::string formatTimestamp(
    std::chrono::system_clock::time_point time_point) {
  auto time = std::chrono::system_clock::to_time_t(time_point);
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &time);
#else
  localtime_r(&time, &tm);
#endif
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
               time_point.time_since_epoch()) %
            1000;
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << '.'
      << std::setw(3) << std::setfill('0') << ms.count();
  oss << std::setfill(' ');
  return oss.str();
}

static void setManagerDebugEnabled(bool enabled) {
  bool current = g_manager_debug_enabled.load(std::memory_order_acquire);
  if (current == enabled && (!enabled || g_manager_debug.is_open()))
    return;

  std::lock_guard<std::mutex> lk(g_manager_debug_mutex);

  current = g_manager_debug_enabled.load(std::memory_order_relaxed);
  if (current == enabled && (!enabled || g_manager_debug.is_open()))
    return;

  if (enabled) {
    if (!g_manager_debug.is_open()) {
      g_manager_debug.open(kManagerDebugLogPath, std::ios::app);
    }
    if (!g_manager_debug.is_open()) {
      std::cerr << "Failed to open camera manager debug log at "
                << kManagerDebugLogPath << std::endl;
      g_manager_debug_enabled.store(false, std::memory_order_release);
      return;
    }

    auto now = std::chrono::system_clock::now();
    auto stamp = formatTimestamp(now);
    g_manager_debug << "=== Manager debug logging enabled at " << stamp
                    << " ===" << std::endl;
    g_manager_debug.flush();
    g_manager_debug_enabled.store(true, std::memory_order_release);
  } else {
    if (g_manager_debug.is_open()) {
      g_manager_debug << "=== Manager debug logging disabled ===" << std::endl;
      g_manager_debug.flush();
      g_manager_debug.close();
    }
    g_last_poll_time.clear();
    g_manager_debug_enabled.store(false, std::memory_order_release);
  }
}

static void writeCameraLog(const std::string &cam_id,
                           const std::string &body,
                           std::chrono::system_clock::time_point wall_time,
                           std::chrono::steady_clock::time_point steady_time) {
  if (!g_manager_debug_enabled.load(std::memory_order_acquire))
    return;

  std::lock_guard<std::mutex> lk(g_manager_debug_mutex);
  if (!g_manager_debug_enabled.load(std::memory_order_relaxed) ||
      !g_manager_debug.is_open())
    return;

  double delta_ms = -1.0;
  auto it = g_last_poll_time.find(cam_id);
  if (it != g_last_poll_time.end()) {
    delta_ms = std::chrono::duration<double, std::milli>(steady_time - it->second)
                   .count();
  }
  g_last_poll_time[cam_id] = steady_time;

  g_manager_debug << formatTimestamp(wall_time);
  if (delta_ms >= 0.0) {
    std::ostringstream delta_ss;
    delta_ss << std::fixed << std::setprecision(1) << delta_ms;
    g_manager_debug << " (+" << delta_ss.str() << " ms)";
  }
  g_manager_debug << " [" << cam_id << "]\n";
  g_manager_debug << body;
  if (!body.empty() && body.back() != '\n')
    g_manager_debug << '\n';
  g_manager_debug.flush();
}


static std::unique_ptr<CalibrationWatcher> g_calib_watcher;
static MultiCamera::CameraRoleManager g_role_mgr;
static httplib::Server g_server;
static bool g_preview_enabled = true;
static bool g_use_grayscale_tracking = false;
static std::unique_ptr<CalibrationSession> g_calib;
static std::filesystem::path g_config_path;
static std::filesystem::path g_exe_dir;
static nlohmann::json readMainConfig();

static CalibrationWatcher* ensureCalibrationWatcher() {
  if (!g_calib_watcher) {
    std::filesystem::create_directories("/tmp/rec");
    std::filesystem::create_directories("/tmp/calibration");
    g_calib_watcher = std::make_unique<CalibrationWatcher>("/tmp/rec", "/tmp/calibration");
    g_calib_watcher->setStatusCallback([](const std::string &msg, float progress) {
      printf("Calibration Status: %s (%.1f%%)\n", msg.c_str(), progress);
    });
    g_calib_watcher->setLogCallback([](const std::string &msg) {
      printf("Calibration Log: %s\n", msg.c_str());
    });
    auto manager_state = std::make_shared<std::atomic<bool>>(false);
    g_calib_watcher->setLifecycleCallbacks(
        [manager_state]() {
          bool was_running = g_mgr.isRunning();
          manager_state->store(was_running);
          g_mgr.stop();
        },
        [manager_state]() {
          if (manager_state->load()) {
            g_mgr.start();
          }
        });
  }
  g_global_tracker.setCalibrationWatcher(g_calib_watcher.get());
  return g_calib_watcher.get();
}

static nlohmann::json serializeGlobalObjects(const std::vector<GlobalObject>& objects) {
  nlohmann::json result = nlohmann::json::array();
  for (const auto& obj : objects) {
    nlohmann::json item;
    item["id"] = obj.global_id;
    item["confidence"] = obj.confidence;
    item["primary_camera"] = obj.primary_camera_id;
    item["world_position"] = {obj.world_position.x, obj.world_position.y, obj.world_position.z};


    const double world_dx = static_cast<double>(obj.world_position.x);
    const double world_dy = static_cast<double>(obj.world_position.y);
    const double world_dz = static_cast<double>(obj.world_position.z);
    const double distance_world = std::sqrt(world_dx * world_dx + world_dy * world_dy + world_dz * world_dz);
    item["distance_world_m"] = distance_world;

    double distance_primary = distance_world;
    bool has_primary_distance = false;
    if (!obj.primary_camera_id.empty()) {
      if (auto camera_position = g_global_tracker.getCameraWorldPosition(obj.primary_camera_id)) {
        const double dx = static_cast<double>(obj.world_position.x) - static_cast<double>(camera_position->x);
        const double dy = static_cast<double>(obj.world_position.y) - static_cast<double>(camera_position->y);
        const double dz = static_cast<double>(obj.world_position.z) - static_cast<double>(camera_position->z);
        distance_primary = std::sqrt(dx * dx + dy * dy + dz * dz);
        has_primary_distance = true;
        item["primary_camera_position"] = {camera_position->x, camera_position->y, camera_position->z};
      }
    }
    item["distance_mono_m"] = distance_primary;
    item["distance_source"] = has_primary_distance ? "primary_camera" : "world_origin";

    nlohmann::json cams = nlohmann::json::array();
    for (const auto& [cam_id, detection] : obj.camera_detections) {
      const auto& bbox = detection.box;
      nlohmann::json cam_entry = {
          {"camera", cam_id},
          {"bbox", {bbox.x, bbox.y, bbox.width, bbox.height}},
          {"track_id", detection.track_id}
      };


      if (auto camera_position = g_global_tracker.getCameraWorldPosition(cam_id)) {
        const double dx = static_cast<double>(obj.world_position.x) - static_cast<double>(camera_position->x);
        const double dy = static_cast<double>(obj.world_position.y) - static_cast<double>(camera_position->y);
        const double dz = static_cast<double>(obj.world_position.z) - static_cast<double>(camera_position->z);
        const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        cam_entry["distance_m"] = distance;
        cam_entry["camera_position"] = {camera_position->x, camera_position->y, camera_position->z};
      }

      cams.push_back(cam_entry);
    }
    item["cameras"] = cams;
    result.push_back(item);
  }
  return result;
}

static void fetchAndUpdateDetections(std::string cam_id, int det_port) {
  try {
    httplib::Client client("localhost", det_port);
    client.set_connection_timeout(0, 500000);
    client.set_read_timeout(0, 500000);

    auto result = client.Get("/api/last.json");
    if (result && result->status == 200) {
      auto detections_json = nlohmann::json::parse(result->body);
      std::vector<LocalDetection> detections;
      if (detections_json.contains("detections")) {
        for (const auto &det : detections_json["detections"]) {
          const auto &box = det["box"];
          auto jsonToInt = [](const nlohmann::json &value) {
            if (value.is_number_integer()) {
              return value.get<int>();
            }
            if (value.is_number_float()) {
              return static_cast<int>(std::lround(value.get<double>()));
            }
            return 0;
          };

          auto jsonToFloat = [](const nlohmann::json &value) {
            if (value.is_number_float()) {
              return static_cast<float>(value.get<double>());
            }
            if (value.is_number_integer()) {
              return static_cast<float>(value.get<int>());
            }
            return 0.0f;
          };

          int left = 0;
          int top = 0;
          int width = 0;
          int height = 0;
          int track_id = -1;

          if (box.is_array() && box.size() >= 4) {
            left = jsonToInt(box[0]);
            top = jsonToInt(box[1]);
            int right = jsonToInt(box[2]);
            int bottom = jsonToInt(box[3]);
            width = std::max(0, right - left);
            height = std::max(0, bottom - top);
          } else if (box.is_object()) {
            if (box.contains("left")) {
              left = jsonToInt(box["left"]);
            }
            if (box.contains("top")) {
              top = jsonToInt(box["top"]);
            }
            if (box.contains("width") && box.contains("height")) {
              width = jsonToInt(box["width"]);
              height = jsonToInt(box["height"]);
            } else if (box.contains("right") && box.contains("bottom")) {
              int right = jsonToInt(box["right"]);
              int bottom = jsonToInt(box["bottom"]);
              width = std::max(0, right - left);
              height = std::max(0, bottom - top);
            }
          }

          if (det.contains("track_id")) {
            track_id = jsonToInt(det["track_id"]);
          }

          DetectionDescriptor descriptor;
          if (det.contains("descriptor") && det["descriptor"].is_object()) {
            const auto &desc = det["descriptor"];
            if (desc.contains("rgb") && desc["rgb"].is_array() && desc["rgb"].size() >= 3) {
              descriptor.has_color = true;
              for (size_t i = 0; i < 3; ++i) {
                descriptor.color[i] = jsonToFloat(desc["rgb"][i]);
              }
            }
            if (desc.contains("intensity")) {
              descriptor.has_grayscale = true;
              descriptor.grayscale_intensity = jsonToFloat(desc["intensity"]);
            }
            if (desc.contains("texture") && desc["texture"].is_array()) {
              auto &tex = desc["texture"];
              size_t limit = std::min<size_t>(4, tex.size());
              if (limit > 0) {
                descriptor.has_grayscale = true;
                for (size_t i = 0; i < limit; ++i) {
                  descriptor.texture[i] = jsonToFloat(tex[i]);
                }
              }
            }
          }
          if (!descriptor.has_color && det.contains("color") && det["color"].is_array() && det["color"].size() >= 3) {
            descriptor.has_color = true;
            for (size_t i = 0; i < 3; ++i) {
              descriptor.color[i] = jsonToFloat(det["color"][i]);
            }
          }
          if (!descriptor.has_grayscale) {
            if (det.contains("grayscale_intensity")) {
              descriptor.has_grayscale = true;
              descriptor.grayscale_intensity = jsonToFloat(det["grayscale_intensity"]);
            }
            if (det.contains("texture_features") && det["texture_features"].is_array()) {
              const auto &tex = det["texture_features"];
              size_t limit = std::min<size_t>(4, tex.size());
              if (limit > 0) {
                descriptor.has_grayscale = true;
                for (size_t i = 0; i < limit; ++i) {
                  descriptor.texture[i] = jsonToFloat(tex[i]);
                }
              }
            }
          }
          LocalDetection local_detection;
          local_detection.box = cv::Rect(left, top, width, height);
          local_detection.track_id = track_id;
          local_detection.descriptor = descriptor;
          detections.push_back(local_detection);
        }
      }


      auto cycle_wall_time = std::chrono::system_clock::now();
      auto cycle_steady_time = std::chrono::steady_clock::now();
      std::ostringstream log_stream;
      log_stream << "detections=" << detections.size();
      for (const auto &det : detections) {
        log_stream << "\n  track=" << det.track_id << " bbox=[" << det.box.x << ','
                   << det.box.y << ',' << det.box.width << ',' << det.box.height
                   << ']';
      }

      auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count();
      g_global_tracker.updateDetections(cam_id, detections, ts);

      auto mapping = g_global_tracker.getTrackToGlobalMapForCamera(cam_id);
      nlohmann::json labels_json = nlohmann::json::array();
      for (const auto &assoc : mapping) {
        nlohmann::json entry;
        entry["track_id"] = assoc.track_id;
        entry["global_id"] = assoc.global_id;
        if (assoc.distance_m.has_value()) {
          entry["distance_m"] = *assoc.distance_m;
        }
        entry["visible_cameras"] = assoc.visible_camera_count;
        entry["active_cameras"] = assoc.total_active_cameras;
        labels_json.push_back(entry);
      }
      nlohmann::json payload;
      payload["camera"] = cam_id;
      payload["labels"] = labels_json;
      auto post_result =
          client.Post("/api/global-labels", payload.dump(), "application/json");

      int post_status = -1;
      std::string post_error;
      if (!post_result) {
        post_error = httplib::to_string(post_result.error());
      } else {
        post_status = post_result->status;
      }

      log_stream << "\ntrack_to_global: ";
      if (mapping.empty()) {
        log_stream << "(empty)";
      } else {
        bool first = true;
        for (const auto &assoc : mapping) {
          if (!first)
            log_stream << ", ";
          first = false;
          log_stream << assoc.track_id << "->" << assoc.global_id;
          if (assoc.distance_m.has_value()) {
            log_stream << " (" << *assoc.distance_m << "m)";
          }
        }
      }
      log_stream << "; post_status=";
      if (post_result) {
        log_stream << post_status;
      } else {
        log_stream << "error";
      }
      if (!post_error.empty()) {
        log_stream << " (" << post_error << ")";
      }
      log_stream << '\n';

      writeCameraLog(cam_id, log_stream.str(), cycle_wall_time,
                     cycle_steady_time);

      if (!post_result) {
        printf(
            "Failed to update global labels for camera %s (port %d): HTTP status %d (%s)\n",
            cam_id.c_str(), det_port, 0,
            httplib::to_string(post_result.error()).c_str());
        return;
      }
      if (post_result->status < 200 || post_result->status >= 300) {
        printf(
            "Failed to update global labels for camera %s (port %d): HTTP status %d\n",
            cam_id.c_str(), det_port, post_result->status);
        return;
      }
    }
  } catch (...) {
    // Ignore errors from individual detection fetches.
  }
}


// Cached v4l2 formats per device. Avoids reconfiguring a device that already
// runs with the requested format.
static std::unordered_map<std::string, v4l2_format> g_format_cache;
static std::mutex g_format_cache_mutex;

// Backup of global preview flag and per-camera preview states while
// calibration is in progress. This lets us restore the exact state when the
// user finishes calibration.
static bool g_prev_preview_enabled = true;
struct CamPreviewState {
  std::string id;
  bool preview;
};
static std::vector<CamPreviewState> g_prev_cam_states;

static bool fileExists(const std::string &p) {
  struct stat st {
  };
  return stat(p.c_str(), &st) == 0;
}

static bool dirExists(const std::string &p) {
  struct stat st {
  };
  return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static nlohmann::json readMainConfig() {
  nlohmann::json cfg = nlohmann::json::object();
  auto p = std::filesystem::absolute(g_config_path);
  printf("readMainConfig path: %s\n", p.c_str());
  std::ifstream f(p);
  if (f) {
    try {
      f >> cfg;
    } catch (...) {
    }
  }
  if (!cfg.is_object())
    cfg = nlohmann::json::object();
  cfg["manager_debug_enabled"] = cfg.value("manager_debug_enabled", false);
  return cfg;
}

static bool writeMainConfig(const nlohmann::json &j) {
  auto file = std::filesystem::absolute(g_config_path);
  auto dir = file.parent_path();
  printf("writeMainConfig path: %s\n", file.c_str());
  if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST)
    return false;
  std::ofstream f(file);
  if (!f)
    return false;
  auto normalized = j;
  if (!normalized.is_object())
    normalized = nlohmann::json::object();
  normalized["manager_debug_enabled"] =
      normalized.value("manager_debug_enabled", false);
  f << normalized.dump(2);
  return f.good();
}


static void sigint(int) {
  g_mgr.stop();
  ensureCalibrationWatcher();
  g_global_tracker.initialize();
  printf("Global tracker initialized\n");
  g_server.stop();
}

static bool formatsEqual(const v4l2_format &a, const v4l2_format &b) {
  if (a.type != b.type)
    return false;
  if (a.type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
    return a.fmt.pix_mp.width == b.fmt.pix_mp.width &&
           a.fmt.pix_mp.height == b.fmt.pix_mp.height &&
           a.fmt.pix_mp.pixelformat == b.fmt.pix_mp.pixelformat;
  }
  return a.fmt.pix.width == b.fmt.pix.width &&
         a.fmt.pix.height == b.fmt.pix.height &&
         a.fmt.pix.pixelformat == b.fmt.pix.pixelformat;
}

// Populate cached formats for all currently active devices.
static void populate_format_cache() {
  std::lock_guard<std::mutex> lk(g_format_cache_mutex);
  g_format_cache.clear();
  auto cams = g_mgr.configuredCameras();
  for (auto &c : cams) {
    std::string dev = g_mgr.devicePath(c.id);
    if (dev.empty())
      continue;
    int fd = open(dev.c_str(), O_RDWR);
    if (fd < 0)
      continue;
    v4l2_format f{};
    v4l2_buf_type types[] = {V4L2_BUF_TYPE_VIDEO_CAPTURE,
                             V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE};
    for (auto t : types) {
      std::memset(&f, 0, sizeof(f));
      f.type = t;
      if (ioctl(fd, VIDIOC_G_FMT, &f) == 0) {
        g_format_cache[dev] = f;
        break;
      }
    }
    close(fd);
  }
}


static bool capture_jpeg(const std::string &dev,
                         std::vector<unsigned char> &out,
                         const std::string &cam_id = std::string(),
                         v4l2_buf_type req_type = static_cast<v4l2_buf_type>(0)) {

  // If cam_id is provided, the camera is already streaming and frames are
  // available via the CameraManager buffer. Avoid touching the device again
  // and fetch the latest frame directly from the running stream.
  if (!cam_id.empty()) {
    CameraManager::Frame fr;
    if (!g_mgr.getFrame(cam_id, g_mgr.nowMonoNs(), fr))
      return false;
    out = fr.jpeg;
    return true;
  }

  int fd = open(dev.c_str(), O_RDWR);
  if (fd < 0) {
    std::cerr << "Failed to open " << dev << ": "
              << std::strerror(errno) << std::endl;
    return false;
  }

 v4l2_buf_type type = req_type;
  if (type != V4L2_BUF_TYPE_VIDEO_CAPTURE &&
      type != V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
    v4l2_capability cap{};
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0) {
      if (cap.device_caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE)
        type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
      else
        type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    }
  }

  v4l2_format fmt{};
  bool fmt_ok = false;
  std::vector<v4l2_buf_type> try_types;
  if (type == V4L2_BUF_TYPE_VIDEO_CAPTURE ||
      type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE)
    try_types.push_back(type);
  else {
    try_types.push_back(V4L2_BUF_TYPE_VIDEO_CAPTURE);
    try_types.push_back(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE);
  }


  for (auto t : try_types) {
    v4l2_format desired{};
    std::memset(&desired, 0, sizeof(desired));
    desired.type = t;
    if (t == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
      desired.fmt.pix_mp.width = 320;
      desired.fmt.pix_mp.height = 240;
      desired.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_MJPEG;
      desired.fmt.pix_mp.field = V4L2_FIELD_NONE;
      desired.fmt.pix_mp.num_planes = 1;
    } else {
      desired.fmt.pix.width = 320;
      desired.fmt.pix.height = 240;
      desired.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
      desired.fmt.pix.field = V4L2_FIELD_NONE;
    }

    v4l2_format cur{};
    std::memset(&cur, 0, sizeof(cur));
    cur.type = t;
    bool have_cur = (ioctl(fd, VIDIOC_G_FMT, &cur) == 0);
    if (have_cur) {
      {
        std::lock_guard<std::mutex> lk(g_format_cache_mutex);
        g_format_cache[dev] = cur;
      }
      if (formatsEqual(cur, desired)) {
        fmt = cur;
        type = t;
        fmt_ok = true;
        break;
      }
    }

    if (ioctl(fd, VIDIOC_S_FMT, &desired) == 0) {
      fmt = desired;
      type = t;
      fmt_ok = true;
      {
        std::lock_guard<std::mutex> lk(g_format_cache_mutex);
        g_format_cache[dev] = fmt;
      }
      break;
    }
  }
  if (!fmt_ok) {
    int err = errno;
    std::cerr << "VIDIOC_S_FMT failed for " << dev << ": "
              << std::strerror(err) << std::endl;
    close(fd);
    return false;
  }

  v4l2_requestbuffers req{};
  req.count = 1;
  req.type = type;
  req.memory = V4L2_MEMORY_MMAP;
  if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
    int err = errno;
    std::cerr << "VIDIOC_REQBUFS failed for " << dev << ": "
              << std::strerror(err) << std::endl;
    close(fd);
    return false;
  }
  v4l2_buffer buf{};
  buf.type = type;
  buf.memory = V4L2_MEMORY_MMAP;
  buf.index = 0;
  v4l2_plane planes[VIDEO_MAX_PLANES];
  if (type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
    std::memset(planes, 0, sizeof(planes));
    buf.length = 1;
    buf.m.planes = planes;
  }
  if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
    int err = errno;
    std::cerr << "VIDIOC_QUERYBUF failed for " << dev << ": "
              << std::strerror(err) << std::endl;
    close(fd);
    return false;
  }

  void *mem = nullptr;
  size_t map_len = 0;
  if (type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
    mem = mmap(NULL, buf.m.planes[0].length, PROT_READ | PROT_WRITE,
               MAP_SHARED, fd, buf.m.planes[0].m.mem_offset);
    map_len = buf.m.planes[0].length;
  } else {
    mem = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
               buf.m.offset);
    map_len = buf.length;
  }
  if (mem == MAP_FAILED) {
    int err = errno;
    std::cerr << "mmap failed for " << dev << ": " << std::strerror(err)
              << std::endl;
    close(fd);
    return false;
  }

  if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
    int err = errno;
    std::cerr << "VIDIOC_QBUF failed for " << dev << ": "
              << std::strerror(err) << std::endl;
    munmap(mem, map_len);
    close(fd);
    return false;
  }

  if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
    int err = errno;
    std::cerr << "VIDIOC_STREAMON failed for " << dev << ": "
              << std::strerror(err) << std::endl;
    munmap(mem, map_len);
    close(fd);
    return false;
  }
  if (ioctl(fd, VIDIOC_DQBUF, &buf) < 0) {
    int err = errno;
    std::cerr << "VIDIOC_DQBUF failed for " << dev << ": "
              << std::strerror(err) << std::endl;
    ioctl(fd, VIDIOC_STREAMOFF, &type);
    munmap(mem, map_len);
    close(fd);
    return false;
  }

  size_t used = (type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE)
                    ? buf.m.planes[0].bytesused
                    : buf.bytesused;

  out.assign(static_cast<unsigned char *>(mem),
             static_cast<unsigned char *>(mem) + used);
  ioctl(fd, VIDIOC_STREAMOFF, &type);
  munmap(mem, map_len);
  close(fd);
  if (!cam_id.empty()) {
    uint64_t t_ns = g_mgr.nowMonoNs();
    int w = (type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE)
                ? fmt.fmt.pix_mp.width
                : fmt.fmt.pix.width;
    int h = (type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE)
                ? fmt.fmt.pix_mp.height
                : fmt.fmt.pix.height;
    g_mgr.pushFrame(cam_id, w, h, out, t_ns);
  }
  return true;
}


// Захват кадра в формате cv::Mat. Используется существующая функция
// capture_jpeg, после чего изображение декодируется в градации серого.
static cv::Mat capture_mat(const std::string &dev) {
  std::vector<unsigned char> buf;
  if (!capture_jpeg(dev, buf, std::string(), V4L2_BUF_TYPE_VIDEO_CAPTURE))
    return cv::Mat();
  return cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
}

// Флаг работы стерео-потока.
static std::atomic<bool> g_stereo_running{true};

// Основной цикл обработки стереопар. Для каждой активной пары вычисляется
// карта диспаритета, точки переводятся в систему cam0, после чего карты глубин
// объединяются. Также выполняется простой KLT-трекер для оценки движения
// между кадрами.
static void stereo_loop() {
  cv::Mat prev_gray;
  std::vector<cv::Point2f> prev_pts;
  while (g_stereo_running) {
    cv::Mat merged;
    auto pairs = g_mgr.getActivePairs();
    for (auto &pair : pairs) {
      std::string dev0 = g_mgr.devicePath(pair.cam0);
      std::string dev1 = g_mgr.devicePath(pair.cam1);
      if (dev0.empty() || dev1.empty())
        continue;
      cv::Mat left = capture_mat(dev0);
      cv::Mat right = capture_mat(dev1);
      if (left.empty() || right.empty())
        continue;
      cv::Mat disp;
      pair.matcher->compute(left, right, disp);
      cv::Mat pts3d;
      cv::reprojectImageTo3D(disp, pts3d, pair.Q);
      cv::Mat zmap;
      cv::extractChannel(pts3d, zmap, 2);
      if (merged.empty())
        merged = zmap;
      else
        cv::min(merged, zmap, merged);
    }

    if (!merged.empty()) {
      cv::Mat gray;
      merged.convertTo(gray, CV_8U, 255.0 / 10.0);
      if (prev_pts.empty()) {
        cv::goodFeaturesToTrack(gray, prev_pts, 200, 0.01, 3);
      } else {
        std::vector<cv::Point2f> next_pts;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, next_pts, status,
                                 err);
        prev_pts.clear();
        for (size_t i = 0; i < status.size(); ++i) {
          if (status[i])
            prev_pts.push_back(next_pts[i]);
        }
      }
      prev_gray = gray;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(30));
  }
}



int main(int argc, char **argv) {

  std::filesystem::path exe_dir = std::filesystem::canonical(argv[0]).parent_path();
  g_exe_dir = exe_dir;
  g_config_path = argc > 1 ? std::filesystem::path(argv[1]) : exe_dir / "config.json";

  g_role_mgr.loadSystemConfiguration(MultiCamera::SystemConfiguration::HEMISPHERE,
                                     (exe_dir / "roles.json").string());


  if (!g_mgr.loadConfig(g_config_path.string()))
    return 1;
  std::ifstream jf(g_config_path);
  nlohmann::json j = nlohmann::json::object();
  if (jf) {
    try {
      jf >> j;
    } catch (...) {
      j = nlohmann::json::object();
    }
  }
  if (!j.is_object())
    j = nlohmann::json::object();
  g_preview_enabled = j.value("preview_enabled", true);
  g_use_grayscale_tracking = j.value("use_grayscale_tracking", false);
  bool manager_debug = j.value("manager_debug_enabled", false);
  setManagerDebugEnabled(manager_debug);
  int port = j.value("http", nlohmann::json::object()).value("port", 8080);

  g_calib = std::make_unique<CalibrationSession>(
      g_mgr, g_preview_enabled,
      std::filesystem::absolute(readMainConfig().value("calib_root", ".")));

  std::signal(SIGINT, sigint);
  g_mgr.start();

  g_scheme_manager.initialize(g_config_path.string());
  ensureCalibrationWatcher();
  g_global_tracker.initialize();


  // Initialize camera roles from current manager configuration
  for (const auto &c : g_mgr.configuredCameras()) {
    MultiCamera::CameraRole r = MultiCamera::CameraRole::WIDE_ANGLE;
    if (c.role == "zoom")
      r = MultiCamera::CameraRole::ZOOM;
    else if (c.role == "zoom_variable")
      r = MultiCamera::CameraRole::ZOOM_VARIABLE;
    g_role_mgr.assignRole(c.id, r);
  }


  // Seed format cache with current formats of active devices.
  populate_format_cache();



  // Отдельный поток обработки стереопар.
  std::thread stereo_thread(stereo_loop);


  g_server.set_mount_point("/", "./web");

  g_server.Get("/api/status",
               [](const httplib::Request &, httplib::Response &res) {
                 bool detect = false;
                 for (auto &c : g_mgr.configuredCameras()) {
                   if (c.det_running) {
                     detect = true;
                     break;
                   }
                 }
                 nlohmann::json out{{"preview", g_preview_enabled},
                                      {"detect", detect}};
                 res.set_content(out.dump(), "application/json");
               });

  g_server.Get("/api/config", [](const httplib::Request &, httplib::Response &res) {
    auto config = readMainConfig();
    config["preview_enabled"] = g_preview_enabled;
    config["grayscale_tracking"] =
        g_use_grayscale_tracking; // Include current grayscale mode
    config["manager_debug_enabled"] =
        g_manager_debug_enabled.load(std::memory_order_acquire);
    res.set_content(config.dump(), "application/json");
  });


  g_server.Post("/api/config", [](const httplib::Request &req,
                                   httplib::Response &res) {
    try {
      auto body = nlohmann::json::parse(req.body);
      auto config = readMainConfig();

      bool current = g_manager_debug_enabled.load(std::memory_order_acquire);
      bool desired = body.value("manager_debug_enabled", current);

      if (current != desired)
        setManagerDebugEnabled(desired);

      config["manager_debug_enabled"] = desired;
      if (!writeMainConfig(config)) {
        if (current != desired)
          setManagerDebugEnabled(current);
        res.status = 500;
        res.set_content("{\"error\":\"write failure\"}",
                        "application/json");
        return;
      }

      config["preview_enabled"] = g_preview_enabled;
      config["grayscale_tracking"] =
          g_use_grayscale_tracking; // Include current grayscale mode
      config["manager_debug_enabled"] =
          g_manager_debug_enabled.load(std::memory_order_acquire);
      res.set_content(config.dump(), "application/json");
    } catch (...) {
      res.status = 400;
      res.set_content("{\"error\":\"invalid json\"}",
                      "application/json");
    }
  });



  g_server.Get(
      "/api/configured", [](const httplib::Request &, httplib::Response &res) {
        auto cams = g_mgr.configuredCameras();
        nlohmann::json out = nlohmann::json::array();
        for (auto &c : cams)
          out.push_back({{"id", c.id},
                         {"present", c.present},
                         {"mode",
                          (c.mode == CameraManager::CamConfig::Mode::Detect)
                              ? "detect"
                              : (c.mode == CameraManager::CamConfig::Mode::Calibration)
                                    ? "calibration"
                                    : "preview"},
                         {"preferred",
                         {{"w", c.preferred.w},
                          {"h", c.preferred.h},
                          {"pixfmt", c.preferred.pixfmt},
                          {"fps", c.preferred.fps}}},
                         {"npu_worker", c.npu_worker},
                         {"auto_profiles", c.auto_profiles},
                         {"profile", c.profile},
                         {"det_port", c.det_port},
                         {"det_running", c.det_running},
                         {"position", {{"x", c.position.x}, {"y", c.position.y}, {"z", c.position.z}}},
                         {"fps", c.fps},
                         {"model_path", c.model_path},
                         {"labels_path", c.labels_path},
                         {"cap_fps", c.cap_fps},
                         {"buffers", c.buffers},
                         {"buffer_type", c.buffer_type},
                         {"jpeg_quality", c.jpeg_quality},
                         {"http_fps_limit", c.http_fps_limit},
                         {"show_fps", c.show_det_fps},
                         {"npu_core", c.npu_core},
                         {"log_file", c.log_file}});
        res.set_content(out.dump(), "application/json");
      });

  g_server.Get("/api/models",
               [](const httplib::Request &, httplib::Response &res) {
                 nlohmann::json out;
                 out["rknn"] = nlohmann::json::array();
                 out["labels"] = nlohmann::json::array();
                 namespace fs = std::filesystem;
                 try {
                   for (auto &p : fs::directory_iterator("model_rknn"))
                     if (p.is_regular_file())
                       out["rknn"].push_back(std::string("model_rknn/") +
                                             p.path().filename().string());
                 } catch (...) {
                 }
                 try {
                   for (auto &p : fs::directory_iterator("model"))
                     if (p.is_regular_file())
                       out["labels"].push_back(std::string("model/") +
                                               p.path().filename().string());
                 } catch (...) {
                 }
                 res.set_content(out.dump(), "application/json");
               });

  g_server.Get("/api/new",
               [](const httplib::Request &, httplib::Response &res) {
                 nlohmann::json out = g_mgr.unconfiguredCameras();
                 res.set_content(out.dump(), "application/json");
               });

  g_server.Post("/api/add",
                [](const httplib::Request &req, httplib::Response &res) {
                  try {
                    auto j = nlohmann::json::parse(req.body);
                    std::string id = j.at("id").get<std::string>();
                    std::string by = j.at("by_id").get<std::string>();
                    if (!g_mgr.addCamera(id, by))
                      res.status = 400;
                  } catch (...) {
                    res.status = 400;
                  }
                });

  g_server.Post("/api/delete",
                [](const httplib::Request &req, httplib::Response &res) {
                  try {
                    auto j = nlohmann::json::parse(req.body);
                    std::string id = j.at("id").get<std::string>();
                    if (!g_mgr.removeCamera(id))
                      res.status = 400;
                  } catch (...) {
                    res.status = 400;
                  }
                });

/*
  g_server.Get("/api/roles",
               [](const httplib::Request &, httplib::Response &res) {
                 nlohmann::json out = nlohmann::json::array();
                 for (const auto &cfg : g_role_mgr.getAllCameraConfigs()) {
                   std::string role = "wide_angle";
                   if (cfg.role == MultiCamera::CameraRole::ZOOM)
                     role = "zoom";
                   else if (cfg.role == MultiCamera::CameraRole::ZOOM_VARIABLE)
                     role = "zoom_variable";
                   out.push_back({{"id", cfg.camera_id}, {"role", role}});
                 }
                 res.set_content(out.dump(), "application/json");
               });
*/
  g_server.Get(
      "/api/roles",
      [](const httplib::Request &, httplib::Response &res) {
        res.set_content(g_role_mgr.getRoles().dump(), "application/json");
      });
/*
  g_server.Post("/api/roles",
                [](const httplib::Request &req, httplib::Response &res) {
                  try {
                    auto j = nlohmann::json::parse(req.body);
                    std::string id = j.at("id").get<std::string>();
                    std::string role_str = j.at("role").get<std::string>();
                    MultiCamera::CameraRole role =
                        role_str == "zoom"
                            ? MultiCamera::CameraRole::ZOOM
                            : (role_str == "zoom_variable"
                                   ? MultiCamera::CameraRole::ZOOM_VARIABLE
                                   : MultiCamera::CameraRole::WIDE_ANGLE);
                    if (!g_role_mgr.assignRole(id, role) ||
                        !g_mgr.setRole(id, role_str)) {
                      res.status = 400;
                      return;
                    }
                    std::map<std::string, std::string> roles_map;
                    for (const auto &cfg : g_role_mgr.getAllCameraConfigs()) {
                      std::string r = "wide_angle";
                      if (cfg.role == MultiCamera::CameraRole::ZOOM)
                        r = "zoom";
                      else if (cfg.role == MultiCamera::CameraRole::ZOOM_VARIABLE)
                        r = "zoom_variable";
                      roles_map[cfg.camera_id] = r;
                    }
                    g_mgr.saveConfig(g_mgr.schemeType(), roles_map);
                  } catch (...) {
                    res.status = 400;
                  }
                });
*/
  g_server.Post(
      "/api/roles",
      [](const httplib::Request &req, httplib::Response &res) {
        try {
          auto j = nlohmann::json::parse(req.body);
          std::string id = j.at("id").get<std::string>();
          std::string role_str = j.at("role").get<std::string>();
          if (!g_role_mgr.assignRole(id, role_str) ||
              !g_mgr.setRole(id, role_str)) {
            res.status = 400;
            return;
          }
          std::map<std::string, std::string> roles_map;
          for (const auto &r : g_role_mgr.getRoles()) {
            roles_map[r.at("id").get<std::string>()] =
                r.at("role").get<std::string>();
          }
          g_mgr.saveConfig(g_mgr.schemeType(), roles_map);
        } catch (...) {
          res.status = 400;
        }
      });

  g_server.Post("/api/calib/setup",
                [](const httplib::Request &req, httplib::Response &res) {
                  try {
                    auto j = nlohmann::json::parse(req.body);
                    std::string cam = j.value("camera", "");
                    if (cam.empty()) {
                      res.status = 400;
                      res.set_content("{\"error\":\"missing camera\"}",
                                      "application/json");
                      return;
                    }
                    auto cfg = readMainConfig();
                    cfg["calib_camera"] = cam;
                    if (!writeMainConfig(cfg)) {
                      res.status = 500;
                      res.set_content("{\"error\":\"write failure\"}",
                                      "application/json");
                      return;
                    }
                    std::filesystem::path dir = std::filesystem::current_path() /
                                              "calibration" /
                                              ("cam_" + cam) / "images";
                    std::error_code ec;
                    std::filesystem::create_directories(dir, ec);
                    auto absDir = std::filesystem::absolute(dir);
                    printf("calibration dir: %s\n", absDir.c_str());
                    if (ec) {
                      res.status = 500;
                      res.set_content("{\"error\":\"mkdir failure\"}",
                                      "application/json");
                      return;
                    }
                    res.set_content("{\"status\":\"ok\"}",
                                    "application/json");
                  } catch (...) {
                    res.status = 400;
                    res.set_content("{\"error\":\"invalid json\"}",
                                    "application/json");
                  }
                });

  g_server.Get("/api/calib/status",
               [](const httplib::Request &req, httplib::Response &res) {
                 std::string cam;
                 if (req.has_param("camera")) {
                   cam = req.get_param_value("camera");
                 } else {
                   auto cfg = readMainConfig();
                   cam = cfg.value("calib_camera", "");
                 }
                 nlohmann::json resp;
                 resp["camera"] = cam;
                 std::string dir = cam.empty()
                                        ? std::string()
                                        : "calibration/cam_" + cam + "/images";
                 bool folder = !cam.empty() && dirExists(dir);
                 bool mono_done =
                     !cam.empty() &&
                     fileExists("calibration/results/cam_" + cam + ".yml");
                 bool stereo_ready =
                     mono_done &&
                     fileExists("calibration/results/cam_0.yml") &&
                     !fileExists("calibration/results/stereo_0_" + cam +
                                 ".yml");
                 resp["folder_exists"] = folder;
                 resp["mono_done"] = mono_done;
                 resp["stereo_ready"] = stereo_ready;
                 res.set_content(resp.dump(), "application/json");
               });
  g_server.Post("/api/calib/start",
                [](const httplib::Request &, httplib::Response &res) {
                  g_calib->start();
                  res.set_content("{\"status\":\"ok\"}", "application/json");
                });

  g_server.Post("/api/calib/stop",
                [](const httplib::Request &, httplib::Response &res) {
                  // Restore per-camera preview states before restarting.
                  g_calib->stop();
                  res.set_content("{\"status\":\"ok\"}", "application/json");
                });

  g_server.Post("/api/calibration/start",
                [](const httplib::Request &, httplib::Response &res) {
                  g_calib->start();
                  nlohmann::json j; j["status"] = "ok";
                  res.set_content(j.dump(), "application/json");
                });

  g_server.Post("/api/calibration/stop",
                [](const httplib::Request &, httplib::Response &res) {
                  g_calib->stop();
                  nlohmann::json j; j["status"] = "ok";
                  res.set_content(j.dump(), "application/json");
                });

  g_server.Post("/api/calibration/run",
                [](const httplib::Request &req, httplib::Response &res) {
                  nlohmann::json resp;
                  bool started = false;
                  try {
                    auto j = nlohmann::json::parse(req.body);
                    auto ids = j.at("ids").get<std::vector<std::string>>();
                    int duration = j.value("duration", 30);
                    int bw = j.value("board_w", 0);
                    int bh = j.value("board_h", 0);
                    g_calib->start();
                    started = true;
                    char q = '"';
                    std::string cmd = std::string(1, q) + (g_exe_dir / "calibration_cli").string() + std::string("\" ");
                    if (ids.size() == 1) {
                      cmd += "mono " + ids[0];
                    } else {
                      cmd += "stereo";
                      for (auto &id : ids) cmd += " " + id;
                    }
                    cmd += " " + std::to_string(duration);
                    if (bw > 0 && bh > 0) {
                      cmd += " --board " + std::to_string(bw) + "x" + std::to_string(bh);
                    }
                    cmd += " --config " + std::string(1, q) + g_config_path.string() + std::string(1, q);
                    FILE *pipe = popen(cmd.c_str(), "r");
                    if (!pipe) {
                      resp["error"] = "popen failed";
                      res.status = 500;
                    } else {
                      std::string output; char buffer[256];
                      while (fgets(buffer, sizeof(buffer), pipe)) output += buffer;
                      int rc = pclose(pipe);
                      if (rc == 0) {
                        try { resp["paths"] = nlohmann::json::parse(output); }
                        catch (...) { resp["paths"] = nlohmann::json::array(); }
                      } else {
                        resp["error"] = rc;
                        res.status = 500;
                      }
                    }
                  } catch (const std::exception &e) {
                    resp["error"] = e.what();
                    res.status = 400;
                  }
                  if (started) g_calib->stop();
                  res.set_content(resp.dump(), "application/json");
                });


  g_server.Post(
      "/api/calib/stereo-capture",
      [](const httplib::Request &req, httplib::Response &res) {
        try {
          auto j = nlohmann::json::parse(req.body);
          if (!j.contains("cameras") || !j["cameras"].is_array()) {
            res.status = 400;
            res.set_content("{\"error\":\"missing cameras\"}",
                            "application/json");

            return;
          }
          std::vector<std::string> cams =
              j["cameras"].get<std::vector<std::string>>();
          int frames = j.value("frames", 0);
          int interval = j.value("interval", 0);
          auto r = g_calib->captureStereo(cams, frames, interval);
          res.status = r.status;
          res.set_content(r.body.dump(), "application/json");
        } catch (...) {
          res.status = 400;
          res.set_content("{\"error\":\"invalid json\"}",
                          "application/json");
        }
      });


  g_server.Post(
      "/api/calib/mono",
      [](const httplib::Request &req, httplib::Response &res) {
        try {
          auto j = nlohmann::json::parse(req.body);
          std::string cam = j.value("camera", "");
          if (cam.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"missing camera\"}",
                            "application/json");
            return;
          }
          auto r = g_calib->captureMono(cam);
          res.status = r.status;
          res.set_content(r.body.dump(), "application/json");
        } catch (...) {
          res.status = 400;
          res.set_content("{\"error\":\"invalid json\"}",
                          "application/json");
        }
      });

 g_server.Post(
      "/api/calib/mono/start",
      [](const httplib::Request &req, httplib::Response &res) {
        try {
          auto j = nlohmann::json::parse(req.body);
          std::string cam = j.value("camera", "");
          int bw = j.value("board_w", 0);
          int bh = j.value("board_h", 0);
          if (cam.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"missing camera\"}",
                            "application/json");
            return;
          }
          int job = g_calib->startMonoJob(cam, bw, bh);
          nlohmann::json out; out["job_id"] = job;
          res.set_content(out.dump(), "application/json");
        } catch (...) {
          res.status = 400;
          res.set_content("{\"error\":\"invalid json\"}",
                          "application/json");
        }
      });

  g_server.Get(
      "/api/calib/mono/progress",
      [](const httplib::Request &req, httplib::Response &res) {
        if (!req.has_param("job_id")) {
          res.status = 400;
          res.set_content("{\"error\":\"missing job_id\"}",
                          "application/json");
          return;
        }
        int job = std::stoi(req.get_param_value("job_id"));
        auto r = g_calib->monoProgress(job);
        res.status = r.status;
        res.set_content(r.body.dump(), "application/json");
      });


  g_server.Post("/api/calib/prepare",
                [](const httplib::Request &, httplib::Response &res) {
                  g_mgr.stop();
                  // Keep preview so the web UI can show frames during setup.
                  g_preview_enabled = true;
                  res.set_content("{\"status\":\"ok\"}", "application/json");
                });

  g_server.Post(
      "/api/preview/enable",
      [](const httplib::Request &req, httplib::Response &res) {
        try {
          auto j = nlohmann::json::parse(req.body);
          g_preview_enabled = j.at("enable").get<bool>();
          res.set_content("{\"status\":\"ok\"}", "application/json");
        } catch (...) {
          res.status = 400;
        }
      });



  g_server.Post("/api/preview",
                [](const httplib::Request &req, httplib::Response &res) {
                if (!g_preview_enabled) {
                  res.status = 403;
                  return;
                }
                try {
                  auto j = nlohmann::json::parse(req.body);
                  std::string id = j.at("id").get<std::string>();
                  std::string mode = j.at("mode").get<std::string>();
                  bool ok = false;
                  if (mode == "preview")
                    ok = g_mgr.setMode(id, CameraManager::CamConfig::Mode::Preview);
                  else if (mode == "detect")
                    ok = g_mgr.setMode(id, CameraManager::CamConfig::Mode::Detect);
                  else if (mode == "calibration")
                    ok = g_mgr.setMode(id, CameraManager::CamConfig::Mode::Calibration);
                  else {
                    res.status = 400;
                    return;
                  }
                  if (!ok)
                    res.status = 400;
                  else
                    g_mgr.notify();
                } catch (...) {
                  res.status = 400;
                }
                });

  g_server.Post("/api/settings",
                [](const httplib::Request &req, httplib::Response &res) {
                  try {
                    auto j = nlohmann::json::parse(req.body);
                    std::string id = j.at("id").get<std::string>();
                    auto pref = j.at("preferred");
                    CameraManager::CamConfig::VideoMode vm;
                    vm.w = pref.value("w", 1280);
                    vm.h = pref.value("h", 720);
                    vm.pixfmt = pref.value("pixfmt", std::string("MJPG"));
                    vm.fps = pref.value("fps", 30);
                    int worker = j.value("npu_worker", 0);
                    bool auto_profiles = j.value("auto_profiles", true);
                    std::string profile =
                        j.value("profile", std::string("auto"));
                    std::string model_path =
                        j.value("model_path", std::string(""));
                    std::string labels_path =
                        j.value("labels_path", std::string(""));
                    int cap_fps = j.value("cap_fps", 30);
                    int buffers = j.value("buffers", 3);
                    std::string buffer_type =
                        j.value("buffer_type", std::string("auto"));
                    if (buffer_type != "auto" && buffer_type != "single" &&
                        buffer_type != "mplane") {
                      res.status = 400;
                      return;
                    }
                    int jpeg_quality = j.value("jpeg_quality", 60);
                    int http_fps_limit = j.value("http_fps_limit", 20);
                    bool show_fps = j.value("show_fps", false);
                    std::string npu_core =
                        j.value("npu_core", std::string("auto"));
                    std::string log_file =
                        j.value("log_file", std::string(""));
                    if (!g_mgr.updateSettings(id, vm, worker, auto_profiles,
                                              profile, model_path, labels_path,
                                              cap_fps, buffers, buffer_type,
                                              jpeg_quality, http_fps_limit,
                                              show_fps,
                                              npu_core, log_file))
                      res.status = 400;
                  } catch (...) {
                    res.status = 400;
                  }
                });

 g_server.Post("/api/settings/reset",
                [](const httplib::Request &req, httplib::Response &res) {
                  try {
                    auto j = nlohmann::json::parse(req.body);
                    std::string id = j.at("id").get<std::string>();
                    if (!g_mgr.resetSettings(id))
                      res.status = 400;
                  } catch (...) {
                    res.status = 400;
                  }
                });
 
 g_server.Get(
      "/api/preview", [](const httplib::Request &req, httplib::Response &res) {
     if (!g_preview_enabled) {
          res.status = 403;
          return;
        }

        if (req.has_param("id")) {
          std::string id = req.get_param_value("id");
          CameraManager::Frame fr;
          if (!g_mgr.getFrame(id, g_mgr.nowMonoNs(), fr)) {
            res.status = 404;
            return;
          }
          g_mgr.reportFrame(id);
          res.set_content(reinterpret_cast<const char *>(fr.jpeg.data()),
                          fr.jpeg.size(), "image/jpeg");
          return;
        }


        std::string dev;
        if (req.has_param("by"))
          dev = std::string("/dev/v4l/by-id/") + req.get_param_value("by");
        if (dev.empty()) {
          res.status = 404;
          return;
        }

        std::vector<unsigned char> jpg;
        if (!capture_jpeg(dev, jpg, std::string(), V4L2_BUF_TYPE_VIDEO_CAPTURE)) {
          std::cerr << "capture_jpeg failed for device '" << dev << "'" << std::endl;
          res.status = 404;
          return;
        }
        res.set_content(reinterpret_cast<const char *>(jpg.data()), jpg.size(),
                        "image/jpeg");
      });

 g_server.Get(
      "/api/preview.mjpg",
      [](const httplib::Request &req, httplib::Response &res) {
        if (!g_preview_enabled) {
          res.status = 403;
          return;
        }
        std::string dev;
        std::string id;
        int http_fps_limit = 0;
        if (req.has_param("id")) {
          id = req.get_param_value("id");
          for (auto &ci : g_mgr.configuredCameras()) {
            if (ci.id == id) {
              http_fps_limit = ci.http_fps_limit;
              break;
            }
          }
        } else if (req.has_param("by")) {
          dev = std::string("/dev/v4l/by-id/") + req.get_param_value("by");
        } else {
          res.status = 404;
          return;
        }

        res.set_header("Cache-Control",
                       "no-store, no-cache, must-revalidate");
        res.set_header("Pragma", "no-cache");
        res.set_header("Connection", "keep-alive");
        auto last_push = std::chrono::steady_clock::now();
        res.set_chunked_content_provider(
            "multipart/x-mixed-replace; boundary=frame",
            [dev, id, http_fps_limit, last_push](size_t,
                                                httplib::DataSink &sink) mutable {
              while (true) {
                std::vector<unsigned char> jpg;
                if (!id.empty()) {
                  CameraManager::Frame fr;
                  if (!g_mgr.getFrame(id, g_mgr.nowMonoNs(), fr))
                    break;
                  jpg = std::move(fr.jpeg);
                  g_mgr.reportFrame(id);
                } else {
                  if (!capture_jpeg(dev, jpg, std::string(),
                                    V4L2_BUF_TYPE_VIDEO_CAPTURE))
                    break;
                }
                if (http_fps_limit > 0) {
                  auto now = std::chrono::steady_clock::now();
                  double elapsed =
                      std::chrono::duration<double>(now - last_push).count();
                  double min_dt = 1.0 / http_fps_limit;
                  if (elapsed < min_dt) {
                    auto sleep_d =
                        std::chrono::duration<double>(min_dt - elapsed);
                    std::this_thread::sleep_for(
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            sleep_d));
                  }
                  last_push = std::chrono::steady_clock::now();
                }

                std::string header =
                    "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                    std::to_string(jpg.size()) + "\r\n\r\n";
                if (!sink.write(header.data(), header.size()))
                  break;
                if (!sink.write(reinterpret_cast<const char *>(jpg.data()),
                                jpg.size()))
                  break;
                if (!sink.write("\r\n", 2))
                  break;
              }
              return true;
            },
            [](bool) {});
      });

 g_server.Get(
      "/api/capture", [](const httplib::Request &req, httplib::Response &res) {
        if (!req.has_param("id")) {
          res.status = 400;
          return;
        }
        std::string id = req.get_param_value("id");
        uint64_t t_ns = g_mgr.nowMonoNs();
        if (req.has_param("t")) {
          try {
            t_ns = std::stoull(req.get_param_value("t"));
          } catch (...) {
          }
        }
        CameraManager::Frame fr;
        if (!g_mgr.getFrame(id, t_ns, fr)) {
          res.status = 404;
          return;
        }
        std::string dir = "captures";
        std::filesystem::create_directories(dir);
        std::string file = dir + "/" + id + "_" + std::to_string(fr.t_mono_ns) + ".jpg";
        if (!g_mgr.saveFrameWithMeta(file, fr, id)) {
          res.status = 500;
          return;
        }
        nlohmann::json j;
        j["path"] = file;
        j["t_mono_ns"] = fr.t_mono_ns;
        res.set_content(j.dump(), "application/json");
      });

g_server.Post("/api/record/start", [](const httplib::Request& req, httplib::Response& res){
    nlohmann::json resp;
    nlohmann::json req_json;
    try {
        if (req.body.empty()) {
            throw std::runtime_error("empty body");
        }
        req_json = nlohmann::json::parse(req.body);
    } catch (...) {
        resp["status"] = "error";
        resp["message"] = "Invalid JSON";
        res.status = 400;
        res.set_content(resp.dump(), "application/json");
        return;
    }

    const int duration = req_json.value("duration", 30);
    const std::string mode = req_json.value("mode", std::string("mono"));

    std::vector<std::string> camera_ids;
    if (req_json.contains("camera_ids") && req_json["camera_ids"].is_array()) {
        for (const auto& item : req_json["camera_ids"]) {
            if (!item.is_string()) {
                resp["status"] = "error";
                resp["message"] = "camera_ids must contain strings";
                res.status = 400;
                res.set_content(resp.dump(), "application/json");
                return;
            }
            camera_ids.push_back(item.get<std::string>());
        }

    }

    if (camera_ids.empty()) {
        resp["status"] = "error";
        resp["message"] = "camera_ids is required";
        res.status = 400;
        res.set_content(resp.dump(), "application/json");
        return;
    }

    size_t expected_cameras = 0;
    if (mode == "mono") {
        expected_cameras = 1;
    } else if (mode == "stereo") {
        expected_cameras = 2;
    }

    if (expected_cameras == 0) {
        resp["status"] = "error";
        resp["message"] = "Unsupported recording mode";
        res.status = 400;
        res.set_content(resp.dump(), "application/json");
        return;
    }

    if (camera_ids.size() != expected_cameras) {
        resp["status"] = "error";
        resp["message"] = "Recording mode does not match number of cameras";
        res.status = 400;
        res.set_content(resp.dump(), "application/json");
        return;
    }

    auto configured = g_mgr.configuredCameras();
    std::unordered_map<std::string, CameraManager::ConfiguredInfo> configured_map;
    configured_map.reserve(configured.size());
    for (const auto& cfg : configured) {
        configured_map.emplace(cfg.id, cfg);
    }

    std::string base_prefix;
    if (mode == "mono" && !camera_ids.empty()) {
        base_prefix = "mono_" + camera_ids.front();
    } else if (mode == "stereo" && camera_ids.size() == 2) {
        std::vector<std::string> sorted_ids = camera_ids;
        std::sort(sorted_ids.begin(), sorted_ids.end());
        base_prefix = "stereo_" + sorted_ids[0] + "_" + sorted_ids[1];
    }
    nlohmann::json results = nlohmann::json::array();
    size_t success_count = 0;

    for (const auto& camera_id : camera_ids) {
        nlohmann::json cam_result;
        cam_result["camera_id"] = camera_id;
        auto it = configured_map.find(camera_id);
        if (it == configured_map.end()) {
            cam_result["status"] = "error";
            cam_result["success"] = false;
            cam_result["message"] = "Camera not found";
            results.push_back(cam_result);
            continue;
        }

        const auto& info = it->second;
        if (!info.present) {
            cam_result["status"] = "error";
            cam_result["success"] = false;
            cam_result["message"] = "Camera is not present";
            results.push_back(cam_result);
            continue;
        }

        if (info.mode != CameraManager::CamConfig::Mode::Calibration) {
            cam_result["status"] = "error";
            cam_result["success"] = false;
            cam_result["message"] = "Camera is not in calibration mode";
            results.push_back(cam_result);
            continue;
        }

        httplib::Client client("localhost", info.det_port);
        client.set_connection_timeout(5, 0);
        client.set_read_timeout(5, 0);

        nlohmann::json cam_req;
        cam_req["duration"] = duration;
        if (!base_prefix.empty()) {
            cam_req["filename_prefix"] = base_prefix;
            cam_req["prefix"] = base_prefix;
        }
        cam_req["mode"] = mode;
        cam_req["camera_id"] = camera_id;

        auto cam_res = client.Post("/api/record/start", cam_req.dump(), "application/json");
        bool cam_ok = false;
        if (cam_res && cam_res->status == 200) {
            cam_result["http_status"] = cam_res->status;
            try {
                auto cam_json = nlohmann::json::parse(cam_res->body);
                cam_result["camera_response"] = cam_json;
                std::string cam_status = cam_json.value("status", std::string(""));
                cam_ok = (cam_status == "ok");
                if (!cam_ok) {
                    cam_result["message"] = cam_json.value("message", std::string("Camera reported failure"));
                }

            } catch (...) {
                cam_result["message"] = "Invalid response from camera";
            }
        } else {
            if (cam_res) {
                cam_result["http_status"] = cam_res->status;
            }
            cam_result["message"] = "Failed to contact camera";
        }

        cam_result["status"] = cam_ok ? "ok" : "error";
        cam_result["success"] = cam_ok;
        if (cam_ok) {
            ++success_count;
        }
        results.push_back(cam_result);
    }

    resp["duration"] = duration;
    resp["mode"] = mode;
    resp["results"] = results;
    resp["successful"] = success_count;
    resp["total"] = results.size();

    if (success_count == results.size() && !results.empty()) {
        resp["status"] = "ok";
        resp["message"] = "Recording started on all cameras";
    } else if (success_count > 0) {
        resp["status"] = "partial";
        resp["message"] = "Recording started on some cameras";
    } else {
        resp["status"] = "error";
        resp["message"] = "Failed to start recording";
    }

    res.set_content(resp.dump(), "application/json");
});

g_server.Post("/api/record/stop", [](const httplib::Request& req, httplib::Response& res){
    nlohmann::json resp;
    nlohmann::json req_json;
    try {
        if (!req.body.empty()) {
            req_json = nlohmann::json::parse(req.body);
        }
    } catch (...) {
        resp["status"] = "error";
        resp["message"] = "Invalid JSON";
        res.status = 400;
        res.set_content(resp.dump(), "application/json");
        return;
    }

    const std::string mode = req_json.value("mode", std::string("mono"));
    std::vector<std::string> camera_ids;
    if (req_json.contains("camera_ids") && req_json["camera_ids"].is_array()) {
        for (const auto& item : req_json["camera_ids"]) {
            if (!item.is_string()) {
                resp["status"] = "error";
                resp["message"] = "camera_ids must contain strings";
                res.status = 400;
                res.set_content(resp.dump(), "application/json");
                return;
            }
            camera_ids.push_back(item.get<std::string>());
        }
    }

    if (camera_ids.empty()) {
        resp["status"] = "error";
        resp["message"] = "camera_ids is required";
        res.status = 400;
        res.set_content(resp.dump(), "application/json");
        return;
    }

    size_t expected_cameras = 0;
    if (mode == "mono") {
        expected_cameras = 1;
    } else if (mode == "stereo") {
        expected_cameras = 2;
    }

    if (expected_cameras == 0) {
        resp["status"] = "error";
        resp["message"] = "Unsupported recording mode";
        res.status = 400;
        res.set_content(resp.dump(), "application/json");
        return;
    }

    if (camera_ids.size() != expected_cameras) {
        resp["status"] = "error";
        resp["message"] = "Recording mode does not match number of cameras";
        res.status = 400;
        res.set_content(resp.dump(), "application/json");
        return;
    }

    auto configured = g_mgr.configuredCameras();
    std::unordered_map<std::string, CameraManager::ConfiguredInfo> configured_map;
    configured_map.reserve(configured.size());
    for (const auto& cfg : configured) {
        configured_map.emplace(cfg.id, cfg);
    }

    nlohmann::json results = nlohmann::json::array();
    size_t success_count = 0;

    for (const auto& camera_id : camera_ids) {
        nlohmann::json cam_result;
        cam_result["camera_id"] = camera_id;
        auto it = configured_map.find(camera_id);
        if (it == configured_map.end()) {
            cam_result["status"] = "error";
            cam_result["success"] = false;
            cam_result["message"] = "Camera not found";
            results.push_back(cam_result);
            continue;
        }

        const auto& info = it->second;
        if (!info.present) {
            cam_result["status"] = "error";
            cam_result["success"] = false;
            cam_result["message"] = "Camera is not present";
            results.push_back(cam_result);
            continue;
        }

        httplib::Client client("localhost", info.det_port);
        client.set_connection_timeout(5, 0);
        client.set_read_timeout(5, 0);

        nlohmann::json cam_req;
        cam_req["mode"] = mode;
        cam_req["camera_id"] = camera_id;

        auto cam_res = client.Post("/api/record/stop", cam_req.dump(), "application/json");
        bool cam_ok = false;
        if (cam_res && cam_res->status == 200) {
            cam_result["http_status"] = cam_res->status;
            try {
                auto cam_json = nlohmann::json::parse(cam_res->body);
                cam_result["camera_response"] = cam_json;
                std::string cam_status = cam_json.value("status", std::string(""));
                cam_ok = (cam_status == "ok");
                if (!cam_ok) {
                    cam_result["message"] = cam_json.value("message", std::string("Camera reported failure"));
                }
            } catch (...) {
                cam_result["message"] = "Invalid response from camera";
            }
        } else {
            if (cam_res) {
                cam_result["http_status"] = cam_res->status;
            }
            cam_result["message"] = "Failed to contact camera";
        }

        cam_result["status"] = cam_ok ? "ok" : "error";
        cam_result["success"] = cam_ok;
        if (cam_ok) {
            ++success_count;
        }
        results.push_back(cam_result);
    }

    resp["mode"] = mode;
    resp["results"] = results;
    resp["successful"] = success_count;
    resp["total"] = results.size();

    if (success_count == results.size() && !results.empty()) {
        resp["status"] = "ok";
        resp["message"] = "Recording stopped on all cameras";
    } else if (success_count > 0) {
        resp["status"] = "partial";
        resp["message"] = "Recording stopped on some cameras";
    } else {
        resp["status"] = "error";
        resp["message"] = "Failed to stop recording";
    }

    res.set_content(resp.dump(), "application/json");
});

// Automatic calibration API endpoints
// Automatic calibration API endpoints
g_server.Get("/api/calibration/params", [](const httplib::Request&, httplib::Response& res){
    auto cfg = readMainConfig();
    nlohmann::json resp;
    
    resp["board_cols"] = cfg.value("calib_board_cols", 10);
    resp["board_rows"] = cfg.value("calib_board_rows", 7); 
    resp["square_size"] = cfg.value("calib_square_size", 30.0f);
    resp["min_frames"] = cfg.value("calib_min_frames", 15);
    resp["max_frames"] = cfg.value("calib_max_frames", 50);
    resp["quality_threshold"] = cfg.value("calib_quality_threshold", 50.0f);
    resp["delete_videos"] = cfg.value("calib_delete_videos", true);
    resp["time_tolerance_ms"] = cfg.value("calib_time_tolerance_ms", 33);
    resp["enable_multithreading"] = cfg.value("calib_enable_multithreading", false);
    const float default_vertical = 0.05f;
    const float default_horizontal = 0.35f;
    const float stereo_center_vertical = cfg.contains("calib_stereo_pose_max_center_diff_vertical")
        ? cfg.value("calib_stereo_pose_max_center_diff_vertical", default_vertical)
        : cfg.value("calib_stereo_pose_max_center_diff", default_vertical);
    const float stereo_center_horizontal = cfg.value(
        "calib_stereo_pose_max_center_diff_horizontal", default_horizontal);
    resp["stereo_pose_max_center_diff_vertical"] = stereo_center_vertical;
    resp["stereo_pose_max_center_diff_horizontal"] = stereo_center_horizontal;
    const float default_span_ratio = cfg.contains("calib_stereo_pose_max_normalized_span_ratio")
        ? cfg.value("calib_stereo_pose_max_normalized_span_ratio", 0.3f)
        : cfg.value("calib_stereo_pose_max_scale_diff", 0.3f);
    resp["stereo_pose_max_normalized_span_ratio"] = default_span_ratio;
    resp["stereo_pose_max_scale_diff"] = cfg.value("calib_stereo_pose_max_scale_diff", 0.1f);
    resp["stereo_pose_max_tilt_diff"] = cfg.value("calib_stereo_pose_max_tilt_diff", 10.0f);

    res.set_content(resp.dump(), "application/json");
});

g_server.Post("/api/calibration/params", [](const httplib::Request& req, httplib::Response& res){
    nlohmann::json resp;
    
    try {
        auto j = nlohmann::json::parse(req.body);
        auto cfg = readMainConfig();
        
        cfg["calib_board_cols"] = j.value("board_cols", 10);
        cfg["calib_board_rows"] = j.value("board_rows", 7);
        cfg["calib_square_size"] = j.value("square_size", 30.0f);
        cfg["calib_min_frames"] = j.value("min_frames", 15);
        cfg["calib_max_frames"] = j.value("max_frames", 50);
        cfg["calib_quality_threshold"] = j.value("quality_threshold", 50.0f);
        cfg["calib_delete_videos"] = j.value("delete_videos", true);
        cfg["calib_time_tolerance_ms"] = j.value("time_tolerance_ms", 33);
        cfg["calib_enable_multithreading"] = j.value("enable_multithreading", false);
        const float center_vertical = j.contains("stereo_pose_max_center_diff_vertical")
            ? j.value("stereo_pose_max_center_diff_vertical", 0.05f)
            : j.value("stereo_pose_max_center_diff", 0.05f);
        const float center_horizontal = j.value("stereo_pose_max_center_diff_horizontal", 0.35f);
        cfg["calib_stereo_pose_max_center_diff_vertical"] = center_vertical;
        cfg["calib_stereo_pose_max_center_diff_horizontal"] = center_horizontal;
        const float normalized_span_ratio = j.contains("stereo_pose_max_normalized_span_ratio")
            ? j.value("stereo_pose_max_normalized_span_ratio", 0.3f)
            : j.value("stereo_pose_max_scale_diff", 0.3f);
        cfg["calib_stereo_pose_max_normalized_span_ratio"] = normalized_span_ratio;
        cfg["calib_stereo_pose_max_scale_diff"] = j.value("stereo_pose_max_scale_diff", normalized_span_ratio);
        cfg["calib_stereo_pose_max_tilt_diff"] = j.value("stereo_pose_max_tilt_diff", 10.0f);

        if (!writeMainConfig(cfg)) {
            resp["status"] = "error";
            resp["error"] = "Failed to save configuration";
            res.status = 500;
        } else {
            resp["status"] = "ok";
            resp["message"] = "Calibration parameters updated";
        }
        
    } catch (const std::exception& e) {
        resp["status"] = "error";
        resp["error"] = e.what();
        res.status = 400;
    }
    
    res.set_content(resp.dump(), "application/json");
});



g_server.Get("/api/calibration/recordings", [](const httplib::Request&, httplib::Response& res){
    nlohmann::json resp;

    CalibrationWatcher* watcher = ensureCalibrationWatcher();
    auto inventory = watcher->getRecordingInventory();

    nlohmann::json mono = nlohmann::json::array();
    for (const auto& camera : inventory.mono_cameras) {
        mono.push_back(camera);
    }

    nlohmann::json stereo = nlohmann::json::array();
    for (const auto& pair : inventory.stereo_pairs) {
        nlohmann::json pair_json = nlohmann::json::array();
        for (const auto& camera : pair) {
            pair_json.push_back(camera);
        }
        stereo.push_back(pair_json);
    }

    nlohmann::json videos = nlohmann::json::array();
    for (const auto& video : inventory.videos) {
        nlohmann::json video_json;
        video_json["camera_id"] = video.camera_id;
        if (!video.mode.empty()) {
            video_json["mode"] = video.mode;
        }
        if (!video.capture_type.empty()) {
            video_json["capture_type"] = video.capture_type;
        }
        if (!video.capture_group.empty()) {
            video_json["capture_group"] = video.capture_group;
        }
        video_json["path"] = video.path.string();
        video_json["file_size"] = video.file_size;
        video_json["last_modified"] = static_cast<int64_t>(video.last_modified);
        videos.push_back(std::move(video_json));
    }

    resp["mono_cameras"] = std::move(mono);
    resp["stereo_pairs"] = std::move(stereo);
    resp["videos"] = std::move(videos);

    res.set_content(resp.dump(), "application/json");
});



g_server.Post("/api/calibration/start-auto", [](const httplib::Request&, httplib::Response& res){
    nlohmann::json resp;

    try {
        auto cfg = readMainConfig();
        CalibrationWatcher* watcher = ensureCalibrationWatcher();

        if (watcher->isProcessing()) {
            resp["status"] = "error";
            resp["error"] = "Calibration already in progress";
            res.status = 400;
        } else {
            CalibrationParams params;
            params.board_cols = cfg.value("calib_board_cols", 10);
            params.board_rows = cfg.value("calib_board_rows", 7);
            params.square_size = cfg.value("calib_square_size", 30.0f);
            params.min_frames = cfg.value("calib_min_frames", 15);
            params.max_frames = cfg.value("calib_max_frames", 50);
            params.quality_threshold = cfg.value("calib_quality_threshold", 50.0f);
            params.delete_videos = cfg.value("calib_delete_videos", true);
            params.time_tolerance_ms = cfg.value("calib_time_tolerance_ms", 33);
            params.enable_multithreading = cfg.value("calib_enable_multithreading", false);
            params.stereo_pose_max_center_diff_vertical = cfg.contains("calib_stereo_pose_max_center_diff_vertical")
                ? cfg.value("calib_stereo_pose_max_center_diff_vertical", 0.05f)
                : cfg.value("calib_stereo_pose_max_center_diff", 0.05f);
            params.stereo_pose_max_center_diff_horizontal =
                cfg.value("calib_stereo_pose_max_center_diff_horizontal", 0.35f);
            params.stereo_pose_max_normalized_span_ratio = cfg.contains("calib_stereo_pose_max_normalized_span_ratio")
                ? cfg.value("calib_stereo_pose_max_normalized_span_ratio", 0.3f)
                : cfg.value("calib_stereo_pose_max_scale_diff", 0.3f);
            params.stereo_pose_max_tilt_diff = cfg.value("calib_stereo_pose_max_tilt_diff", 10.0f);

            printf("Starting calibration with params: board=%dx%d, square=%.1fmm\n",
                   params.board_cols, params.board_rows, params.square_size);

            // Создаем папки
            std::filesystem::create_directories("/tmp/rec");
            std::filesystem::create_directories("/tmp/calibration");

            if (watcher->startCalibration(params)) {
                resp["status"] = "ok";
                resp["message"] = "Calibration started successfully";
                resp["parameters"] = {
                    {"board_cols", params.board_cols},
                    {"board_rows", params.board_rows},
                    {"square_size", params.square_size}
                };
            } else {
                resp["status"] = "error";
                resp["error"] = "Failed to start calibration";
                res.status = 500;
            }
        }
        
    } catch (const std::exception& e) {
        resp["status"] = "error";
        resp["error"] = e.what();
        res.status = 500;
    }
    
    res.set_content(resp.dump(), "application/json");
});

g_server.Post("/api/calibration/stop-auto", [](const httplib::Request&, httplib::Response& res){
    nlohmann::json resp;
    
    if (g_calib_watcher && g_calib_watcher->isProcessing()) {
        g_calib_watcher->stopCalibration();
        resp["status"] = "ok";
        resp["message"] = "Calibration stopped";
    } else {
        resp["status"] = "ok"; 
        resp["message"] = "No calibration was running";
    }
    
    res.set_content(resp.dump(), "application/json");
});

g_server.Get("/api/calibration/status-auto", [](const httplib::Request&, httplib::Response& res){
    nlohmann::json resp;

    if (g_calib_watcher) {
        g_global_tracker.setCalibrationWatcher(g_calib_watcher.get());
        g_global_tracker.checkAndUpdateCalibration();
        resp["processing"] = g_calib_watcher->isProcessing();
        resp["progress"] = g_calib_watcher->getProgress();
        resp["status_message"] = g_calib_watcher->getStatus();

        // Получаем результаты
        auto mono_results = g_calib_watcher->getMonoResults();
        auto stereo_results = g_calib_watcher->getStereoResults();

        resp["mono_calibrations"] = mono_results.size();
        resp["stereo_calibrations"] = stereo_results.size();

        nlohmann::json mono_summary = nlohmann::json::array();
        for (const auto& result : mono_results) {
            nlohmann::json summary;
            summary["camera_id"] = result.camera_id;
            summary["success"] = result.success;
            summary["reprojection_error"] = result.reprojection_error;
            summary["frames_used"] = result.frames_used;
            summary["calibration_time"] = result.calibration_time;
            mono_summary.push_back(summary);
        }
        resp["mono_results"] = mono_summary;
        
        nlohmann::json stereo_summary = nlohmann::json::array();
        for (const auto& result : stereo_results) {
            nlohmann::json summary;
            summary["camera_pair"] = result.camera_pair;
            summary["success"] = result.success;
            summary["reprojection_error"] = result.reprojection_error;
            summary["calibration_time"] = result.calibration_time;
            stereo_summary.push_back(summary);
        }
        resp["stereo_results"] = stereo_summary;
        
    } else {
        resp["processing"] = false;
        resp["progress"] = 0.0f;
        resp["status_message"] = "Calibration not initialized";
        resp["mono_calibrations"] = 0;
        resp["stereo_calibrations"] = 0;
        resp["mono_results"] = nlohmann::json::array();
        resp["stereo_results"] = nlohmann::json::array();
    }
    
    res.set_content(resp.dump(), "application/json");
});

g_server.Post("/api/calibration/copy-results", [](const httplib::Request&, httplib::Response& res){
    nlohmann::json resp;
    
    try {
        std::filesystem::path src_dir = "/tmp/calibration/results";
        std::filesystem::path dst_dir = "./calibration/results";
        
        std::filesystem::create_directories(dst_dir);
        int copied = 0;
        
        if (std::filesystem::exists(src_dir)) {
            for (const auto& entry : std::filesystem::directory_iterator(src_dir)) {
                if (entry.is_regular_file()) {
                    std::filesystem::path dst_file = dst_dir / entry.path().filename();
                    std::filesystem::copy_file(entry.path(), dst_file, 
                        std::filesystem::copy_options::overwrite_existing);
                    copied++;
                }
            }
            // Очищаем временные файлы после копирования
            std::filesystem::remove_all("/tmp/calibration");
            std::filesystem::remove_all("/tmp/rec");
            printf("Cleaned temporary calibration files\n");

            resp["status"] = "ok";
            resp["message"] = "Copied " + std::to_string(copied) + " calibration files";
        } else {
            resp["status"] = "error";
            resp["error"] = "No calibration results found";
        }
        
    } catch (const std::exception& e) {
        resp["status"] = "error";
        resp["error"] = e.what();
    }
    
    res.set_content(resp.dump(), "application/json");
});

g_server.Get("/api/tracking/global", [](const httplib::Request&, httplib::Response& res){
    nlohmann::json resp = serializeGlobalObjects(g_global_tracker.getActiveObjects());
    res.set_content(resp.dump(), "application/json");
});

g_server.Post("/api/tracking/mode", [](const httplib::Request& req, httplib::Response& res){
    nlohmann::json resp;
    try {
        auto j = nlohmann::json::parse(req.body);
        g_use_global_tracking = j.value("global", false);
        
        if (g_use_global_tracking) {
            g_scheme_manager.initialize(g_config_path.string());
            ensureCalibrationWatcher();
            g_global_tracker.initialize();
        }
        
        resp["status"] = "ok";
        resp["global_tracking"] = g_use_global_tracking;
        
    } catch (const std::exception& e) {
        resp["status"] = "error";
        resp["error"] = e.what();
    }
    res.set_content(resp.dump(), "application/json");
});


g_server.Post("/api/tracking/grayscale-mode", [](const httplib::Request& req, httplib::Response& res){
    nlohmann::json resp;
    try {
        auto j = nlohmann::json::parse(req.body);
        bool new_grayscale_mode = j.value("grayscale", g_use_grayscale_tracking);

        g_use_grayscale_tracking = new_grayscale_mode;

        // Update config file to persist setting
        auto config = readMainConfig();
        config["use_grayscale_tracking"] = g_use_grayscale_tracking;
        if (writeMainConfig(config)) {
            resp["status"] = "ok";
            resp["grayscale_tracking"] = g_use_grayscale_tracking;
        } else {
            resp["status"] = "error";
            resp["error"] = "Failed to save configuration";
        }

    } catch (const std::exception& e) {
        resp["status"] = "error";
        resp["error"] = e.what();
    }
    res.set_content(resp.dump(), "application/json");
});

g_server.Get("/api/tracking/grayscale-mode", [](const httplib::Request&, httplib::Response& res){
    nlohmann::json resp;
    resp["status"] = "ok";
    resp["grayscale_tracking"] = g_use_grayscale_tracking;
    res.set_content(resp.dump(), "application/json");
});



g_server.Get("/api/detections/update", [](const httplib::Request& req, httplib::Response& res){
    if (!g_use_global_tracking) {
        res.set_content("{\"error\":\"Global tracking disabled\"}", "application/json");
        return;
    }
    
    // Ограничиваем частоту обращений
    static auto last_update = std::chrono::steady_clock::now();
    static nlohmann::json cached_response;
    auto now = std::chrono::steady_clock::now();
    
    if (now - last_update < std::chrono::milliseconds(2000)) { // 2 секунды вместо постоянных запросов
        res.set_content(cached_response.dump(), "application/json");
        return;
    }
    last_update = now;
    
    // Собираем детекции только с активных камер, но с таймаутом
    auto cams = g_mgr.configuredCameras();
    std::vector<std::future<void>> futures;
    
    for (const auto& cam : cams) {
        if (cam.mode == CameraManager::CamConfig::Mode::Detect && cam.det_running) {
            futures.emplace_back(std::async(std::launch::async,
                                            [cam_id = cam.id, det_port = cam.det_port]() {
                                                fetchAndUpdateDetections(cam_id, det_port);
                                            }));
        }
    }
    
    // Ждем завершения всех запросов с таймаутом
    for (auto& future : futures) {
        if (future.wait_for(std::chrono::milliseconds(600)) == std::future_status::timeout) {
            // Таймаут - продолжаем без этой камеры
        }
    }
    
    cached_response = serializeGlobalObjects(g_global_tracker.getActiveObjects());
    res.set_content(cached_response.dump(), "application/json");
});


  std::thread http_thr([&] { g_server.listen("0.0.0.0", port); });
  std::cout << "CameraManager running. Press Ctrl+C to exit." << std::endl;
  while (g_server.is_running()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  if (http_thr.joinable())
    http_thr.join();
  g_stereo_running = false;
  if (stereo_thread.joinable())
    stereo_thread.join();
  return 0;
}
