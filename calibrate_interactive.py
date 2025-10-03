#!/usr/bin/env python3
"""
Interactive Camera Calibration Tool - STEREO MODE
Usage: python3 calibrate_interactive.py
"""

import cv2
import numpy as np
import json
import time
import argparse
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from enum import Enum
import os
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ============================================================================
# Конфигурация
# ============================================================================

@dataclass
class CalibConfig:
    pattern_cols: int = 16
    pattern_rows: int = 8
    square_size: float = 22.0
    min_frames: int = 100
    max_frames: int = 200
    quality_threshold: float = 50.0
    
    # Стерео критерии
    max_center_diff: float = 0.05
    max_center_diff_horizontal: float = 0.20
    max_center_diff_vertical: float = 0.20
    max_normalized_span_ratio: float = 0.50
    max_tilt_diff: float = 5.0
    
    # Качество позы
    min_coverage: float = 0.25
    max_coverage: float = 0.75
    min_distance_between_frames: float = 0.08
    
    @classmethod
    def from_json(cls, config_data: dict):
        return cls(
            pattern_cols=config_data.get('calib_board_cols', 16),
            pattern_rows=config_data.get('calib_board_rows', 8),
            square_size=config_data.get('calib_square_size', 22.0),
            min_frames=config_data.get('calib_min_frames', 100),
            max_frames=config_data.get('calib_max_frames', 200),
            quality_threshold=config_data.get('calib_quality_threshold', 50.0),
            max_center_diff=config_data.get('calib_stereo_pose_max_center_diff', 0.05),
            max_center_diff_horizontal=config_data.get('calib_stereo_pose_max_center_diff_horizontal', 0.20),
            max_center_diff_vertical=config_data.get('calib_stereo_pose_max_center_diff_vertical', 0.20),
            max_normalized_span_ratio=config_data.get('calib_stereo_pose_max_normalized_span_ratio', 0.50),
            max_tilt_diff=config_data.get('calib_stereo_pose_max_tilt_diff', 5.0)
        )

@dataclass
class CameraInfo:
    id: str
    device: str
    width: int
    height: int
    fps: int
    role: str
    mode: str
    
    @classmethod
    def from_json(cls, cam_data: dict):
        preferred = cam_data.get('preferred', {})
        return cls(
            id=cam_data.get('id', ''),
            device=cam_data.get('device', ''),
            width=preferred.get('w', 800),
            height=preferred.get('h', 600),
            fps=preferred.get('fps', 30),
            role=cam_data.get('role', 'unknown'),
            mode=cam_data.get('mode', 'detect')
        )

class HintType(Enum):
    SEARCHING = "Ищу шахматную доску..."
    TOO_CLOSE = "Отодвиньте доску подальше"
    TOO_FAR = "Приблизьте доску к камере"
    TOO_FLAT = "Наклоните доску (больше угол)"
    TOO_TILTED = "Доска слишком наклонена"
    MOVE_LEFT = "Сдвиньте доску влево"
    MOVE_RIGHT = "Сдвиньте доску вправо"
    MOVE_UP = "Сдвиньте доску вверх"
    MOVE_DOWN = "Сдвиньте доску вниз"
    HOLD_STILL = "Держите неподвижно... ({:.1f}с)"
    CAPTURED = "Кадр захвачен! ({}/{})"
    TOO_SIMILAR = "Переместите доску в другую позицию"
    COMPLETE = "Калибровка завершена!"
    
    # Стерео подсказки
    STEREO_NOT_BOTH = "Доска видна только на одной камере"
    STEREO_TILT_DIFF = "Камеры видят разный наклон доски"
    STEREO_CENTER_DIFF = "Доска в разных местах на камерах"
    STEREO_GOOD = "Обе камеры готовы!"

# ============================================================================
# Загрузка конфига
# ============================================================================

def load_config(config_path: str) -> Tuple[CalibConfig, List[CameraInfo]]:
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    calib_config = CalibConfig.from_json(data)
    cameras = []
    for cam_data in data.get('cameras', []):
        if cam_data.get('mode') == 'calibration' and cam_data.get('device'):
            cameras.append(CameraInfo.from_json(cam_data))
    
    return calib_config, cameras

# ============================================================================
# Stereo Calibration Engine
# ============================================================================

class StereoCalibrationEngine:
    def __init__(self, config: CalibConfig, cam1_info: CameraInfo, cam2_info: CameraInfo):
        self.config = config
        self.cam1_info = cam1_info
        self.cam2_info = cam2_info
        
        self.cap1: Optional[cv2.VideoCapture] = None
        self.cap2: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.is_calibrating = False
        
        # Накопленные данные (стерео)
        self.object_points = []
        self.image_points_1 = []
        self.image_points_2 = []
        self.captured_frames = []
        self.hold_start_time = 0
        self.hold_duration = 2.0
        
        # Результаты
        self.camera_matrix_1 = None
        self.dist_coeffs_1 = None
        self.camera_matrix_2 = None
        self.dist_coeffs_2 = None
        self.R = None  # Rotation matrix
        self.T = None  # Translation vector
        self.E = None  # Essential matrix
        self.F = None  # Fundamental matrix
        self.stereo_error = None
        
        # Объектные точки
        self.objp = np.zeros((config.pattern_rows * config.pattern_cols, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:config.pattern_cols, 0:config.pattern_rows].T.reshape(-1, 2)
        self.objp *= config.square_size
        
        # Для видео стрима
        self.latest_frame1 = None
        self.latest_frame2 = None
        self.frame_lock = threading.Lock()
    
    def start_cameras(self) -> bool:
        """Открыть обе камеры"""
        self.cap1 = self._open_camera(self.cam1_info)
        self.cap2 = self._open_camera(self.cam2_info)
        
        if not (self.cap1 and self.cap1.isOpened() and self.cap2 and self.cap2.isOpened()):
            return False
        
        self.is_running = True
        return True
    
    def _open_camera(self, cam_info: CameraInfo) -> Optional[cv2.VideoCapture]:
        """Открыть одну камеру"""
        cap = cv2.VideoCapture(cam_info.device)
        
        if not cap.isOpened() and cam_info.device.startswith('/dev/video'):
            try:
                idx = int(cam_info.device.replace('/dev/video', ''))
                cap = cv2.VideoCapture(idx)
            except:
                pass
        
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_info.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_info.height)
            cap.set(cv2.CAP_PROP_FPS, cam_info.fps)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        return cap
    
    def stop_cameras(self):
        """Закрыть камеры"""
        if self.cap1:
            self.cap1.release()
        if self.cap2:
            self.cap2.release()
        self.is_running = False
    
    def start_calibration(self):
        """Начать сбор"""
        self.is_calibrating = True
        self.object_points.clear()
        self.image_points_1.clear()
        self.image_points_2.clear()
        self.captured_frames.clear()
        self.hold_start_time = 0
    
    def analyze_stereo_frame(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[HintType, dict]:
        """Анализ стерео-пары"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Поиск углов на обеих камерах
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret1, corners1 = cv2.findChessboardCorners(
            gray1, (self.config.pattern_cols, self.config.pattern_rows), flags
        )
        ret2, corners2 = cv2.findChessboardCorners(
            gray2, (self.config.pattern_cols, self.config.pattern_rows), flags
        )
        
        # Обе камеры должны видеть доску
        if not (ret1 and ret2):
            self.hold_start_time = 0
            if ret1 and not ret2:
                return False, None, None, HintType.STEREO_NOT_BOTH, {"cam": 2}
            if ret2 and not ret1:
                return False, None, None, HintType.STEREO_NOT_BOTH, {"cam": 1}
            return False, None, None, HintType.SEARCHING, {}
        
        # Уточнение
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
        corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
        
        # Проверка стерео-согласованности
        hint, data = self._check_stereo_consistency(corners1, corners2, frame1.shape, frame2.shape)
        
        return True, corners1, corners2, hint, data
    
    def _check_stereo_consistency(self, corners1: np.ndarray, corners2: np.ndarray, 
                                   shape1: Tuple, shape2: Tuple) -> Tuple[HintType, dict]:
        """Проверка стерео-согласованности"""
        h1, w1 = shape1[:2]
        h2, w2 = shape2[:2]
        
        c1 = corners1.reshape(-1, 2)
        c2 = corners2.reshape(-1, 2)
        
        # 1. Центры
        center1 = c1.mean(axis=0)
        center2 = c2.mean(axis=0)
        
        center1_norm = center1 / [w1, h1]
        center2_norm = center2 / [w2, h2]
        
        center_diff = np.abs(center1_norm - center2_norm)
        
        if center_diff[0] > self.config.max_center_diff_horizontal:
            self.hold_start_time = 0
            return HintType.STEREO_CENTER_DIFF, {"axis": "horizontal", "diff": center_diff[0]}
        
        if center_diff[1] > self.config.max_center_diff_vertical:
            self.hold_start_time = 0
            return HintType.STEREO_CENTER_DIFF, {"axis": "vertical", "diff": center_diff[1]}
        
        # 2. Наклон (tilt)
        tilt1 = self._compute_tilt(c1)
        tilt2 = self._compute_tilt(c2)
        tilt_diff = abs(tilt1 - tilt2)
        
        if tilt_diff > self.config.max_tilt_diff:
            self.hold_start_time = 0
            return HintType.STEREO_TILT_DIFF, {"diff": tilt_diff}
        
        # 3. Проверка похожести с предыдущими
        if self.image_points_1:
            last_c1 = self.image_points_1[-1].reshape(-1, 2)
            diff = np.mean(np.linalg.norm(c1 - last_c1, axis=1))
            normalized_diff = diff / np.sqrt(w1**2 + h1**2)
            
            if normalized_diff < self.config.min_distance_between_frames:
                self.hold_start_time = 0
                return HintType.TOO_SIMILAR, {"diff": normalized_diff}
        
        # 4. Удержание
        current_time = time.time()
        if self.hold_start_time == 0:
            self.hold_start_time = current_time
        
        elapsed = current_time - self.hold_start_time
        if elapsed < self.hold_duration:
            return HintType.HOLD_STILL, {"remaining": self.hold_duration - elapsed}
        
        return HintType.CAPTURED, {}
    
    def _compute_tilt(self, corners: np.ndarray) -> float:
        """Вычислить угол наклона доски"""
        top_left = corners[0]
        top_right = corners[self.config.pattern_cols - 1]
        vec = top_right - top_left
        angle = np.degrees(np.arctan2(vec[1], vec[0]))
        return angle
    
    def capture_stereo_frame(self, corners1: np.ndarray, corners2: np.ndarray):
        """Сохранить стерео-пару"""
        self.object_points.append(self.objp)
        self.image_points_1.append(corners1)
        self.image_points_2.append(corners2)
        self.captured_frames.append(time.time())
        self.hold_start_time = 0
    
    def calibrate_stereo(self) -> bool:
        """Стереокалибровка"""
        if len(self.image_points_1) < self.config.min_frames:
            return False
        
        img_shape = (self.cam1_info.width, self.cam1_info.height)
        
        # Стереокалибровка
        flags = cv2.CALIB_FIX_INTRINSIC
        
        ret, self.camera_matrix_1, self.dist_coeffs_1, self.camera_matrix_2, self.dist_coeffs_2, \
        self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            self.object_points,
            self.image_points_1,
            self.image_points_2,
            None, None, None, None,
            img_shape,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
            flags=0
        )
        
        if not ret:
            return False
        
        # Вычисление ошибки
        total_error = 0
        for i in range(len(self.object_points)):
            imgpoints1, _ = cv2.projectPoints(
                self.object_points[i],
                cv2.Rodrigues(self.R)[0] if i == 0 else np.zeros((3, 1)),
                self.T if i == 0 else np.zeros((3, 1)),
                self.camera_matrix_1,
                self.dist_coeffs_1
            )
            error1 = cv2.norm(self.image_points_1[i], imgpoints1, cv2.NORM_L2) / len(imgpoints1)
            
            imgpoints2, _ = cv2.projectPoints(
                self.object_points[i],
                np.zeros((3, 1)),
                np.zeros((3, 1)),
                self.camera_matrix_2,
                self.dist_coeffs_2
            )
            error2 = cv2.norm(self.image_points_2[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            
            total_error += (error1 + error2) / 2
        
        self.stereo_error = total_error / len(self.object_points)
        return True
    
    def save_results(self, output_path: str):
        """Сохранить стерео-результаты"""
        data = {
            "camera_1": {
                "id": self.cam1_info.id,
                "device": self.cam1_info.device,
                "camera_matrix": self.camera_matrix_1.tolist(),
                "distortion_coeffs": self.dist_coeffs_1.tolist()
            },
            "camera_2": {
                "id": self.cam2_info.id,
                "device": self.cam2_info.device,
                "camera_matrix": self.camera_matrix_2.tolist(),
                "distortion_coeffs": self.dist_coeffs_2.tolist()
            },
            "stereo": {
                "rotation_matrix": self.R.tolist(),
                "translation_vector": self.T.tolist(),
                "essential_matrix": self.E.tolist(),
                "fundamental_matrix": self.F.tolist()
            },
            "calibration_error": float(self.stereo_error),
            "frames_used": len(self.image_points_1),
            "pattern": {
                "cols": self.config.pattern_cols,
                "rows": self.config.pattern_rows,
                "square_size": self.config.square_size
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

# ============================================================================
# Web API
# ============================================================================

config_path = "config.json"
calib_config: Optional[CalibConfig] = None
available_cameras: List[CameraInfo] = []
stereo_engine: Optional[StereoCalibrationEngine] = None

def init_config():
    global calib_config, available_cameras
    if os.path.exists(config_path):
        calib_config, available_cameras = load_config(config_path)
        print(f"Loaded {len(available_cameras)} calibration cameras")

@app.route('/')
def index():
    return render_template('calibrate_stereo.html')

@app.route('/api/cameras')
def get_cameras():
    return jsonify([{
        'id': cam.id,
        'device': cam.device,
        'width': cam.width,
        'height': cam.height,
        'role': cam.role
    } for cam in available_cameras])

@app.route('/api/config')
def get_config():
    return jsonify({
        'pattern_cols': calib_config.pattern_cols,
        'pattern_rows': calib_config.pattern_rows,
        'square_size': calib_config.square_size,
        'min_frames': calib_config.min_frames,
        'max_frames': calib_config.max_frames
    })

@socketio.on('start_stereo')
def handle_start_stereo(data):
    global stereo_engine
    
    cam1_id = data.get('camera1_id')
    cam2_id = data.get('camera2_id')
    
    cam1 = next((c for c in available_cameras if c.id == cam1_id), None)
    cam2 = next((c for c in available_cameras if c.id == cam2_id), None)
    
    if not (cam1 and cam2):
        emit('stereo_started', {'success': False, 'error': 'Cameras not found'})
        return
    
    stereo_engine = StereoCalibrationEngine(calib_config, cam1, cam2)
    if stereo_engine.start_cameras():
        emit('stereo_started', {'success': True})
    else:
        emit('stereo_started', {'success': False, 'error': 'Failed to open cameras'})

@socketio.on('stop_stereo')
def handle_stop_stereo():
    if stereo_engine:
        stereo_engine.stop_cameras()
    emit('stereo_stopped')

@socketio.on('start_calibration')
def handle_start_calibration():
    if stereo_engine:
        stereo_engine.start_calibration()
        emit('calibration_started')

@socketio.on('run_calibration')
def handle_run_calibration():
    if stereo_engine and stereo_engine.calibrate_stereo():
        emit('calibration_complete', {
            'error': stereo_engine.stereo_error,
            'frames': len(stereo_engine.image_points_1)
        })
    else:
        emit('calibration_failed', {'error': 'Not enough frames'})

@socketio.on('save_results')
def handle_save_results(data):
    if stereo_engine:
        path = data.get('path', f'stereo_calibration_{stereo_engine.cam1_info.id}_{stereo_engine.cam2_info.id}.json')
        stereo_engine.save_results(path)
        emit('results_saved', {'path': path})

def generate_stereo_frames():
    """Генератор стерео-видео"""
    global stereo_engine
    
    while stereo_engine and stereo_engine.is_running:
        ret1, frame1 = stereo_engine.cap1.read()
        ret2, frame2 = stereo_engine.cap2.read()
        
        if not (ret1 and ret2):
            break
        
        hint_type = HintType.SEARCHING
        hint_data = {}
        
        if stereo_engine.is_calibrating:
            found, corners1, corners2, hint_type, hint_data = stereo_engine.analyze_stereo_frame(frame1, frame2)
            
            if found:
                cv2.drawChessboardCorners(frame1, (stereo_engine.config.pattern_cols, stereo_engine.config.pattern_rows), corners1, True)
                cv2.drawChessboardCorners(frame2, (stereo_engine.config.pattern_cols, stereo_engine.config.pattern_rows), corners2, True)
                
                if hint_type == HintType.CAPTURED:
                    stereo_engine.capture_stereo_frame(corners1, corners2)
                    hint_data['current'] = len(stereo_engine.image_points_1)
                    hint_data['total'] = stereo_engine.config.max_frames
                    
                    if len(stereo_engine.image_points_1) >= stereo_engine.config.max_frames:
                        hint_type = HintType.COMPLETE
                        stereo_engine.is_calibrating = False
        
        # Подсказка
        hint_text = hint_type.value
        if '{}' in hint_text or '{:.1f}' in hint_text:
            try:
                hint_text = hint_text.format(*hint_data.values())
            except:
                pass
        
        # Комбинированный кадр (side-by-side)
        combined = np.hstack([frame1, frame2])
        
        color = (0, 255, 0) if hint_type == HintType.CAPTURED else (0, 165, 255)
        cv2.putText(combined, hint_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if stereo_engine.is_calibrating:
            progress = f"{len(stereo_engine.image_points_1)}/{stereo_engine.config.max_frames}"
            cv2.putText(combined, progress, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Разделитель
        h = combined.shape[0]
        cv2.line(combined, (frame1.shape[1], 0), (frame1.shape[1], h), (255, 255, 255), 2)
        
        _, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_stereo_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    
    config_path = args.config
    init_config()
    
    if len(available_cameras) < 2:
        print("ERROR: Need at least 2 cameras with mode='calibration'")
        exit(1)
    
    print(f"\nStarting STEREO calibration server on http://{args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=False)
