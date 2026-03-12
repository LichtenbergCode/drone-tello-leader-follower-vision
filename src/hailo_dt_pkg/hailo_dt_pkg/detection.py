import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import setproctitle
import socket
import hailo
import cv2
import time
import threading
import cvzone

import rclpy
from rclpy.executors import MultiThreadedExecutor

from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerApp, app_callback_class
from hailo_apps.hailo_app_python.core.common.installation_utils import detect_hailo_arch
from hailo_apps.hailo_app_python.core.common.core import get_default_parser
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_dt_pkg.ros_comm import *

class MyGStreamerDetectionApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        if parser is None:
            parser = get_default_parser()

        parser.add_argument("--labels-json", default=None, help="Path to custom labels JSON file")
        super().__init__(parser, user_data)

        # Hailo settings
        self.batch_size = 3
        self.nms_score_threshold = 0.6
        self.nms_iou_threshold = 0.1

        # Detectar arquitectura de Hailo
        if self.options_menu.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError("No se pudo detectar arquitectura Hailo automáticamente.")
            self.arch = detected_arch
            print(f"✔ Arquitectura detectada: {self.arch}")
        else:
            self.arch = self.options_menu.arch
        
        # To create the pipeline
        ############################
        # Resources Paths
        self.hef_path = "/home/ros/hailo-apps/drone_resources/drone_det.hef"
        self.post_process_so = "/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so"
        self.post_function_name = "filter_letterbox"
        self.labels_json = "/home/ros/hailo-apps/drone_resources/drone_detection.json"

        # Fuente de video (stream de Tello en puerto 11111)
        self.video_source = ('udpsrc port=11111 caps="video/x-h264, stream-format=(string)byte-stream, width=(int)640, height=(int)640, framerate=(fraction)24/1, skip-fist-bytes=2" !'\
                            'queue !'
                            'decodebin !'
                            'videoconvert !')# 
        self.video_height = 640
        self.video_width = 640
        self.frame_rate = 30
        self.sync = "false"
        self.video_sink = "autovideosink"
        self.show_fps = True

        self.thresholds_str = (
            f"nms-score-threshold={self.nms_score_threshold} "
            f"nms-iou-threshold={self.nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )
        #############################
        self.app_callback = app_callback
        setproctitle.setproctitle("Hailo Detection App")
        self.create_pipeline()

    def get_pipeline_string(self):
        pipeline = (
            'udpsrc port=11111 caps="video/x-h264, stream-format=(string)byte-stream, width=(int)640, height=(int)640, framerate=(fraction)24/1, skip-fist-bytes=2" !'
            'queue !' 
            'decodebin !' 
            'videoconvert !'
            'queue name=source_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! '
            'videoscale name=source_videoscale n-threads=2 ! '
            'queue name=source_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! '
            'videoconvert n-threads=3 name=source_convert qos=false ! '
            'video/x-raw, pixel-aspect-ratio=1/1, format=RGB, width=640, height=640 ! '
            'videorate name=source_videorate ! '
            'capsfilter name=source_fps_caps caps="video/x-raw, framerate=30/1"  ! '
            'queue name=inference_wrapper_input_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! '
            'hailocropper name=inference_wrapper_crop so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true hailoaggregator name=inference_wrapper_agg inference_wrapper_crop. ! '
            'queue name=inference_wrapper_bypass_q leaky=no max-size-buffers=20 max-size-bytes=0 max-size-time=0  ! '
            'inference_wrapper_agg.sink_0 inference_wrapper_crop. ! '
            'queue name=inference_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! '
            'videoscale name=inference_videoscale n-threads=2 qos=false ! '
            'queue name=inference_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! '
            'video/x-raw, pixel-aspect-ratio=1/1 ! '
            'videoconvert name=inference_videoconvert n-threads=2 ! '
            'queue name=inference_hailonet_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! '
            f'hailonet name=inference_hailonet hef-path={self.hef_path} batch-size={self.batch_size}  vdevice-group-id=1 nms-score-threshold={self.nms_score_threshold} nms-iou-threshold={self.nms_iou_threshold} output-format-type=HAILO_FORMAT_TYPE_FLOAT32 force-writable=true  ! '
            'queue name=inference_hailofilter_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! '
            'hailofilter name=inference_hailofilter '
            'so-path=/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so   '
            'function-name=filter_letterbox '  
            f'config-path={self.labels_json} '
            'qos=false ! '
            'queue name=inference_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0   ! '
            'inference_wrapper_agg.sink_1 inference_wrapper_agg. ! '
            'queue name=inference_wrapper_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0   ! '
            'hailotracker name=hailo_tracker class-id=1 kalman-dist-thr=0.8 iou-thr=0.9 init-iou-thr=0.7 keep-new-frames=2 keep-tracked-frames=15 keep-lost-frames=2 keep-past-metadata=False qos=False ! '
            'queue name=hailo_tracker_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0   ! '
            'queue name=identity_callback_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! '
            'identity name=identity_callback  ! '
            'queue name=hailo_display_overlay_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! '
            'hailooverlay name=hailo_display_overlay  ! queue name=hailo_display_videoconvert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! '
            'videoconvert name=hailo_display_videoconvert n-threads=2 qos=false ! '
            'queue name=hailo_display_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0  ! '
            #'fpsdisplaysink name=hailo_display video-sink=autovideosink sync=False text-overlay=False signal-fps-measurements=true'
            'fpsdisplaysink name=hailo_display video-sink=fakesink sync=False text-overlay=False signal-fps-measurements=true'
        )
        return pipeline

class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()

        self.bbox_variable = None
    
    def set_bbox(self, bbox):
        self.bbox_variable = bbox

    def get_bbox(self):
        return self.bbox_variable

class DroneDetection:
    def __init__(self):

        #tello = Tello()
        #tello.start_video()

        self.frame = None # video frame to send 
        
        self.start_loop = False # Activated when the loop callback starts
        
        self.detection = (0, 0, 0, 0, 0, 0) # bbox info
        self.detection_bool = False # if detect something send  True if not send False
                                    # used in Bboox 
        self.start_detection = False
        self.start_video_var = False #Flag to activate or deactivate ros2 comm
        self.start_bbox = False #Flag to activate or deactivate ros2 comm
        self.start_hailo = False
        th1 = threading.Thread(target=self.start_ros_communication, daemon=True)
        th1.start()
        
        while not self.start_hailo:
            print("Waiting connection")
            time.sleep(1.5)
            
        self.user_data = user_app_callback_class()
        self.app = MyGStreamerDetectionApp(self.app_callback, self.user_data)
        self.app.run()
    
    def start_ros_communication(self):
        rclpy.init()
        self.call_bbox = Bbox(self.get_bbox)
        self.call_get_img = GetImg(self.get_video)
        self.call_start_data = StartData(self.activation)

        executor = MultiThreadedExecutor()
        executor.add_node(self.call_bbox)
        executor.add_node(self.call_get_img)
        executor.add_node(self.call_start_data)
        executor.spin()
        rclpy.shutdown()
    
    def get_video(self, *args):
        if self.start_loop:
            return True, self.frame
        else :
            return False, None

    def get_bbox(self, *args):
        if self.start_loop:
            return True, self.detection_bool, self.detection
        else: 
            return False, None, None
    
    def activation(self, var):
        # StartDetection ok
        # StartVideo
        # StartBbox
        match var:
            
            case "StartDetection":
                if not self.start_detection:
                    self.start_hailo = True
                    self.start_detection = False
                    return True
                return False
            
            case "StartVideo":
                if not self.start_video_var:
                    self.call_get_img.start_timer()
                    self.start_video_var = True
                    return True
                else:
                    return False
            
            case "StopVideo":
                if self.start_video_var:
                    self.call_get_img.stop_timer()
                    self.start_video_var = False
                    return True
                else: 
                    return False
            
            case "StartBbox":
                if not self.start_bbox:
                    self.call_bbox.start_timer()
                    self.start_bbox = True
                    return True
                else:
                    return False
            
            case "StopBbox":
                if self.start_bbox:
                    self.call_bbox.stop_timer()
                    self.start_bbox = False
                    return True
                else:
                    return False
            
            case _ :
                print("Something Wrong")
                return False
    
    def deactivate_detection(self):
        self.app.shutdown()

    def app_callback(self, pad, info, user_data):
        try:
            buffer = info.get_buffer()
            if buffer is None:
                return Gst.PadProbeReturn.OK

            # Incrementa contador de frames en hailo_app
            user_data.increment()

            # Obtener formato y dimensiones
            format, width, height = get_caps_from_pad(pad)
            if format is None or width is None or height is None:
                print("⚠ Caps inválidos, descartando frame")
                return Gst.PadProbeReturn.OK

            # Convertir buffer a numpy
            frame = get_numpy_from_buffer(buffer, format, width, height)
            if frame is None or frame.ndim != 3:
                print("⚠ Frame inválido o corrupto")
                return Gst.PadProbeReturn.OK

            # Copia defensiva (para evitar acceso a memoria liberada)
            frame = frame.copy()

            # Detecciones Hailo
            roi = hailo.get_roi_from_buffer(buffer)
            detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

            detection_bool_var = False
            last_bbox = None
            last_label, last_conf = None, 0.0

            for detection in detections:
                detection_bool_var = True
                last_label = detection.get_label()
                bbox = detection.get_bbox()
                last_conf = detection.get_confidence()

                last_bbox = (
                    round(bbox.xmin(), 3), round(bbox.ymin(), 3),
                    round(bbox.xmax(), 3), round(bbox.ymax(), 3),
                    round(bbox.width(), 3), round(bbox.height(), 3)
                )
            self.detection_bool = detection_bool_var

            # Dibujar solo si hubo detección
            height_frame, width_frame, _ = frame.shape
            print(height_frame, ":", width_frame)
            if detection_bool_var and last_bbox:
                xmin, ymin, xmax, ymax, w_norm, h_norm = last_bbox
                # Escalar a coordenadas absolutas
                x_min = max(0, int(xmin * width_frame))
                y_min = max(0, int(ymin * height_frame))
                x_max = min(width_frame - 1, int(xmax * width_frame))
                y_max = min(height_frame - 1, int(ymax * height_frame))
                w_box = int(w_norm*width_frame)
                h_box = int(h_norm*height_frame)
                self.detection = (x_min, y_min, x_max, y_max, w_box, h_box)
                

                if x_max > x_min and y_max > y_min:
                    bbox_cvzone = [x_min, y_min, x_max - x_min, y_max - y_min]
                    cvzone.cornerRect(frame, bbox_cvzone, colorC=(255, 0, 0), colorR=(255, 0, 0))
                    #cvzone.putTextRect(frame,
                    #                   last_label if last_label else "Obj",
                    #                   (x_min, y_min - 10),
                    #                   scale=1, thickness=1,
                    #                   colorR=(255, 0, 0),
                    #                   font=cv2.FONT_HERSHEY_COMPLEX)
                    print(f"[DETECCIÓN BBOX] {last_label} "
                          f"({x_min}, {y_min}, {x_max}, {y_max}) "
                          f"conf={last_conf:.2f}")

            # Redimensionar frame para salida
            img = frame[56:height_frame-56, 0:width_frame]
            img = cv2.resize(img, None, fx=0.82, fy=0.82, interpolation=cv2.INTER_AREA)
            self.frame = img
            # Actualizar en user_data
            user_data.set_frame(self.frame)

            # Flag para indicar que ya hay loop válido
            self.start_loop = True

        except Exception as e:
            print(f"❌ Error en app_callback: {e}")

        return Gst.PadProbeReturn.OK

def main():
    DroneDetection()
    
