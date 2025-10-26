import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from drone_detection_interface.msg import HailoDetection
from drone_detection_interface.srv import StartSomething

from typing import Callable


class Bbox(Node): # Ready
    def __init__(self, get_bbox_method: Callable[[], tuple]):
        super().__init__("bbox_publisher")
        self.cb = ReentrantCallbackGroup()
        self.get_battery_method = get_bbox_method
        self.publisher_ = self.create_publisher(
                                    HailoDetection, 
                                    "get_bbox", 
                                    10, callback_group=self.cb)
        #self.timer_ = self.create_timer(0.2, self.bbox_callback, callback_group=self.cb)
        self.timer_running = False

    def bbox_callback(self):
        bbox_data = HailoDetection()
        verify, det_bool, bbox_tuple = self.get_battery_method()

        if verify:
            bbox_data.x_min = bbox_tuple[0]
            bbox_data.y_min = bbox_tuple[1]
            bbox_data.x_max = bbox_tuple[2]
            bbox_data.y_max = bbox_tuple[3]
            bbox_data.width = bbox_tuple[4]
            bbox_data.height = bbox_tuple[5]
            bbox_data.detection = det_bool
            self.publisher_.publish(bbox_data)

    def stop_timer(self):
        if self.timer_running:
            self.destroy_timer(self.timer_)
            self.timer_running = False
    
    def start_timer(self):
        if not self.timer_running:
            self.timer_ = self.create_timer(0.1, self.bbox_callback, callback_group=self.cb)
            self.timer_running = True

class GetImg(Node): # Ready
    def __init__(self, get_frame):
        super().__init__("get_img_publisher")
        self.get_frame = get_frame
        self.bridge_object = CvBridge()
        self.cb = ReentrantCallbackGroup()
        self.publisher_ = self.create_publisher(Image, "camara_img", 20, callback_group=self.cb)
        #self.timer_ = self.create_timer(0.1, self.camara_callback)
        self.timer_running = False

    def camara_callback(self):
        success, frame = self.get_frame()
        if success:
            img_msg = self.bridge_object.cv2_to_imgmsg(frame)
            self.publisher_.publish(img_msg)
    
    def stop_timer(self):
        if self.timer_running:
            self.destroy_timer(self.timer_)
            self.timer_running = False
    
    def start_timer(self):
        if not self.timer_running:
            self.timer_ = self.create_timer(0.1, self.camara_callback, callback_group=self.cb)
            self.timer_running = True

class StartData(Node):
    def __init__(self, start_data_method):
        super().__init__("start_data_det_srv")
        self.cb = MutuallyExclusiveCallbackGroup()
        self.start_data_method = start_data_method
        self.server_ = self.create_service(StartSomething, "start_data_detection", self.start_video_callback, callback_group = self.cb)
        self._logger.info("start video server has been started")
        
    def start_video_callback(self, request: StartSomething.Request, response: StartSomething.Response):
        response.verify = self.start_data_method(request.start)
        return response