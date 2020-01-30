import cv2 as cv

from .errors import VideoCaptureError



def video_capture_frames(video_capture):
    while True:
        was_recieved, frame = video_capture.read()
        if not was_recieved:
            raise VideoCaptureError("Frame not recieved!")
        yield frame



class VideoCaptureFrameManager:
    def __init__(self, device=0):
        self.device = device

    def __enter__(self):
        self.video_capture = cv.VideoCapture(self.device)
        if not self.video_capture.isOpened():
            raise VideoCaptureError(f"Capture would not open on devices '{self.device}'!")
        return video_capture_frames(self.video_capture)

    def __exit__(self, *args):
        self.video_capture.release()
