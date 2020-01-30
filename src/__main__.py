import cv2 as cv

from .video_capture import VideoCaptureFrameManager
from .facial_detection import detect_faces



if __name__ == "__main__":
    with VideoCaptureFrameManager() as frames:
        for frame in frames:
            for x, y, width, height in detect_faces(frame, algorithm="viola-jones"):
                cv.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2
                )
            cv.imshow("frame", frame)
            if cv.waitKey(1) == ord("q"):
                break
    cv.destroyAllWindows()
