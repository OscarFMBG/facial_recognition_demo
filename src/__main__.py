import cv2 as cv

from .video_capture import VideoCaptureFrameManager
from .facial_detection import ViolaJonesFaceDetector, DlibFaceDetector
from .location import UnknownLocation, BBox


if __name__ == "__main__":
    def display_bbox(image, bbox, color=(0, 255, 0), border_width=2):
        cv.rectangle(
            image,
            (int(bbox.left), int(bbox.top)),
            (int(bbox.right), int(bbox.bottom)),
            color,
            border_width
        )

    def display_face_id(image,
                        face_id,
                        bbox,
                        color=(0, 255, 0),
                        font=cv.FONT_HERSHEY_DUPLEX,
                        font_size=0.7,
                        x_offset=0,
                        y_offset=-10):
        cv.putText(
            image,
            face_id,
            (int(round(bbox.left + x_offset)), int(round(bbox.top + y_offset))),
            font,
            font_size,
            color,
            1
        )

    def display_face_data(image, face_location, face_id, extra_data):
        if isinstance(face_location, UnknownLocation):
            return
        if isinstance(face_location, BBox):
            display_bbox(image, face_location)
            if face_id is not None:
                display_face_id(image, face_id, face_location)

    face_detector = DlibFaceDetector()

    with VideoCaptureFrameManager() as frames:
        for frame in frames:
            face_detector.update_from_image(frame)
            for face_data in face_detector.faces:
                display_face_data(frame, *face_data)
            cv.imshow("frame", frame)
            if cv.waitKey(1) == ord("q"):
                break
    cv.destroyAllWindows()
