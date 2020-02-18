class VideoCaptureError(Exception):
    pass


class FaceDetectorError(Exception):
    pass


class DlibFaceDetectorError(FaceDetectorError):
    pass


class NoMatchingEncodingError(DlibFaceDetectorError):
    pass
