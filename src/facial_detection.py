import functools

import cv2 as cv


algorithms = {}



def register_algorithm(name=None, remove_suffix="_detect_faces"):
    def dectorator(func):
        name = dectorator.name
        if name is None:
            name = func.__qualname__
            if name.endswith(remove_suffix):
                name = name[:-len(remove_suffix)]
        algorithms[name] = func
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, *kwargs)
        return wrapper
    dectorator.name = name
    return dectorator



@register_algorithm(name="viola-jones")
def viola_jones_detect_faces(original_image):
    """
    Detects faces using Viola-Jones algorithm.

    Uses code from https://realpython.com/traditional-face-detection-python/.
    """
    grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    face_cascade_classifier = cv.CascadeClassifier(
        f"{cv.data.haarcascades}/haarcascade_frontalface_alt.xml"
    )
    return face_cascade_classifier.detectMultiScale(grayscale_image)



def detect_faces(*args, algorithm=None):
    return algorithms[algorithm](*args)
