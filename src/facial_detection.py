import abc
import functools
import math
import statistics

import cv2 as cv
import face_recognition as dlib_face_recognition
import numpy as np

from .errors import NoMatchingEncodingError
from .location import UnknownLocation, BBox


class BaseFaceDetector(abc.ABC):

    @abc.abstractmethod
    def update_from_image(self):
        pass


class ViolaJonesFaceDetector(BaseFaceDetector):
    """
    Detects faces using Viola-Jones algorithm.

    Uses code from https://realpython.com/traditional-face-detection-python/.
    """

    def __init__(self):
        self.faces = []

    def update_from_image(self, original_image):
        grayscale_frame = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
        face_cascade_classifier = cv.CascadeClassifier(
            f"{cv.data.haarcascades}/haarcascade_frontalface_alt.xml"
        )
        self.faces = [
            (BBox(left=left, top=top, width=width, height=height), None, {})
            for left, top, width, height in face_cascade_classifier.detectMultiScale(
                grayscale_frame
            )
        ]


class DlibFaceDetector(BaseFaceDetector):
    """
    Detects faces using dlib face recognition algorithmself.

    Uses amazing python API from https://github.com/ageitgey/face_recognition.
    """

    class Faces:

        def __init__(self,
                     known_face_images=None,
                     known_face_encodings=None,
                     default_max_encoding_match_dist=0.6):
            self.default_max_encoding_match_dist = default_max_encoding_match_dist
            self.ids = []
            self.locations = []
            self.encodings = []
            self.id_index_location_index_map = {}
            self.location_index_id_index_map = {}
            self.id_index_encoding_indicies_map = {}
            self.encoding_index_id_index_map = {}
            if known_face_encodings is None:
                known_face_encodings = {}
            if known_face_images is not None:
                for id, images in known_face_images.items():
                    encodings = [
                        dlib_face_recognition.face_encodings(
                            dlib_face_recognition.load_image_file(image)
                        )[0]
                        for image in images
                    ]
                    known_face_encodings[id] = encodings
            for id, encodings in known_face_encodings.items():
                for encoding in encodings:
                    self.add_id_with_encoding_and_location(id, encoding, UnknownLocation())

        def __iter__(self):
            for id_index, id in enumerate(self.ids):
                location = self.locations[self.id_index_location_index_map[id_index]]
                yield location, id, {}

        def add_id(self, id):
            self.ids.append(id)

        def add_location(self, location):
            self.locations.append(location)

        def add_encoding(self, encoding):
            self.encodings.append(encoding)

        def map_id_index_to_location_index(self, id_index, location_index):
            self.id_index_location_index_map[id_index] = location_index
            self.location_index_id_index_map[location_index] = id_index

        def map_id_index_to_encoding_index(self, id_index, encoding_index):
            self.id_index_encoding_indicies_map[id_index] = [encoding_index]
            self.encoding_index_id_index_map[encoding_index] = id_index

        def add_id_with_encoding_and_location(self, id, encoding, location):
            id_index = len(self.ids)
            encoding_index = len(self.encodings)
            location_index = len(self.locations)
            self.add_id(id)
            self.add_encoding(encoding)
            self.add_location(location)
            self.map_id_index_to_encoding_index(id_index, encoding_index)
            self.map_id_index_to_location_index(id_index, location_index)

        def change_location_from_id_index(self, id_index, new_location):
            """Changes the location of a face with a given id."""
            location_index = self.id_index_location_index_map[id_index]
            self.locations[location_index] = new_location

        def set_all_locations_to_unknown(self):
            self.locations = [UnknownLocation() for _ in range(len(self.locations))]

        def encoding_match_index(self, encoding, max_encoding_match_dist=None):
            """Returns the index of the nearest matching cached encoding."""
            if max_encoding_match_dist is None:
                max_encoding_match_dist = self.default_max_encoding_match_dist
            min_encoding_dist = math.inf
            min_encoding_dist_index = -1
            for encoding_index, encoding_dist in enumerate(
                dlib_face_recognition.face_distance(self.encodings, encoding)
            ):
                if encoding_dist < min_encoding_dist:
                    min_encoding_dist = encoding_dist
                    min_encoding_dist_index = encoding_index
            if min_encoding_dist <= max_encoding_match_dist:
                return min_encoding_dist_index
            raise NoMatchingEncodingError()

        def id_index_from_encoding_match(self, encoding, *args, **kwargs):
            encoding_index = self.encoding_match_index(encoding, *args, **kwargs)
            return self.encoding_index_id_index_map[encoding_index]

    def __init__(self,
                 *args,
                 image_scale_factor=0.25,
                 **kwargs):
        self.image_scale_factor = image_scale_factor
        self.faces = self.Faces(*args, **kwargs)
        self.unknown_face_num = 1

    def update_from_image(self, original_image):
        rgb_small_frame = cv.resize(
            original_image,
            (0, 0),
            fx=self.image_scale_factor,
            fy=self.image_scale_factor
        )[:, :, ::-1]
        dlib_face_locations = dlib_face_recognition.face_locations(rgb_small_frame)
        face_encodings = dlib_face_recognition.face_encodings(rgb_small_frame, dlib_face_locations)
        self.faces.set_all_locations_to_unknown()
        for (top, right, bottom, left), face_encoding in zip(dlib_face_locations, face_encodings):
            face_location = BBox(
                left=left/self.image_scale_factor,
                top=top/self.image_scale_factor,
                right=right/self.image_scale_factor,
                bottom=bottom/self.image_scale_factor
            )
            try:
                face_id_index = self.faces.id_index_from_encoding_match(face_encoding)
                self.faces.change_location_from_id_index(face_id_index, face_location)
            except NoMatchingEncodingError:
                self.faces.add_id_with_encoding_and_location(
                    f"unknown face#{self.unknown_face_num}",
                    face_encoding,
                    face_location
                )
                self.unknown_face_num += 1
