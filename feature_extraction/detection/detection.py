# face library
import dlib
# data structure
import numpy as np
# math and geometry
import math
# image manipulation
from PIL import Image

class FaceAligner():
    def __init__(self, dest_sz, offset_pct):
        self.ler = [36, 42]  # left eye range
        self.rer = [42, 48]  # right eye range

        self.dest_sz = np.array(dest_sz)
        self.offset_pct = np.array(offset_pct)

        self.ref_dist = dest_sz[0] * (1 - 2 * offset_pct[0])

    def __affine_rotate(self, image, angle, center, resample=Image.BICUBIC):
        if center is None:
            image.rotate(angle=angle, resample=resample)
        (x0, y0) = center
        cosine = math.cos(angle)
        sine = math.sin(angle)
        a = cosine
        b = sine
        c = x0 - x0 * cosine - y0 * sine
        d = - sine
        e = cosine
        f = x0 * sine + y0 - y0 * cosine
        return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample)

    def align(self, image, landmarks):
        # find the eye
        left_eye = [landmarks[i] for i in range(self.ler[0], self.ler[1])]
        right_eye = [landmarks[i] for i in range(self.rer[0], self.rer[1])]

        # find the centroid of eyes
        lec = np.sum(left_eye, axis=0) / len(left_eye)
        rec = np.sum(right_eye, axis=0) / len(right_eye)

        # find the angle
        angle = - math.atan2(float(rec[1] - lec[1]), float(rec[0] - lec[0]))

        # find scale
        dis = np.sqrt(np.sum(np.array(lec - rec) ** 2))
        scale = dis / self.ref_dist

        # rotate the face
        rotated_image = self.__affine_rotate(image, angle, center=lec)

        # find the actual crop location
        crop_xy = (lec[0] - scale * self.dest_sz[0] * self.offset_pct[0], lec[1] - scale * self.dest_sz[1] * self.offset_pct[1])
        crop_sz = self.dest_sz * scale

        # crop the face from origin image
        face = rotated_image.crop(
            (int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0] + crop_sz[0]), int(crop_xy[1] + crop_sz[1])))
        resize_face = face.resize(self.dest_sz, Image.BICUBIC)

        return resize_face

face_size = [150, 170]
eye_offset_percentage = [0.25, 0.25]

class Detector():
    landmark_model_path = './shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(landmark_model_path)
    aligner = FaceAligner(dest_sz=face_size, offset_pct=eye_offset_percentage)

    def detect(self, image):
        imarray = np.array(image)

        detections = self.detector(imarray)

        faces = []
        for detection in detections:
            shape = self.predictor(imarray, detection)
            landmarks = list(map(lambda p: (p.x, p.y), shape.parts()))

            face = self.aligner.align(image, landmarks)
            faces.append(face)

        return faces
