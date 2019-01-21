from statistics import mode

import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model, load_s3fd_model, detect_faces_sfd
from utils.preprocessor import preprocess_input

# parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
sfd_model_path = './SFD_pytorch/s3fd_convert.pth'
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
frame_window = 10
gender_offsets = (30, 60)

# loading models
# face_detection = load_detection_model(detection_model_path)
face_detection = load_s3fd_model(sfd_model_path)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
gender_target_size = gender_classifier.input_shape[1:3]

# starting lists for calculating modes
gender_window = []

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)


def classify_frame(bgr_image):
    # bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    max_size = 512 

    aspect_ratio = rgb.shape[1] / rgb.shape[0]
    scaling_factor = max([rgb.shape[1], rgb.shape[0]]) / max_size

    new_width = int(rgb.shape[1] / scaling_factor)
    new_height = int(rgb.shape[0] / scaling_factor)

    rgb_image = cv2.resize(rgb, (new_width, new_height))

    faces = detect_faces_sfd(face_detection, rgb_image)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
        rgb_face = rgb_image[y1:y2, x1:x2]

        gray_face = gray_image[y1:y2, x1:x2]
        try:
            rgb_face = cv2.resize(rgb_face, (gender_target_size))
        except:
            continue
        gray_face = preprocess_input(gray_face, False)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        rgb_face = np.expand_dims(rgb_face, 0)
        rgb_face = preprocess_input(rgb_face, False)
        gender_prediction = gender_classifier.predict(rgb_face)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]
        gender_window.append(gender_text)

        if len(gender_window) > frame_window:
            gender_window.pop(0)
        try:
            gender_mode = mode(gender_window)
        except:
            continue

        if gender_text == gender_labels[0]:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, gender_mode,
                  color, 0, -20, 1, 1)
    
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)


img_counter = 0

while True:
    ret, frame = video_capture.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        classify_frame(frame)
        # SPACE pressed
#        img_name = "opencv_frame_{}.png".format(img_counter)
#        cv2.imwrite(img_name, frame)
        img_counter += 1

video_capture.release()

cv2.destroyAllWindows()

#while True:
#    if cv2.waitKey(0) & 0xFF == ord('p'):
#        take_picture()
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
