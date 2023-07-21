import base64
import io

import dlib
import os
import cv2
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify

face_discriptors = None
threshold = 0.5
file = open(r'C:\Users\Zaid Mahmud\PycharmProjects\pythonProject\face_discriptors_data.txt', 'r')
s = file.read()
s = s.replace('[', ' ')
s = s.replace(']', ' ')
s = s.replace('\n', ' ')
s_lis = s.split()

app = Flask(__name__)

for i in range(1, 128):
    face_discriptor = []
    for j in range((i - 1) * 128, i * 128):
        face_discriptor.append(float(s_lis[j]))
    face_discriptor = np.asarray(face_discriptor, dtype=np.float64)
    face_discriptor = face_discriptor[np.newaxis, :]
    if face_discriptors is None:
        face_discriptors = face_discriptor
    else:
        face_discriptors = np.concatenate((face_discriptors, face_discriptor), axis=0)

file = open(r'C:\Users\Zaid Mahmud\PycharmProjects\pythonProject\index_data.txt', 'r')
s1 = file.read()

index = s1.split('\n')

face_detector = dlib.get_frontal_face_detector()
points_detector = dlib.shape_predictor(r'C:\Users\Zaid Mahmud\PycharmProjects\pythonProject'
                                       r'\shape_predictor_68_face_landmarks.dat')
face_descriptor_extractor = dlib.face_recognition_model_v1(r"C:\Users\Zaid Mahmud\PycharmProjects\pythonProject"
                                                           r"\dlib_face_recognition_resnet_model_v1.dat")


@app.route('/predict', methods=['POST'])
def predict():
    img_str = request.form.get('Image_string')
    imgdata = base64.b64decode(str(img_str))
    img = Image.open(io.BytesIO(imgdata))
    img_con = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_np = np.array(img_con, 'uint8')
    face_detection = face_detector(image_np, 1)
    for face in face_detection:
        points = points_detector(image_np, face)
        face_descriptor = face_descriptor_extractor.compute_face_descriptor(image_np, points)
        face_descriptor = [f for f in face_descriptor]
        face_descriptor = np.asarray(face_descriptor, dtype=np.float64)
        face_descriptor = face_descriptor[np.newaxis, :]

        distances = np.linalg.norm(face_descriptor - face_discriptors, axis=1)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        if min_distance <= threshold:
            name_pred = index[min_index].split('/')[4].split('_')[0]
        else:
            name_pred = 'Not identified'
        l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(img, (l, t), (r, b), (0, 255, 255), 2)
        cv2.putText(img, 'Pred: ' + name_pred, (l - 20, b), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
    pred_str=base64.b64encode(img)
    return jsonify({'Result_string': str(pred_str)})


if __name__ == '__main__':
    app.run(debug=True)

# video_capture = cv2.VideoCapture(0)
#
# while True:
#     # Capture frame-by-frame
#     ret, frame = video_capture.read()
#     img_con = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     image_np = np.array(img_con, 'uint8')
#     face_detection = face_detector(image_np, 1)
#     for face in face_detection:
#         points = points_detector(image_np, face)
#         face_descriptor = face_descriptor_extractor.compute_face_descriptor(image_np, points)
#         face_descriptor = [f for f in face_descriptor]
#         face_descriptor = np.asarray(face_descriptor, dtype=np.float64)
#         face_descriptor = face_descriptor[np.newaxis, :]
#
#         distances = np.linalg.norm(face_descriptor - face_discriptors, axis=1)
#         min_index = np.argmin(distances)
#         min_distance = distances[min_index]
#         if min_distance <= threshold:
#             name_pred = index[min_index].split('/')[4].split('_')[0]
#         else:
#             name_pred = 'Not identified'
#         l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
#         cv2.rectangle(frame, (l, t), (r, b), (0, 255, 255), 2)
#         cv2.putText(frame, 'Pred: ' + name_pred, (l - 20, b), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
#
#     # Display the resulting frame
#     cv2.imshow('Video', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything is done, release the capture
# video_capture.release()
# cv2.destroyAllWindows()
