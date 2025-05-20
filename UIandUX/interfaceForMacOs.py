import numpy as np
import tensorflow as tf
import cv2
from tensorflow.lite.python.interpreter import Interpreter
import matplotlib.pyplot as plt
import os
import streamlit as st
from PIL import Image

# Function to perform object detection using TensorFlow Lite model
def tflite_detect_image(modelpath, image, lblpath, min_conf=0.1, savepath='./results'):

    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    float_input = (input_details[0]['dtype'] == np.float32)
    input_mean = 127.5
    input_std = 127.5

    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    imH, imW, _ = image_rgb.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]

    detections = []

    for i in range(len(classes)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            cv2.rectangle(image_rgb, (xmin,ymin), (xmax,ymax), (10, 255, 0), 1)

            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            label_ymin = max(ymin-10, labelSize[1] + 10)
            cv2.rectangle(image_rgb, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
            cv2.putText(image_rgb, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_COMPLEX_SMALL, .6, (255, 0, 0), 1)

            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

            accuracy = int(scores[i]*100)

            if(accuracy > 50):
                st.write("Detected class:", object_name, ", Accuracy:", accuracy)

    st.image(image_rgb, channels="RGB", use_column_width=True)

    image_fn = os.path.basename(lblpath)
    base_fn, ext = os.path.splitext(image_fn)
    txt_result_fn = base_fn +'.txt'
    txt_savepath = os.path.join(savepath, txt_result_fn)

    with open(txt_savepath,'w') as f:
        for detection in detections:
            f.write('%s %.4f %d %d %d %d\n' % (detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))

# Streamlit UI
st.title("Object Detection from Image")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Object Detection"):
        PATH_TO_MODEL = 'trainedcom.tflite'   # Path to .tflite model file
        PATH_TO_LABELS = 'labelmap.txt'   # Path to labelmap.txt file

        tflite_detect_image(PATH_TO_MODEL, image, PATH_TO_LABELS)