import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, Text, messagebox
import cv2
import json
import glob
import random
from tensorflow.lite.python.interpreter import Interpreter
import matplotlib.pyplot as plt
import os

# Function to perform object detection using TensorFlow Lite model
def tflite_detect_image(modelpath, imgpath, lblpath, min_conf=0.1, savepath='/content/results'):

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

    image = cv2.imread(imgpath)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
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

            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 1)

            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            label_ymin = max(ymin-10, labelSize[1] + 10)
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_COMPLEX_SMALL, .6, (255, 0, 0), 1)

            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

            accuracy = int(scores[i]*100)

            if(accuracy > 50):
                print("Detected class:", object_name, ", Accuracy:", accuracy)

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12,16))
    plt.imshow(image)
    plt.show()

    image_fn = os.path.basename(imgpath)
    base_fn, ext = os.path.splitext(image_fn)
    txt_result_fn = base_fn +'.txt'
    txt_savepath = os.path.join(savepath, txt_result_fn)

    with open(txt_savepath,'w') as f:
        for detection in detections:
            f.write('%s %.4f %d %d %d %d\n' % (detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))

    return

def run_model():
    # Set up variables for running user's model
    PATH_TO_MODEL='trainedcom.tflite'   # Path to .tflite model file
    PATH_TO_LABELS='labelmap.txt'   # Path to labelmap.txt file
    
    # Explicitly focus the Tkinter window
    root.focus_force()
    
    # Directly open the file dialog to select an input image file
    filename = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg; *.jpeg; *.png; *.bmp")])
    
    if not filename:
        messagebox.showinfo("Info", "No file selected!")
        return
    
    # Run inferencing function
    tflite_detect_image(PATH_TO_MODEL, filename, PATH_TO_LABELS)

# Initialize Tkinter window
root = tk.Tk()
root.title("Object Detection from Image")

# Create a button to load and process an image
load_button = tk.Button(root, text="Load Image", command=run_model)
load_button.pack()

# Run Tkinter event loop
root.mainloop()
