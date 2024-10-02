import cv2
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image as tf_image

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results


def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    results = predict(chosen_model, img, classes, conf=conf)

    classic_img=img.copy()
    dog_crop=[]

    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            dog_crop.append(classic_img[y1:y2, x1:x2])

    return img, dog_crop

def format_label(label):
    formatted_label = label.split('-', 1)[1].replace('_', ' ') if '-' in label else label
    return formatted_label.replace('-', ' ').title()

# Loading models
yolov8 = YOLO('yolov8n.pt')
model = tf.lite.Interpreter(model_path='model.tflite')
model.allocate_tensors()

# Loading labels
with open('labels.json', 'r') as json_file:
    labels = json.load(json_file)
labels = {key: format_label(value) for key, value in labels.items()}

# User input
file_path = input('Path to the image file: ')
user_image = cv2.imread(file_path)
if user_image is None:
    print('Unable to open the image')
    sys.exit()
user_image = cv2.cvtColor(user_image, cv2.COLOR_BGR2RGB)

# Detecting dogs
detection_results = predict_and_detect(yolov8, user_image, classes=[16], conf=0.5)
if len(detection_results[1]) == 0:
    print('No object detected')
    sys.exit()

plt.imshow(detection_results[0])
plt.axis('off')
plt.show()

# Prediction
for result in detection_results[1]:

    # Preprocessing
    image_resized = cv2.resize(result, (224, 224))
    img_array = np.array(image_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Prediction
    model.set_tensor(model.get_input_details()[0]['index'], img_array)
    model.invoke()
    predictions = model.get_tensor(model.get_output_details()[0]['index'])
    
    predicted_class_index = np.argmax(predictions, axis=1)
    predicted_class_label = labels[f'{predicted_class_index[0]}']
    print("Predicted dog breed:", predicted_class_label)

    # Top 3 predictions
    for i in np.argsort(predictions[0])[-3:][::-1]:
        class_label = labels[f'{i}']
        probability = predictions[0][i] * 100
        print(f"{probability:.2f}% \t {class_label}")

    # Display
    plt.imshow(result)
    plt.title(predicted_class_label)
    plt.axis('off')
    plt.show()