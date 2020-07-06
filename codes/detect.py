import cv2
import numpy as np

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
class_to_classify = []


with open("coco.names", "r") as f:
    class_to_classify = [line.strip() for line in f.readlines()]
    
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(class_to_classify), 3))

img = cv2.imread("frame9.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

objects = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(objects)
result_from_yolo = net.forward(output_layers)


class_ids = []
confidences = []
coordinates = []
for result in result_from_yolo:
    for process in result:
        scores = process[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            
            circle_c_x = int(process[0] * width)
            circle_c_y= int(process[1] * height)
            w = int(process[2] * width)
            h = int(process[3] * height)
            
            x = int(circle_c_x - w / 2)
            y = int(circle_c_y - h / 2)
            coordinates.append([x, y, w, h])
            
            confidences.append(float(confidence))
            
            class_ids.append(class_id)

reduce_same_class = cv2.dnn.NMSBoxes(coordinates, confidences, 0.5, 0.4)

for corr in range(len(coordinates)):
    if corr in reduce_same_class:
        x, y, w, h = coordinates[corr]
        label = str(class_to_classify[class_ids[corr]])
        color = colors[corr]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)  #cv2.FONT_HERSHEY_PLAIN
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
