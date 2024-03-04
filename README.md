# IPMV
The YOLO (You Only Look Once) algorithm is an object detection algorithm that divides an  image into a grid and predicts bounding boxes and class probabilities for each grid cell. YOLO is  known for its speed and efficiency as it processes the entire image in a single forward pass  through the neural network.

# IPMV CODE YOLO 
! gdown --id 1QFPS1EHdwakTaPSY1zbzf_17GX3GGmr1   #coco names file
! gdown --id 1w3VWBf4tP8WzDhzD_gpiNWiD-sW8V8mt   #Weight files
! gdown --id 1dAjX0vDKOThxpPOo-065y89fxsqlUve1  #yolo cfg files
! gdown --id 1YbIjc37WvUnIZZYVaaWKec8a5qy9uc6d  #Cat




from google.colab.patches import cv2_imshow
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys




# Objects in databse
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]




object_name = input("Search object in database: ")
if object_name.lower() in classes:
    print("Object found in the database")
else:
    print("Object not found in the database")
    sys.exit()
    
# Importing the image
path="./bus.jpg"
img=cv2.imread(path)
original=cv2.imread(path)
print("Input image is ")
plt.imshow(img)
height, width, channels = img.shape


# preprocessing:
# Scale pixel values to the range [0, 1]
img1 = img.astype(np.float32) / 255.0
# Resize the image to (320, 320)
img1 = cv2.resize(img1, (320, 320))
# Swap the channels from BGR to RGB
img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# Convert the image to a blob
# A blob in this case would be a 4D numpy array
image = np.expand_dims(np.transpose(img2, (2, 0, 1)), axis=0)
i = image[0].reshape(320,320,3)

# Colour and Image
yolo_img = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
plt.subplot(1,2,1)
plt.title("Orignal Image")
plt.imshow(img)
plt.subplot(1,2,2)
plt.title("Resized Image")
plt.imshow(img1)
plt.show()
plt.subplot(1,2,1)
plt.title("RGB Image")
plt.imshow(img2)
plt.subplot(1,2,2)
plt.title("Blob image")
plt.imshow(i)
plt.show()

# Load YOLO weights and configuration files
yolo = cv2.dnn.readNet("./yolov3-tiny.weights","./yolov3-tiny.cfg")
# Set input image for YOLO model
yolo.setInput(image)
# Get the names of the output layers
output_layers_name = yolo.getUnconnectedOutLayersNames()
# Forward pass through the YOLO network
layeroutput = yolo.forward(output_layers_name)

# Initialize lists to store bounding boxes, confidences, and class IDs
boxes=[]
confidences=[]
class_ids=[]

# Iterate through each output layer
for output in layeroutput :
  # Iterate through each detection in the output
  for detection in output:
    # Extract confidence scores and class IDs
    score = detection[5:]
    class_id = np.argmax(score)
    confidence = score[class_id]
    # If confidence is above a certain threshold, process the detection
    if confidence > 0.7:
      # Extract bounding box coordinates
      center_x = int(detection[0]*width)
      center_y = int(detection[0]*height)
      w = int(detection[0]*width)
      h = int(detection[0]*height)

       # Append bounding box coordinates, confidence, and class ID to respective lists
      x = int(center_x- w/2)
      y = int(center_y- h/2)
      boxes.append([x,y,w,h])
      confidences.append(float(confidence))
      class_ids.append(class_id)
print(len(boxes))
print(boxes)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2. FONT_HERSHEY_PLAIN

for i in range(0,len(boxes)):
  x,y,w,h = boxes[i]
  label = str(classes[class_ids[i]])


  cv2.rectangle(img, (x,y),(x+w, y+h), 255, 5)
  cv2.putText(img, label, (x,y), font, 10, (255,255,0), 5)
plt.subplot(1,2,1)
plt.title("Orignal Image")
plt.imshow(original)
plt.subplot(1,2,2)
plt.title("Detected Image")
plt.imshow(img)
plt.show()





