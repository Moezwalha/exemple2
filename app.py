from flask import Flask, request, jsonify
import cv2
import numpy as np
from preprocess4 import Pr1
from preprocess2 import test_transforms
import torch
from PIL import Image
import base64
from torchvision import models
import torch.nn as nn

def get_net():
    finetune_net = nn.Sequential()
    finetune_net.features = models.resnet18(weights='ResNet18_Weights.DEFAULT')


    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 26))

    finetune_net = finetune_net.to('cpu')

    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net


# Load the saved parameters
saved_params = torch.load('my_model61.pt', map_location=torch.device('cpu'))

# Create a new instance of the model and load the parameters
model_test = get_net()
model_test.load_state_dict(saved_params)
model_test.eval()
classes = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


app = Flask(__name__)


@app.route('/', methods=['POST'])
def process_image():
    try:
        encoded_string = request.form['encoded_string']
        # Decode the base64-encoded string to bytes
        image_bytes = base64.b64decode(encoded_string)
        
        # Convert the image bytes to a numpy array
        nparr = np.fromstring(image_bytes, np.uint8)
        # Decode the numpy array to an image using OpenCV
        frame = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        # Process the image as needed
        
        p1 = Pr1(frame)
        processed_frame = p1.detect_crop_and_segment_hands(p1.image)
        if processed_frame is not None: 
                cropped_hand_array = Image.fromarray(processed_frame)
                # Apply the transformations
                img_tensor = test_transforms(cropped_hand_array)
                #Make a prediction using the model
                prediction = model_test(img_tensor[None].to("cpu"))            
                # Get the predicted label
                pred_label = classes[torch.max(prediction, dim=1)[1]]

        return pred_label
    except Exception as e:
        return f'Error: {str(e)}'
    

if __name__ == '__main__':
    app.run(debug=True)