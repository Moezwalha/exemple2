import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


class Pr1():
    
    # init method or constructor
    def __init__(self,image):
        self.image = image

        
    @staticmethod
    def get_bounding_box(hand_landmarks, image, margin=0.2):
        # Initialize minimum and maximum values for x and y coordinates
        x_min = y_min = 1.0
        x_max = y_max = 0.0

        # Loop through each landmark of the hand
        for landmark in hand_landmarks.landmark:
            # Get the x, y, and z coordinates of the landmark
            x, y, z = landmark.x, landmark.y, landmark.z
            # Update the minimum and maximum values for x and y coordinates
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

        # Calculate the width and height of the bounding box
        width, height = x_max - x_min, y_max - y_min
        # Apply the margin to the bounding box coordinates
        x_min -= width * margin
        y_min -= height * margin
        x_max += width * margin
        y_max += height * margin

        # Convert the bounding box coordinates to integer values
        x_min, y_min = int(x_min * image.shape[1]), int(y_min * image.shape[0])
        x_max, y_max = int(x_max * image.shape[1]), int(y_max * image.shape[0])

        # Return the bounding box coordinates
        return x_min, y_min, x_max, y_max

    
    @staticmethod
    def detect_crop_and_segment_hands(image):
        # Define kernel size and number of dilation iterations for the mask
        kernel_size = (1, 1)
        dilation_iterations = 1
        
        # Initialize MediaPipe Hands
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

        # Convert the frame to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run the hand detection model on the frame
        results = hands.process(image)

        # If no hands were detected, return the original frame
        if results.multi_hand_landmarks is None:
            hands.close()
            return image
        
        # Initialize a list to store the cropped images
        cropped_images = []
        
        # Loop through each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the bounding box coordinates for the hand
            x_min, y_min, x_max, y_max = Pr1.get_bounding_box(hand_landmarks, image)

            # Adjust bounding box coordinates to ensure crop falls within frame bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image.shape[1], x_max)
            y_max = min(image.shape[0], y_max)
            
            # Crop the frame to just the hand region
            hand_image = image[y_min:y_max, x_min:x_max]

            # Apply hand segmentation mask
            mask_image = np.zeros_like(image)
            mp_drawing.draw_landmarks(mask_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)
            _, mask_image = cv2.threshold(mask_image, 1, 255, cv2.THRESH_BINARY)
            # Add thickness to the mask
            kernel = np.ones(kernel_size, np.uint8)
            mask_image = cv2.dilate(mask_image, kernel, iterations=dilation_iterations)

            hand_masked = cv2.bitwise_and(hand_image, hand_image, mask=mask_image[y_min:y_max, x_min:x_max])
            
            # Apply thresholding to the cropped hand image
            threshold_value = 20 # choose a threshold value
            max_value = 255 # maximum value to use with THRESH_BINARY
            _, hand_masked = cv2.threshold(hand_masked, threshold_value, max_value, cv2.THRESH_BINARY)
            # Add the masked cropped image to the list of images
            cropped_images.append(hand_masked)
        
        # Clean up
        hands.close()

        # If no hands were detected, return the original frame
        if not cropped_images:
            return image

        # If only one hand was detected, return the cropped image
        else:
            return cropped_images[0]