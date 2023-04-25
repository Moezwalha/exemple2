#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision.transforms as transforms


# In[2]:


test_transforms = transforms.Compose([
        
    # resize the image to 224x224 pixels
    transforms.Resize((224, 224)),
    
    # convert the image to a PyTorch tensor
    transforms.ToTensor(),
        
    # normalize the tensor by subtracting the mean and dividing by the standard deviation of each color channel
    transforms.Normalize([0.5189, 0.4991, 0.5138],
                             [0.2264, 0.2539, 0.2625])
])


# In[ ]:




