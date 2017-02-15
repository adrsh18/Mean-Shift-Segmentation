
# coding: utf-8

# In[25]:


import cv2
from mean_shift import MeanShift
import matplotlib.pyplot as plt

im = cv2.cvtColor(cv2.imread('butterfly.jpg'), cv2.COLOR_BGR2RGB)

ms_object = MeanShift(3,20,kernel="epanechnikov")
result, result_with_borders = ms_object.segment(im)

def display_image(image):
    #plt.set_cmap('gray')
    plt.axis("off")
    plt.imshow(image)
    plt.show()

display_image(im)
display_image(result)
display_image(result_with_borders)
cv2.imwrite('result-one.jpg', cv2.cvtColor(result_with_borders, cv2.COLOR_RGB2BGR))


# In[48]:

import cv2
from mean_shift import MeanShift
import matplotlib.pyplot as plt

im = cv2.cvtColor(cv2.imread('house.jpg'), cv2.COLOR_BGR2RGB)
#im = cv2.imread('house.jpg')

ms_object = MeanShift(6,18)
result, result_with_borders = ms_object.segment(im)

def display_image(image):
    plt.set_cmap('gray')
    plt.axis("off")
    plt.imshow(image)
    plt.show()

display_image(im)
display_image(result)
display_image(result_with_borders)

cv2.imwrite('result-two.jpg', cv2.cvtColor(result_with_borders, cv2.COLOR_RGB2BGR))

# In[53]:


import cv2
from mean_shift import MeanShift
import matplotlib.pyplot as plt

im = cv2.cvtColor(cv2.imread('hill_house.jpg'), cv2.COLOR_BGR2RGB)
#im = cv2.imread('house.jpg')

ms_object = MeanShift(2,30)
result, result_with_borders = ms_object.segment(im)

def display_image(image):
    #plt.set_cmap('gray')
    plt.axis("off")
    plt.imshow(image)
    plt.show()

display_image(im)
display_image(result)
display_image(result_with_borders)

cv2.imwrite('result-three.jpg', cv2.cvtColor(result_with_borders, cv2.COLOR_RGB2BGR))


# In[ ]:



