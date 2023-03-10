import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.preprocessing import image
import keras
from keras.preprocessing import image
import numpy as np

import numpy as np
from keras.applications.imagenet_utils import preprocess_input

image_path = r'D:\qianSpace\NongYexinxichulizongheshijian\WorkSpace\archive\rice_leaf_diseases\Brown spot\DSC_0100.jpg'

new_model = keras.models.load_model(r'D:\qianSpace\NongYexinxichulizongheshijian\path_to_saved_model')


# 加载图像
img = image.load_img(image_path, target_size=(224, 224))

# 图像预处理
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

#模型预测与预测数据处理
y_pred = new_model.predict(x)

pred=np.argmax(y_pred,axis=-1)
print(pred)
if pred[0] == 1:
    print('Bacterial leaf blight')
elif pred[0] == 2:
    print('Brown spot')
elif pred[0] == 0 :
    print('Leaf smut')
else:
    print("shayemeiy")
    
    