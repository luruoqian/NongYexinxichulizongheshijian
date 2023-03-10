from ast import main
from email.policy import default
import json
import base64
from unicodedata import name
import requests
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.preprocessing import image
import keras
from keras.preprocessing import image
import numpy as np

import numpy as np
from keras.applications.imagenet_utils import preprocess_input

import socket
import os
import sys
import struct




#-------------------------------------------企业微信的参数-------------------------------------------------
Secret = "n4Miog2pJhexNbmRXDXOcYmgX46kWyQPKmrcHSHmPkQ"
corpid = 'wwee95e10e098c4460'
url = 'https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid=wwee95e10e098c4460&corpsecret=n4Miog2pJhexNbmRXDXOcYmgX46kWyQPKmrcHSHmPkQ'

#---------------------------------------------------------------------------------------------------------

#-----------------------------------------AI算法的参数---------------------------------------------------
image_path = r'D:\qianSpace\NongYexinxichulizongheshijian\WorkSpace\image.jpg'

new_model = keras.models.load_model(r'D:\qianSpace\NongYexinxichulizongheshijian\path_to_saved_model')
#---------------------------------------------------------------------------------------------------------



def sendmessage(name):
    #根据反馈数据读取信息
    fr  =  open ('D://qianSpace//NongYexinxichulizongheshijian//WorkSpace//'+name+'.txt' , encoding='UTF-8')
    message = fr.readlines()
    fr.close()
    
    data = {
        "touser" : "LuRuoQian|QianMo",   # 向这些用户账户发送
        # "toparty" : "PartyID1|PartyID2",   # 向这些部门发送
        "msgtype" : "text",
        "agentid" : 1000002,                       # 应用的 id 号
        "text" : {
                        "content": "%s"%message
                        },
        "safe":0
    }
    
    r = requests.post(url="https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={}".format(access_token),data=json.dumps(data))
    print(r.json())
 
 
def socket_service():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #IP地址留空默认是本机IP地址
        s.bind(('192.168.43.49', 8000))
        s.listen(7)
    except socket.error as msg:
        print(msg)
        sys.exit(1)
 
    print("连接开启，等待传图...")
	
    while True:
        sock, addr = s.accept()
        deal_data(sock, addr)
    
        break
    s.close()
 
def deal_data(sock, addr):
    print("成功连接上 {0}".format(addr))
 
    while True:
        fileinfo_size = struct.calcsize('128sl')
        buf = sock.recv(fileinfo_size)
        if buf:
            filename, filesize = struct.unpack('128sl', buf)
            fn = filename.decode().strip('\x00')
            #PC端图片保存路径
            new_filename = os.path.join(r'D:\qianSpace\NongYexinxichulizongheshijian\WorkSpace', fn)
 
            recvd_size = 0
            fp = open(new_filename, 'wb')
 
            while not recvd_size == filesize:
                if filesize - recvd_size > 1024:
                    data = sock.recv(1024)
                    recvd_size += len(data)
                else:
                    data = sock.recv(1024)
                    recvd_size = filesize
                fp.write(data)
            fp.close()
        sock.close()
        break
     
    
if __name__ == "__main__":
    
    socket_service()
    #企业微信的accsee_token(小写)：
    getr = requests.get(url=url.format(corpid,Secret))
    #print(getr)
    # {'errcode': 0, 'errmsg': 'ok', 'access_token': 't2HxARFMOgge-neHJwYXe4MrIXlFcu2m_Ev1pGQIAcmu-Kt1kQ7pey6jkPfdecqyvvZ9RGb3oSfjL1-lbbp1Y6UGGi8ZjNNd64AALtbR58ot1lh6VjE2ITkiWwgIftwWyryNDw_1AJAtVYYQxKU2O16a7NhHVEdcHG20u8czD-QUDUec1LqI4503OcVGzdR4Cq_4yA6a3fIkVLdQ_u3CHg', 'expires_in': 7200}
    access_token = getr.json().get('access_token')
    
    
    
    
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
        name = 'Bacterial leaf blight'
        print('Bacterial leaf blight')
    elif pred[0] == 2:
        name = 'Brown spot'
        print('Brown spot')
    elif pred[0] == 0 :
        name = 'Leaf smut'
        print('Leaf smut')
    else:
        print("shayemeiy")
        
    sendmessage(name)