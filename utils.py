from ultralytics import YOLO
import numpy as np
import clip
from typing import Union
import torch
import gradio as gr
from PIL import Image
from io import BytesIO
from torchvision import transforms
import pandas as pd
from cls_obj import cls_predict
from configparser import ConfigParser
import cv2

def det_obj(img,yolopath:str,checkobj:list):
    model = YOLO(yolopath)
    res = model(img)
    obj_count = 0
    cls2count = {}
    cls_dict = {}
    # 对结果进行遍历
    for r in res:
        # 调用.names的属性获取所有的类别信息，返回一个dict类型的数据
        cls_dict = r.names
        # 调用检测对象下的boxes属性，遍历边界框相关信息
        for loc in r.boxes:
            # 获取边界框所对应的类别信息
            cls_names = cls_dict[int(loc.cls[0])]
            if cls_names in checkobj:
                obj_count+=1
                cls2count[cls_names] = obj_count
    return cls2count

def format_prompt(clip_path:str,choose_list:list,img,device:str = 'cuda'):
    model, preprocess = clip.load(clip_path, device = device)
    input_img = preprocess(img).unsqueeze(0).to(device)
    final_propmt = []
    with torch.no_grad():
        for i in range(len(choose_list)):
            _ = model.encode_image(input_img)
            choosen_text = clip.tokenize(choose_list[i]).to(device)
            logits_per_image, _ = model(input_img, choosen_text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            final_propmt.append(choose_list[i][np.argmax(probs)])
    return final_propmt

def sendImg2next(evt: gr.SelectData):
    return evt.value['image']['path']

def img2ByteIO(img: Union[np.ndarray, Image.Image]):#(img:np.ndarray|Image.Image):
    if isinstance(img,np.ndarray):
        img = Image.fromarray(img)
    img = img.convert('RGB')
    img_byte = BytesIO()
    img = img.save(img_byte,'PNG')
    bi_img = img_byte.getvalue()
    return bi_img

def img2input(img: Union[np.ndarray , Image.Image] , input_size:int):#(img:np.ndarray|Image.Image,input_size:int):
    predict_transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Resize(input_size),  # 调整图像大小为256x256
    transforms.Normalize(mean=[0.4848, 0.4435, 0.4023], std=[0.2744, 0.2688, 0.2757])])
    re_img = predict_transform(img)
    return re_img

def clsProb2dataframe(img):
    conf = ConfigParser()
    conf.read('./project.ini',encoding='utf-8')
    det_res_dict = det_obj(img,conf['YOLOV8']['root_path']+conf['YOLOV8']['yolov8_s'],conf['YOLOV8']['det_obj'].split())
    assert len(det_res_dict)!=0,'检测不到包含猫咪！请检查图片后重新上传！'
    cls_dict = {}
    cat_cls = []
    for i,cls in enumerate(conf['PREDICT_CLS']):
        if cls.startswith('cls')==True:
            cls_dict[i] = conf['PREDICT_CLS'][cls]
            cat_cls.append(conf['PREDICT_CLS'][cls])
    prob_list = cls_predict(img)[1].tolist()[0]
    round_prob_list = [round(x,3) for x in prob_list]
    my_test_pd = pd.DataFrame({'cat_cls':cat_cls,
                           'prob':round_prob_list,})
    return my_test_pd

def bar_plot_fn(img):
    dataframe = clsProb2dataframe(img)
    return gr.BarPlot(
        dataframe,
        x='cat_cls',
        y='prob',
        title="最有可能属于哪一类猫咪",
        tooltip=["prob","cat_cls"],
        y_lim=[0, 1],
        color = 'prob',
        vertical=False,
        height=380,
        width=500
    )

def resizeImg(img):
    # 输入你想要resize的图像高。
    size = 400
    # 获取原始图像宽高。
    height, width = img.shape[0], img.shape[1]
    # 等比例缩放尺度。
    scale = height/size
    # 获得相应等比例的图像宽度。
    width_size = int(width/scale)
    # resize
    image_resize = cv2.resize(img, (width_size, size))
    return image_resize
