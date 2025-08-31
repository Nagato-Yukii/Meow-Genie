# Meow-Genie

## 文件结构
```plaintext
Meow:.
│
│  cls_obj.py
│  generator.py
│  interface.py
│  main.py
│  project.ini
│  read_conf.ipynb
│  utils.py
│  requirement.txt
│  inf.md
│
├─clipmodel
│       RN50.pt (搜索下载)
│
├─cls_data #训练猫咪分类模型的数据集
│  ├─buoumao
│  ├─jiafeimao
│  ├─jinjilamao
│  ├─lihuamao
│  ├─meiguoduanmaomao
│  ├─mianyinmao
│  ├─sanhuamao
│  ├─wumaomao
│  └─yingguoduanmaomao
│
├─cls_path
│  24010302.pth(自己训练的猫咪分类模型)
│
├─control_net
│  │  control_v11f1p_sd15_depth.pth(搜索下载)
│  │  control_v11p_sd15_softedge.pth(搜索下载)
│  │  res101.pth(搜索下载)
│  │  table5_pidinet.pth(搜索下载)
│  │
│  ├─leres
│  │      config.json
│  │      latest_net_G.pth(搜索下载)
│  │      res101.pth(搜索下载)
│  │
│  └─pidinet
│         config.json
│         table5_pidinet.pth(搜索下载)
│
├─flagged
│
├─img
│
├─lora
│      Cute_Mech_style可爱机甲风_v1.0.safetensors(搜索下载)
│      Gundam_Mecha 高达机甲_v5.2 动态姿势版.safetensors(搜索下载)
│      【黑鸭】立体剪纸风_v1.0.safetensors(搜索下载)
│      东方美学_dongfangmeixue .safetensors(搜索下载)
│      儿童书籍插画_v1.0.safetensors(搜索下载)
│      光环风格机甲 _ 硬机械_v1.0.safetensors(搜索下载)
│      全网首发丨花园里的阿猫阿狗画风模型_v1.0.safetensors(搜索下载)
│      卡通画cartoon illustration_V2.0_v2.0.safetensors(搜索下载)
│      厚涂油画·风格_v1.0.safetensors(搜索下载)
│      国风绘本插图丨画风加强_v1.0.safetensors(搜索下载)
│      手绘简约插画_GP illustration.safetensors(搜索下载)
│      水墨笔触_v1.0.safetensors(搜索下载)
│      青花-版画（泛用版）_v1.0（泛用版）.safetensors(搜索下载)
│      麒麟_v1.0.safetensors(搜索下载)
│
├─models_cache
|
├─safetensors
│      AWPainting_v1.2.safetensors(搜索下载)
│      NingLO-PureCGrealistic_3.0_PureCGrealistic_3.0.safetensors(搜索下载)
│      ReVAnimated_v122_V122.safetensors(搜索下载)
│      sd_xl_base_1.0.safetensors(搜索下载)
│      SHMILY古典炫彩_v1.0.safetensors(搜索下载)
│      手绘插画chinese style_v2.0.safetensors(搜索下载)
│
├─upscale
│      x4-upscaler-ema.ckpt(搜索下载)
│      x4-upscaler-ema.safetensors(搜索下载)
|
├─yolo_pth
│      yolov8m.pt
│      yolov8n.pt
└─     yolov8s.pt


```
在运行代码前,请先按inf.txt安装相应的模型,再按requirement配置环境,然后只需要在该目录下输入指令python main.py即可,当然你可以将此代码作为模板,使用更多模型丰富功能.
