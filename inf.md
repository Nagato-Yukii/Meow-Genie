一些模型因为体积庞大不容易上传到Github,按照我本地的结构,需要补充的模型有
## 文件结构
```plaintext
clipmodel:.
   RN50.pt
```
```plaintext
cls_path:.
   24010302.pth
```
```plaintext
control_net:.
│  control_v11f1p_sd15_depth.pth
│  control_v11p_sd15_softedge.pth
│  res101.pth
│  table5_pidinet.pth
│
├─leres
│      config.json
│      latest_net_G.pth
│      res101.pth
│
└─pidinet
        config.json
        table5_pidinet.pth
```
```plaintext
lora:.
    Cute_Mech_style可爱机甲风_v1.0.safetensors
    Gundam_Mecha 高达机甲_v5.2 动态姿势版.safetensors
    【黑鸭】立体剪纸风_v1.0.safetensors
    东方美学_dongfangmeixue .safetensors
    儿童书籍插画_v1.0.safetensors
    光环风格机甲 _ 硬机械_v1.0.safetensors
    全网首发丨花园里的阿猫阿狗画风模型_v1.0.safetensors
    卡通画cartoon illustration_V2.0_v2.0.safetensors
    厚涂油画·风格_v1.0.safetensors
    国风绘本插图丨画风加强_v1.0.safetensors
    手绘简约插画_GP illustration.safetensors
    水墨笔触_v1.0.safetensors
    青花-版画（泛用版）_v1.0（泛用版）.safetensors
    麒麟_v1.0.safetensors
```
```plaintext
safetensors:.
    AWPainting_v1.2.safetensors
    NingLO-PureCGrealistic_3.0_PureCGrealistic_3.0.safetensors
    ReVAnimated_v122_V122.safetensors
    sd_xl_base_1.0.safetensors
    SHMILY古典炫彩_v1.0.safetensors
    手绘插画chinese style_v2.0.safetensors
```
```plaintext
upscale:.
    x4-upscaler-ema.ckpt
    x4-upscaler-ema.safetensors
```
其中cls_path的24010302.pth是自己训练的猫咪分类模型,其他都能搜索到
