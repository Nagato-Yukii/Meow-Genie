import torch
import numpy as np
import copy
from PIL import Image
from io import BytesIO
import controlnet_aux
from xpinyin import Pinyin
from controlnet_aux import PidiNetDetector,LeresDetector
from diffusers import ControlNetModel,StableDiffusionControlNetPipeline,StableDiffusionUpscalePipeline,StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from configparser import ConfigParser
from utils import det_obj,format_prompt,img2ByteIO
from typing import Union
import os

class GenPic():
    r'''
    基类:生成、放大(without ControlNet)
    @params conf_file:
    @params input_tokens:
    @params sdmodel_file: 底模路径,需要提供一个后缀名是.safetensors的文件
    '''
    def __init__(self,
                 conf_file:str,
                 input_tokens:list,
                 scheduler_type = 'euler',             #把调度器选择为欧拉
                 local_files_only = True) -> None:     #设置只选本地文件
        self.input_tokens = input_tokens
        self.scheduler_type = scheduler_type
        self.local_files_only = local_files_only
        self.torch_dtype = torch.float16
        self.conf = ConfigParser()
        self.conf.read(conf_file,encoding='utf-8')
        self.token_dict = {'布偶猫':'Ragdoll cat',
                           '狸花猫':'dragon-li',
                           '金吉拉':'Chinchilla cat',
                           '英国短毛猫':'British Shorthair cat',
                           '其它':'a cat'}
        self.output_token = None
    #def generatorPic(self,input_image:np.ndarray|Image.Image,sdmodel_file:str,gen_steps:int = 20,num_image_gen:int = 4):
    def generatorPic(self,input_image:Union[np.ndarray,Image.Image],sdmodel_file:str,gen_steps:int = 20,num_image_gen:int = 4):
        r'''
        通过 Standard Stable Diffusion Pipeline
        底模:sdmodel
        @params input_image:
        '''
        input_image = Image.fromarray(np.uint8(input_image))
        det_res_dict = det_obj(input_image,self.conf['YOLOV8']['root_path']+self.conf['YOLOV8']['yolov8_s'],self.conf['YOLOV8']['det_obj'])
        # 加载完整的sd流水线
        generator_pipe = StableDiffusionPipeline.from_single_file(sdmodel_file,
                                                                  local_files_only = self.local_files_only,
                                                                  scheduler_type = self.scheduler_type,
                                                                  torch_dtype = torch.float16)
        print(self.conf['HARD_WARE']['device_name'])
        generator_pipe.to(self.conf['HARD_WARE']['device_name'])
        generator_pipe.unet.set_attn_processor(AttnProcessor2_0()) #启用性能优化
        gen_output = generator_pipe(self.input_tokens,
                                    num_inference_steps= gen_steps,
                                    num_image_per_prompt = num_image_gen).images #生成
        return gen_output

    #def upScaleImg(self,img:np.ndarray|Image.Image):
    def upScaleImg(self,img:Union[np.ndarray,Image.Image]):
        #放大图像(提升分辨率)
        img = Image.fromarray(img).convert("RGB")
        img = img.resize((500, int(img.size[1] * 500 / img.size[0])))
        #加载放大模型
        upscaler = StableDiffusionUpscalePipeline.from_single_file(self.conf['UPSCALE']['root_path']+self.conf['UPSCALE']['x4_scale_file'],
                                                                    torch_dtype = torch.float16,
                                                                    #revision="fp16"
                                                                    )
        #upscaler.enable_model_cpu_offload()
        #upscaler.enable_vae_slicing()#启用分块VAE
        upscaler = upscaler.to("cuda")
        upscaler.enable_attention_slicing()
        upscaler.unet.set_attn_processor(AttnProcessor2_0())
        upscaler_img = upscaler(prompt = self.output_token,
                                image = img,
                                num_inference_steps = 15).images[0]
        return upscaler_img
    

class GenPicWithControlNet(GenPic):
    r'''
    继承GenPic,通过controlnet来生成图像
    @params conf_file:配置文件路径,配置文件通过configparser来进行读取
    @params input_tokens:用来生成图像的prompt,提供一个完整的list,元素为str类型
    @params control_modes:确定加入的controlnet控制模式,从controlnet-aux中获取,可填入参数为PidiNetDetector LeresDetector
    '''
    def __init__(self,conf_file:str,
                 input_tokens:list,
                 *control_modes:str) -> None:
        super().__init__(conf_file=conf_file,
                         input_tokens=input_tokens,
                         )
        self.o_input = copy.deepcopy(input_tokens)
        self.control_modes = control_modes
        self.control_obj = {}
        self.control_pipeline = []
    
        print("prepare for controlnet")
        for control_mode in control_modes:
            #加载controlnet预处理器
            if control_mode in self.conf['CONTROLNET']['control_mode']:
                print("getattr")
                cls = getattr(controlnet_aux,control_mode,None)
                #加载Preprocessor
                #print(f"root_path={self.conf['CONTROLNET']['root_path']}///filename={self.conf['CONTROLNET'][control_mode.lower()+'_filename']}")
                #self.control_obj[control_mode] = cls.from_pretrained(self.conf['CONTROLNET']['root_path'],
                #                    filename = self.conf['CONTROLNET'][control_mode.lower()+'_filename'])
                print("get_path")
                preprocessor_folder_path = self.conf['CONTROLNET'][control_mode.lower() + '_folder']
                print(f"Loading preprocessor [{control_mode}] from LOCAL folder: {preprocessor_folder_path}")


                self.control_obj[control_mode] = cls.from_pretrained(preprocessor_folder_path)

                print("getpath")
                controlnet_model_path = os.path.join(
                    self.conf['CONTROLNET']['root_path'],
                    self.conf['CONTROLNET'][control_mode.lower() + '_singlefile']
                    )
                
                print(f"Loading controlnet [{control_mode}] from LOCAL folder: {controlnet_model_path}")
                #加载ControlNet Model
                #self.control_pipeline.append(ControlNetModel.from_single_file(self.conf['CONTROLNET']['root_path']+self.conf['CONTROLNET'][control_mode.lower()+'_singlefile']))
                self.control_pipeline.append(ControlNetModel.from_single_file(controlnet_model_path))
                print("loop back")

        print("prepare for prompt")
        self.act_prompt_lists = []
        ne_prompt_lists = []
        #解析提示词
        for prompt_list in self.conf['PROMPT']:
            if prompt_list.startswith('ne')!=True:
                self.act_prompt_lists.append(self.conf['PROMPT'][prompt_list].split())
            else:
                ne_prompt_lists.append(self.conf['PROMPT'][prompt_list].split())
        self.ne_prompt = ','.join(ne_prompt_lists[0])
        
    #def generatorPic(self,input_image:np.ndarray|Image.Image,
    def generatorPic(self,input_image:Union[np.ndarray,Image.Image],
                     lora_style:str,
                     gen_steps:int = 40,
                     num_image_gen:int = 4,
                     gc_obj_token:str = 'a cat'):
        r'''
        在controlnet控制下生成图像
        @params input_image:提供一张类型为ndarry或者是PIL.Image.Image的图片,用来提供给controlnet提取信息
        @params lora_style:确定需要生成的lora风格,目前为单个lora,目前可以输入"手绘风"
        @params gen_steps:图像生成的步长,默认20步
        @params num_image_gen:生成的图像张数,默认4张图像
        @params gc_obj_token:要生成的对象token
        @return:         生成的PIL图片构成的list
        '''
        if gc_obj_token == 'a cat':
            self.input_tokens = [gc_obj_token]
        else:
            self.input_tokens = [self.token_dict[gc_obj_token]]

        img = Image.fromarray(np.uint8(input_image))

        det_res_dict = det_obj(input_image,
                               self.conf['YOLOV8']['root_path']+self.conf['YOLOV8']['yolov8_s'],
                               self.conf['YOLOV8']['det_obj'].split())
        
        assert len(det_res_dict)!=0,'检测不到要绘制的主体,请检查图片后重新上传,目前能检测的主体为80个coco数据集的分类!'

        clip_token = format_prompt(self.conf['CLIP']['root_path']+self.conf['CLIP']['rn50_filename'],self.act_prompt_lists,img)
        
        process_list = []
        for key in self.control_obj.keys():
            process = self.control_obj[key](img)
            process_list.append(process)

        p = Pinyin()
        print("\n\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        """
        if lora_style in self.conf['LORA']['lora_mode']:
            pinyin_lora_style = p.get_pinyin(lora_style,'')
            sdmodel_file = self.conf['STABLEDIFFUSION']['root_path']+self.conf['STABLEDIFFUSION'][pinyin_lora_style+'_basic']
        else:
            sdmodel_file = self.conf['STABLEDIFFUSION']['root_path']+self.conf['STABLEDIFFUSION']['define_basic']
            print('gen with no lora,use define model gen!')
        pipeline_with_control = StableDiffusionControlNetPipeline.from_single_file(sdmodel_file,
                                                                                   controlnet = self.control_pipeline,
                                                                                   torch_dtype=torch.float16,
                                                                                   scheduler_type = 'euler',
                                                                                   local_files_only = True
                                                                                   )
        """
        if lora_style in self.conf['LORA']['lora_mode']:
            pinyin_lora_style = p.get_pinyin(lora_style,'')
            relative_path = os.path.join(
                self.conf['STABLEDIFFUSION']['root_path'],
                self.conf['STABLEDIFFUSION'][pinyin_lora_style+'_basic']
            )
        else:
            relative_path = os.path.join(
                self.conf['STABLEDIFFUSION']['root_path'],
                self.conf['STABLEDIFFUSION']['define_basic']
            )
            print('gen with no lora,use define model gen!')

        # 2. 【【【核心解决方案：将相对路径转换为绝对路径】】】
        sdmodel_file = os.path.abspath(relative_path)

        # 3. (可选但推荐) 添加一个打印语句来调试，确保路径是正确的
        print(f"--- Loading SD model from absolute path: {sdmodel_file} ---")

        # 4. 使用这个绝对路径来加载 pipeline
        pipeline_with_control = StableDiffusionControlNetPipeline.from_single_file(sdmodel_file,
                                                                                controlnet = self.control_pipeline,
                                                                                #torch_dtype=torch.float16,
                                                                                scheduler_type = 'euler',
                                                                                local_files_only = True
                                                                           )
        #pipeline_with_control.to(dtype=torch.float16)
        if lora_style in self.conf['LORA']['lora_mode']:
            pinyin_lora_style = p.get_pinyin(lora_style,'')
            pipeline_with_control.load_lora_weights(self.conf['LORA']['root_path']+self.conf['LORA'][pinyin_lora_style+'_lora_file'])
            
            #pipeline_with_control.fuse_lora(self.conf['LORA'][pinyin_lora_style+'_weights'])
            lora_weight = float(self.conf['LORA'][pinyin_lora_style+'_weights'])

            self.input_tokens.insert(0,self.conf['LORA'][pinyin_lora_style+'_keywords'])
        else:
            print('gen with no lora,check your arg lora_style input!')

        #构建提示词
        self.input_tokens = ['best quality','masterpiece','detailed','illustration','8K']+self.input_tokens+['\(']+clip_token+['\)']
        final_prompt = ','.join(self.input_tokens)
        final_prompt = final_prompt.strip()
        print(final_prompt)

        pipeline_with_control.to("cuda")
        pipeline_with_control.unet.set_attn_processor(AttnProcessor2_0())
        """
        output = pipeline_with_control(final_prompt,
                                       image = process_list,
                                       num_inference_steps=gen_steps,
                                       controlnet_conditioning_scale=[0.7, 0.6],
                                       negative_prompt = self.ne_prompt,
                                       num_images_per_prompt=num_image_gen).images
        """
        pipeline_kwargs = {
            "prompt": final_prompt,
            "image": process_list,
            "num_inference_steps": gen_steps,
            "controlnet_conditioning_scale": [0.7, 0.6],
            "negative_prompt": self.ne_prompt,
            "num_images_per_prompt": num_image_gen
            }
        if lora_weight is not None:
            pipeline_kwargs["cross_attention_kwargs"] = {"scale": lora_weight}
        output = pipeline_with_control(**pipeline_kwargs).images

        self.output_token = copy.deepcopy(final_prompt)
        clip_token = None
        self.input_tokens = copy.deepcopy(self.o_input)
        final_prompt = None
        return output