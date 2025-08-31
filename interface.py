import gradio as gr
class GcInterface():
    def __init__(self,epoch_num:tuple,epoch_step:int,gc_num:tuple,gc_step:int,gc_col:int = 2) -> None:
        self.epoch_num_min = epoch_num[0]
        self.epoch_num_max = epoch_num[1]
        self.epoch_step = epoch_step
        self.gc_num_min = gc_num[0]
        self.gc_num_max = gc_num[1]
        self.gc_step = gc_step
        self.gc_col = gc_col
        self.select_index = None

    def gcInterface(self,*func1):
        with gr.Blocks() as iface:
            gr.Markdown("""#喵精灵 #一键生成猫咪形象！""")
            with gr.Tab("识别猫咪品种"):
                with gr.Row():
                    det_input = gr.Image()
                    det_output = gr.BarPlot(scale=1.5)
                det_button = gr.Button("按下识别猫咪种类  可识别9个常见品种")
            with gr.Tab("生成猫咪虚拟形象"):
                with gr.Row():
                    gc_input = gr.Image()
                    gc_output = gr.Gallery(columns=self.gc_col)
                    with gr.Column():
                        cls_pet = gr.Radio(["布偶猫","狸花猫","金吉拉","英国短毛猫","其它"],show_label="点击选择你要生成的猫咪")
                        num_gc = gr.Slider(self.gc_num_min, self.gc_num_max,step=self.gc_step,label="通过滑块调整虚拟形象数")
                        epoch_gc = gr.Slider(self.epoch_num_min, self.epoch_num_max,step=self.epoch_step,label="通过滑块调整迭代次数")
                        choose_gc = gr.Radio(["手绘风", "科技风", "中国风", "油画风"],
                                            show_label='通过点击选择想要的风格')
                gc_button = gr.Button("按下生成猫咪虚拟形象")
                with gr.Row():
                    up_img_input = gr.Image()
                    up_img_output = gr.Image()
                up_button = gr.Button('按下生成高清图像')
            det_input.upload(func1[0], inputs=det_input,outputs=det_input)
            gc_input.upload(func1[0], inputs=gc_input,outputs=gc_input)
            gc_output.select(func1[1], None, up_img_input)
            det_button.click(func1[2], inputs=det_input,outputs=det_output)
            gc_button.click(func1[3], inputs=[gc_input,choose_gc,epoch_gc,num_gc,cls_pet], outputs=gc_output,scroll_to_output = True)
            up_button.click(func1[4], inputs=up_img_input, outputs=up_img_output,scroll_to_output = True)
        #iface = gr.Interface(fn=gen_pipeline, inputs="image", outputs=["image","image"],title="喵精灵    一键生成个性化的猫咪虚拟形象！")  
        iface.launch(share=True)







