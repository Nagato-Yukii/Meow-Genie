from generator import GenPic,GenPicWithControlNet
from diffusers.utils import make_image_grid
from interface import GcInterface
from utils import det_obj,sendImg2next,bar_plot_fn,resizeImg
import cv2
import warnings
warnings.filterwarnings("ignore")
print("-----------------------------START------------------------------")
mygen1 = GenPicWithControlNet('./project.ini',['a cat'],'PidiNetDetector','LeresDetector')
print("-----------------------GenPicWithControlNet---------------------")
myinterface = GcInterface((20,40),5,(1,6),1)
myinterface.gcInterface(resizeImg,sendImg2next,bar_plot_fn,mygen1.generatorPic,mygen1.upScaleImg)
