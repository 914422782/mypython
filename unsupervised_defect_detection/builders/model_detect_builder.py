from model.SegNet_Detect import SegNet_Detect
from model.UNet_Detect import UNet_Detect
from model.UNet_Mydetect import UNet_Mydetect
from model.cae_based_segnet import CAE

def build_model(model_name, input, output):
    if model_name == 'SegNet':
        return SegNet_Detect(input, output)
    elif model_name == 'UNet':
        return UNet_Detect(input, output)
    elif model_name == 'CAE':
        return CAE(input, output)
    elif model_name == 'UNet_Mydetect':
        return UNet_Mydetect(input, output)

