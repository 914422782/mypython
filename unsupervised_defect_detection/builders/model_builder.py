
from model.SegNet import SegNet
from model.UNet import UNet



def build_model(model_name, num_classes):
    if model_name == 'SegNet':
        return SegNet(classes=num_classes)
    elif model_name == 'UNet':
        return UNet(classes=num_classes)

