import importlib
from models.unet import UNet
from models.spynet import SpyNet
from models.res_ednet import ResynthesisNet, ReanalysisNet
from models.flow_ednet import MvsynthesisNet, MvanalysisNet


def find_model_using_name(model_name):
    """Import the module "file_name.py".
    """
    a = importlib.import_module('models')
    for name, cls in a.__dict__.items():
        if name == model_name:
            return cls
