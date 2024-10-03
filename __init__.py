from .nodes.nodes import *

NODE_CLASS_MAPPINGS = { 
        "Print Hello World": PrintHelloWorld,
        "Resize Image Targeting Aspect Ratio": ResizeImageTargetingAspectRatio
    }
    
print("\033[34mComfyUI NedZet Nodes: \033[92mLoaded\033[0m")

#z-stack node to be added