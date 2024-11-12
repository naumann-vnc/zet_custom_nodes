from .nodes.nodes import *

NODE_CLASS_MAPPINGS = { 
        "Laplacian": LaplacianFilter,
        "Tag blacklist": TagBlacklist,
        #"G'MIC Segmentation": GMICSegmentation,
        #"G'MIC Easy Skin Retouch": GMICEasySkinRetouch,
        "Blend Mask Batch": BlendMaskBatch,
        "Resize Image Targeting Aspect Ratio": ResizeImageTargetingAspectRatio
    }
    
print("\033[34mComfyUI NedZet Nodes: \033[92mLoaded\033[0m")

#z-stack node to be added