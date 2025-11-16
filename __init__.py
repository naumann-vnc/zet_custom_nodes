from .nodes.nodes import *

NODE_CLASS_MAPPINGS = {
        "Laplacian": LaplacianFilter,
        "Tag Blacklist": TagBlacklist,
        "Hierarchical Merging of Region Boundary Region Adjacency Graphs": HierarchicalMergingofRegionBoundaryRAGs,
        "SkinToneMask": SkinToneMask,
        #"G'MIC Easy Skin Retouch": GMICEasySkinRetouch,
        "Blend Mask Batch": BlendMaskBatch,
        "Merge Images": MergeImages,
        "Resize Image Targeting Aspect Ratio": ResizeImageTargetingAspectRatio
    }

print("\033[34mComfyUI NedZet Nodes: \033[92mLoaded\033[0m")
