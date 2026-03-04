#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from .gs_render import render_original_gs, render_gs_origin_wrapper, render_combined_gs, render_gs_combined_wrapper
from .pvg_render import render_pvg, render_pvg_wrapper

EPS = 1e-5

rendererFuncCallbacks = {
    "gs": render_original_gs,
    "pvg": render_pvg
}

rendererMergeCallbacks = {
    "gs": render_combined_gs,
    "pvg": None
}

renderWrapperTypeCallbacks = {
    "gs": render_gs_origin_wrapper,
    "pvg": render_pvg_wrapper,
}

rendererMergeWapperTypeCallbacks = {
    "gs": render_gs_combined_wrapper,
    "pvg": None
}




def get_renderer(render_type: str):
    return rendererFuncCallbacks[render_type], renderWrapperTypeCallbacks[render_type],\
        rendererMergeCallbacks[render_type], rendererMergeWapperTypeCallbacks[render_type]
