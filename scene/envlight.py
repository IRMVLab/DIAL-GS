import torch
import nvdiffrast.torch as dr


class EnvLight(torch.nn.Module):

    def __init__(self, resolution=1024):
        super().__init__()
        self.to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")
        
        self.base = torch.nn.Parameter(
            0.5 * torch.ones(6, resolution, resolution, 3, requires_grad=True),
        )
        # 6*resolution*resolution*3
        # 6 faces, resolution x resolution pixels, 3 channels (RGB)
        
    def capture(self):
        return (
            self.base,
            self.optimizer.state_dict(),
        )
        
    def restore(self, model_args, training_args=None):
        self.base, opt_dict = model_args
        if training_args is not None:
            self.training_setup(training_args)
            self.optimizer.load_state_dict(opt_dict)
            
    def training_setup(self, training_args):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=training_args.envmap_lr, eps=1e-15)
        
    def forward(self, l):
        l = (l.reshape(-1, 3) @ self.to_opengl.T).reshape(*l.shape) # （B, H, W, 3）to OpenGL coordinate system
        # 表示每个像素对应的方向向量，相机光心指向像素平面上某像素的方向
        # 决定了从哪个方向采样环境光照
        l = l.contiguous()
        prefix = l.shape[:-1]
        if len(prefix) != 3:  # reshape to [B, H, W, 3]
            l = l.reshape(1, 1, -1, l.shape[-1])

        light = dr.texture(self.base[None, ...], l, filter_mode='linear', boundary_mode='cube')
        light = light.view(*prefix, -1)

        return light
