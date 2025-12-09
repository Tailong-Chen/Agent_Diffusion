import torch
import torch.nn as nn
from generative.networks.nets import DiffusionModelUNet

class DenoiserUNet(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        
        # UNet Configuration based on image size
        if args.img_size >= 1024:
            # 针对 1024x1024 的优化配置：增加下采样层数，减小 bottleneck 分辨率
            # 1024 -> 512 -> 256 -> 128 -> 64 -> 32
            num_channels = (32, 64, 128, 256, 512, 512) 
            attention_levels = (False, False, False, False, True, True)
            num_head_channels = 64
        elif args.img_size >= 512:
            num_channels = (128, 256, 512)
            attention_levels = (False, False, False)
            num_head_channels = 512
        else:
            # Optimized for 256x256
            # 256 -> 128 -> 64 -> 32
            num_channels = (128, 256, 512, 512)
            attention_levels = (False, False, False, True) # Only attention at 32x32
            num_head_channels = 64

        # Initialize MONAI UNet
        # Note: JiT uses in_channels=3, but for grayscale TIFFs we might want 1.
        # We'll check args.in_channels if it exists, otherwise default to 1 for UNet 
        # since we are replacing the model for what likely is grayscale data.
        # However, to be safe with existing data loaders that might output 3 channels,
        # we can check args.
        in_channels = getattr(args, 'in_channels', 1)
        out_channels = getattr(args, 'out_channels', 1)

        self.net = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            num_res_blocks=2,
            num_channels=num_channels,
            attention_levels=attention_levels,
            num_head_channels=num_head_channels,
            num_class_embeds=args.class_num if args.class_num > 1 else None,
        )
        
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels):
        # Ensure input is correct channel count
        if x.shape[1] != self.net.in_channels:
            # If input is 3 channel but model expects 1, take mean
            if x.shape[1] == 3 and self.net.in_channels == 1:
                x = x.mean(dim=1, keepdim=True)
            # If input is 1 channel but model expects 3, repeat
            elif x.shape[1] == 1 and self.net.in_channels == 3:
                x = x.repeat(1, 3, 1, 1)

        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        # Map t (0-1) to integer timesteps (0-999) for UNet
        t_unet = (t.flatten() * 999).long()
        
        # Forward pass through UNet
        # UNet expects (x, timesteps, context=None)
        # If using class embeddings, pass labels as context? 
        # MONAI DiffusionModelUNet uses 'class_labels' arg if num_class_embeds is set.
        if self.num_classes > 1:
            x_pred = self.net(x=z, timesteps=t_unet, class_labels=labels_dropped)
        else:
            x_pred = self.net(x=z, timesteps=t_unet)

        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    @torch.no_grad()
    def generate(self, labels, img_size=None):
        device = labels.device
        bsz = labels.size(0)
        
        # Use provided img_size or default to model's training size
        gen_size = img_size if img_size is not None else self.img_size
        
        # Initialize noise
        z = self.noise_scale * torch.randn(bsz, self.net.in_channels, gen_size, gen_size, device=device)
        
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        # Check if we need tiled sampling
        if z.shape[-1] > self.img_size or z.shape[-2] > self.img_size:
            return self._forward_sample_tiled(z, t, labels)

        # Map t to integer timesteps
        t_unet = (t.flatten() * 999).long()

        # conditional
        if self.num_classes > 1:
            x_cond = self.net(x=z, timesteps=t_unet, class_labels=labels)
        else:
            x_cond = self.net(x=z, timesteps=t_unet)
            
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        if self.num_classes > 1:
            x_uncond = self.net(x=z, timesteps=t_unet, class_labels=torch.full_like(labels, self.num_classes))
        else:
            x_uncond = x_cond # No unconditional if no classes

        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        # Note: If no classes, v_cond == v_uncond, so cfg doesn't do anything unless we implement null token for unconditional
        # For now assuming class_num > 1 or just ignoring cfg if class_num=1
        
        return v_cond, v_uncond

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_cond, v_uncond = self._forward_sample(z, t, labels)
        
        # CFG
        # t shape is (1, bsz, 1, 1, 1) or similar, we just need the scalar value
        t_val = t.flatten()[0].item()
        if self.cfg_interval[0] <= t_val <= self.cfg_interval[1]:
            v = v_uncond + self.cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond

        dt = t_next - t
        z_next = z + v * dt
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_cond, v_uncond = self._forward_sample(z, t, labels)
        
        t_val = t.flatten()[0].item()
        if self.cfg_interval[0] <= t_val <= self.cfg_interval[1]:
            v = v_uncond + self.cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond

        dt = t_next - t
        z_est = z + v * dt
        
        # Second step
        v_cond_next, v_uncond_next = self._forward_sample(z_est, t_next, labels)
        t_next_val = t_next.flatten()[0].item()
        if self.cfg_interval[0] <= t_next_val <= self.cfg_interval[1]:
            v_next = v_uncond_next + self.cfg_scale * (v_cond_next - v_uncond_next)
        else:
            v_next = v_cond_next
            
        z_next = z + (v + v_next) * dt / 2
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)

    def _forward_sample_tiled(self, z, t, labels):
        """
        Tiled forward pass for large images using overlap-tile strategy.
        """
        B, C, H, W = z.shape
        patch_size = self.img_size
        stride = patch_size // 2  # 50% overlap
        
        # Output buffers
        v_cond_out = torch.zeros_like(z)
        v_uncond_out = torch.zeros_like(z)
        count_map = torch.zeros((1, 1, H, W), device=z.device)
        
        # Generate patch coordinates
        h_starts = list(range(0, H - patch_size + 1, stride))
        if h_starts[-1] + patch_size < H:
            h_starts.append(H - patch_size)
            
        w_starts = list(range(0, W - patch_size + 1, stride))
        if w_starts[-1] + patch_size < W:
            w_starts.append(W - patch_size)
            
        patches = []
        coords = []
        
        # Extract all patches
        for h_idx in h_starts:
            for w_idx in w_starts:
                patch = z[:, :, h_idx:h_idx+patch_size, w_idx:w_idx+patch_size]
                patches.append(patch)
                coords.append((h_idx, w_idx))
        
        # Process patches in batches
        mini_batch_size = 8  # Adjust based on VRAM
        
        for i in range(0, len(patches), mini_batch_size):
            # Prepare batch
            batch_patches_list = patches[i:i+mini_batch_size]
            batch_patches = torch.cat(batch_patches_list, dim=0) # [k*B, C, p, p]
            k = len(batch_patches_list)
            
            # Expand t and labels
            # t is [B] or [1, B, ...]. We need [k*B]
            # We assume t is scalar-like for the whole image batch in standard sampling
            t_val = t.flatten()[0]
            t_batch = t_val.expand(batch_patches.shape[0])
            t_unet = (t_batch * 999).long()
            
            # labels is [B]. We need [L_b0, L_b1, ..., L_b0, L_b1]
            # batch_patches is ordered: Patch0(all B), Patch1(all B)...
            labels_batch = labels.repeat(k)
            
            # Forward pass
            if self.num_classes > 1:
                x_cond = self.net(x=batch_patches, timesteps=t_unet, class_labels=labels_batch)
            else:
                x_cond = self.net(x=batch_patches, timesteps=t_unet)
                
            v_cond_p = (x_cond - batch_patches) / (1.0 - t_batch.view(-1, 1, 1, 1)).clamp_min(self.t_eps)
            
            if self.num_classes > 1:
                x_uncond = self.net(x=batch_patches, timesteps=t_unet, class_labels=torch.full_like(labels_batch, self.num_classes))
            else:
                x_uncond = x_cond
            
            v_uncond_p = (x_uncond - batch_patches) / (1.0 - t_batch.view(-1, 1, 1, 1)).clamp_min(self.t_eps)
            
            # Accumulate results
            v_cond_chunks = torch.chunk(v_cond_p, k, dim=0)
            v_uncond_chunks = torch.chunk(v_uncond_p, k, dim=0)
            
            current_coords = coords[i:i+mini_batch_size]
            
            for j, (h, w) in enumerate(current_coords):
                v_cond_out[:, :, h:h+patch_size, w:w+patch_size] += v_cond_chunks[j]
                v_uncond_out[:, :, h:h+patch_size, w:w+patch_size] += v_uncond_chunks[j]
                count_map[:, :, h:h+patch_size, w:w+patch_size] += 1.0
                
        # Average
        v_cond_out /= count_map
        v_uncond_out /= count_map
        
        return v_cond_out, v_uncond_out
