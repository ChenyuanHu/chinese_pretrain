import torch
import math
import typing

class RoPE:
    def __init__(self, dim: int, theta: float = 10000.0, use_scaled: bool = False):
        # 不再预计算block_size相关参数
        self.dim = dim
        self.theta = theta
        self.use_scaled = use_scaled
        # 预计算频率基底（与序列长度无关）
        self.base_freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2)[: (dim//2)].float() / dim))
        if self.use_scaled:
            self.base_freqs = self.apply_scaling(self.base_freqs)

    def apply_rotary_emb_warp(self, xq: torch.Tensor, xk: torch.Tensor):
        """动态生成当前序列长度的位置编码"""
        B, T, H, D = xq.shape  # 假设输入形状 [B, T, H, D]
        device = xq.device
        
        # 这里可以考虑使用lru_cache缓存频率基底
        # from functools import lru_cache
        # class RoPE:
        #     @lru_cache(maxsize=32)
        #     def get_freqs_cis(self, T: int, device):
        #         t = torch.arange(T, dtype=torch.float32, device=device)
        #         return torch.polar(torch.ones_like(t), t[:, None] * self.base_freqs.to(device))

        # 根据当前序列长度生成t
        t = torch.arange(T, dtype=torch.float32, device=device)
        freqs = torch.outer(t, self.base_freqs.to(device))
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        
        return self.apply_rotary_emb(xq, xk, freqs_cis)

    @staticmethod
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        """删除原断言，适配动态形状"""
        ndim = x.ndim
        shape = [1] * ndim
        shape[1] = freqs_cis.size(0)  # 序列维度
        shape[-1] = freqs_cis.size(1) # 特征维度
        return freqs_cis.view(*shape)

    @staticmethod
    def apply_scaling(freqs: torch.Tensor):
        # Values obtained from grid search
        scale_factor = 8
        low_freq_factor = 1
        high_freq_factor = 4
        old_context_len = 8192  # original llama3 length

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
        return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

    @staticmethod
    def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = RoPE.reshape_for_broadcast(freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)