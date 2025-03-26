import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mamba.mamba_ssm import Mamba
from mamba.mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange


# パラメータを順方向と逆方向で共有しない。
class BidirectionalMamba(Mamba):  # Mambaを継承
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Mambaの初期化を再利用
        self.norm = nn.LayerNorm(self.d_model, device=self.out_proj.weight.device)  # LayerNorm追加

    def forward(self, hidden_states, inference_params=None):
        """
        Bidirectional Mambaのフォワードパス
        hidden_states: (B, L, D)
        """
        batch, seqlen, dim = hidden_states.shape

        #print("hidden_states: ", hidden_states.shape)
        hidden_states = self.norm(hidden_states)  # LayerNorm適用

        # キャッシュされた状態を取得
        conv_state_f, conv_state_b, ssm_state_f, ssm_state_b = None, None, None, None
        if inference_params is not None:
            conv_state_f, conv_state_b, ssm_state_f, ssm_state_b = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _, _, _ = self.step(hidden_states, conv_state_f, conv_state_b, ssm_state_f, ssm_state_b)
                return out

        # Linear Projection: X, Z に分割
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias, "d -> d 1")
        x, z = xz.chunk(2, dim=1)

        # print("x: ", x.shape)
        # print("z: ", z.shape)

        # **Forward方向**
        y_f = self._mamba_pass(x, conv_state_f, ssm_state_f, seqlen)

        # **Backward方向**
        x_b = x.flip(dims=(-1,))
        y_b = self._mamba_pass(x_b, conv_state_b, ssm_state_b, seqlen)
        y_b = y_b.flip(dims=(-1,))  # 再び元の順番に戻す

        # print("y_f: ", y_f.shape)
        # print("y_b: ", y_b.shape)

        # 結果の合成
        y = (y_f + y_b) * F.silu(z)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)

        return out + hidden_states

    def _mamba_pass(self, x, conv_state, ssm_state, seqlen):
        """Mambaの処理を共通化（順方向/逆方向の違いを吸収）"""
        if conv_state is not None:
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # 状態更新
        x = self.act(self.conv1d(x)[..., :seqlen])  # 1D畳み込み

        # SSM計算
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        y = selective_scan_fn(
            x, dt, A, B, C, self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )

        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        return y

    def step(self, hidden_states, conv_state_f, conv_state_b, ssm_state_f, ssm_state_b):
        """1トークンずつ処理するデコード用"""
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "デコード時は1トークンのみ処理可能"

        xz = self.in_proj(hidden_states.squeeze(1))
        x, z = xz.chunk(2, dim=-1)

        # **Forward**
        y_f = self._mamba_step(x, conv_state_f, ssm_state_f, dtype)

        # **Backward**
        x_b = x.flip(dims=(-1,))
        y_b = self._mamba_step(x_b, conv_state_b, ssm_state_b, dtype)
        y_b = y_b.flip(dims=(-1,))  # 順序を元に戻す

        # 結果の合成
        y = (y_f + y_b) * F.silu(z)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state_f, conv_state_b, ssm_state_f, ssm_state_b

    def _mamba_step(self, x, conv_state, ssm_state, dtype):
        """1ステップの処理"""
        if conv_state is not None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)

        x_db = self.x_proj(x)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)
        A = -torch.exp(self.A_log.float())

        dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
        dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
        dB = torch.einsum("bd,bn->bdn", dt, B)
        ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)

        y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
        return y + self.D.to(dtype) * x

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32

        conv_state_f = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        conv_state_b = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )

        ssm_state_f = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        ssm_state_b = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )

        return conv_state_f, conv_state_b, ssm_state_f, ssm_state_b

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state_f = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            conv_state_b = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state_f = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            ssm_state_b = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state_f, conv_state_b, ssm_state_f, ssm_state_b)
        else:
            conv_state_f, conv_state_b, ssm_state_f, ssm_state_b = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state_f.zero_()
                conv_state_b.zero_()
                ssm_state_f.zero_()
                ssm_state_b.zero_()

        return conv_state_f, conv_state_b, ssm_state_f, ssm_state_b