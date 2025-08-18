import torch

class CABus:
    """
    Toroidal cellular automaton "bus" with multi-channel state.
    - H x W cells, C channels.
    - One ganglion tick: advance a selected row (shift-register style) and CA update.
    - Wrap-around at edges.
    - Crosstalk: controlled stochastic bleed between channels.
    """
    def __init__(self, height=16, width=16, channels=32, bleed=0.02, device=None):
        self.H = height
        self.W = width
        self.C = channels
        self.bleed = bleed
        self.device = device or torch.device("cpu")
        self.state = torch.zeros(self.C, self.H, self.W, device=self.device)

        # Neighborhood kernel (Moore) for CA update
        kernel = torch.ones(1, 1, 3, 3, device=self.device)
        kernel[0,0,1,1] = 0.0  # exclude center for neighbor sum
        self.kernel = kernel

    def reset(self):
        self.state.zero_()

    def inject(self, channel_idx: int, tensor_2d: torch.Tensor, y: int, x: int):
        """
        Writes a tensor into channel slice, wrapping if needed.
        """
        h, w = tensor_2d.shape[-2], tensor_2d.shape[-1]
        for i in range(h):
            for j in range(w):
                yy = (y + i) % self.H
                xx = (x + j) % self.W
                self.state[channel_idx, yy, xx] = tensor_2d[i, j]

    def read_patch(self, y: int, x: int, h: int, w: int) -> torch.Tensor:
        out = torch.zeros(self.C, h, w, device=self.device)
        for i in range(h):
            for j in range(w):
                yy = (y + i) % self.H
                xx = (x + j) % self.W
                out[:, i, j] = self.state[:, yy, xx]
        return out

    def _toroidal_pad(self, x: torch.Tensor, pad=1) -> torch.Tensor:
        return torch.concat([x[..., -pad:], x, x[..., :pad]], dim=-1)

    def _toroidal_pad2(self, x: torch.Tensor, pad=1) -> torch.Tensor:
        x = self._toroidal_pad(x, pad)
        x = torch.concat([x[:, :, -pad:, :], x, x[:, :, :pad, :]], dim=2)
        return x

    def _ca_update(self):
        # Per-channel local rule: next = sigmoid(alpha*state + beta*neighbor_sum)
        # Same alpha/beta for simplicity; could be learnable later.
        x = self.state.unsqueeze(0)  # (1,C,H,W)
        pad = self._toroidal_pad2(x, pad=1)
        # depthwise conv via grouping trick
        # Build grouped kernel for all channels:
        k = self.kernel.repeat(self.C, 1, 1, 1)  # (C,1,3,3)
        out = torch.nn.functional.conv2d(pad, k, groups=self.C)
        alpha = 0.9
        beta = 0.15
        nxt = torch.sigmoid(alpha * self.state + beta * out.squeeze(0))
        self.state = nxt

    def _shift_register_row(self, row_idx: int):
        # Shift a row to the right by 1 with wrap-around (for all channels)
        row = self.state[:, row_idx, :]
        self.state[:, row_idx, :] = torch.roll(row, shifts=1, dims=-1)

    def _crosstalk(self):
        if self.bleed <= 0.0:
            return
        # Simple linear bleed: each channel receives a tiny fraction of the mean of others
        mean_other = (self.state.sum(dim=0, keepdim=True) - self.state) / max(self.C - 1, 1)
        self.state = (1.0 - self.bleed) * self.state + self.bleed * mean_other

    def tick(self, row_to_shift: int):
        self._shift_register_row(row_to_shift % self.H)
        self._ca_update()
        self._crosstalk()
