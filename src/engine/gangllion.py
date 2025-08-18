import torch
from .ca_bus import CABus

class Ganglion:
    """
    Global clock + signal hub.
    - Ticks time.
    - Steers data into/out of the CA bus.
    - Provides synchronized sampling for the three nets.
    """
    def __init__(self, ca_height=16, ca_width=16, ca_channels=32, bleed=0.02, device=None):
        self.t = 0
        self.device = device or torch.device("cpu")
        self.bus = CABus(ca_height, ca_width, ca_channels, bleed, device=self.device)

    def reset(self):
        self.t = 0
        self.bus.reset()

    def tick(self):
        self.bus.tick(row_to_shift=self.t % self.bus.H)
        self.t += 1

    def write_signal(self, channel_idx: int, tensor2d: torch.Tensor, y=0, x=0):
        self.bus.inject(channel_idx, tensor2d, y, x)

    def read_signal(self, y=0, x=0, h=8, w=8) -> torch.Tensor:
        return self.bus.read_patch(y, x, h, w)

    def bleed_level(self) -> float:
        return self.bus.bleed
