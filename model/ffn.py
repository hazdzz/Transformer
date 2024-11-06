import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


class FeedForward(nn.Module):
    def __init__(self, feat_dim, hid_dim, ffn_drop_prob: float = 0.1) -> None:
        super(FeedForward, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_features=feat_dim, out_features=hid_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p=ffn_drop_prob),
            nn.Linear(in_features=hid_dim, out_features=feat_dim, bias=True)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_normal_(self.ffn[0].weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        init.kaiming_normal_(self.ffn[3].weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        init.zeros_(self.ffn[0].bias)
        init.zeros_(self.ffn[3].bias)

    def forward(self, input: Tensor) -> Tensor:
        return self.ffn(input)