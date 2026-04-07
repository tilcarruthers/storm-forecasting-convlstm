import torch
from storm_forecasting.models.seq2seq_unet import ConvLSTMUNetSeq2Seq


def test_model_forward_shape() -> None:
    torch.set_num_threads(1)
    model = ConvLSTMUNetSeq2Seq(base=4, bottleneck_ch=8, tin=2, tout=2)
    model.eval()
    x = torch.randn(1, 2, 1, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 2, 1, 32, 32)
