import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2

class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.UP = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2, bias=False)

    def forward(self, x):
        return self.UP(x)



def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


###########  Network Modules ###############
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class UpsampleConvLayer(torch.nn.Module):
    """
    Transpose convolution layer to upsample the feature maps
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    """
    Residual convolutional block for feature enhancement in decoder
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out




class LinearProj(nn.Module):
    """
    Linear projection used to reduce the number of channels and feature mixing.
    input_dim: number of channels of input features
    embed_dim: number of channels for output features
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(input_dim, embed_dim, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.proj(x)
        return x










class Fusion_Block(nn.Module):
    def __init__(self, in_channels):
        super(Fusion_Block, self).__init__()
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, int(in_channels // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(in_channels // 4)),
            nn.ReLU(inplace=True),
        )
        self.local_att = nn.Sequential(
            nn.MaxPool2d(1, padding=0, stride=1),
            nn.Conv2d(in_channels, int(in_channels // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(in_channels // 4)),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(in_channels * 2, int(in_channels // 4), kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.conv0 = nn.Conv2d(in_channels // 4, in_channels, 1, 1)
    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        f_dif = torch.abs((x1 - x2))
        # f_con = torch.cat((x1, x2), dim=1)
        f_add = x1 + x2
        f_g = self.global_att(f_add)
        f_l = self.local_att(f_dif)
        w = self.sigmoid(self.conv0(f_g+f_l))
        f = f_add * w + f_dif * (1 - w)
        return f

class Decoder(nn.Module):
    """
    Transformer Decoder
    """

    def __init__(self, in_channels=[32, 64, 128, 256], embedding_dim=64, output_nc=2, align_corners=True):
        super(Decoder, self).__init__()

        # settings
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # Channel reduction of feature maps before merging
        self.linear_c4 = LinearProj(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = LinearProj(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = LinearProj(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = LinearProj(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        # linear fusion layer to combine mult-scale features of all stages
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.embedding_dim * len(in_channels), out_channels=self.embedding_dim, kernel_size=1,
                      padding=0, stride=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        self.diff_c1 = Fusion_Block(in_channels=self.embedding_dim)
        self.diff_c2 = Fusion_Block(in_channels=self.embedding_dim)
        self.diff_c3 = Fusion_Block(in_channels=self.embedding_dim)
        self.diff_c4 = Fusion_Block(in_channels=self.embedding_dim)

        # Final predction head
        self.convd2x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.convd1x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)

        # Final activation
        self.active = nn.Sigmoid()

    def forward(self, inputs1, inputs2):
        # img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = inputs1  # len=4, 1/4, 1/8, 1/16, 1/32
        c1_2, c2_2, c3_2, c4_2 = inputs2  # len=4, 1/4, 1/8, 1/16, 1/32

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1)
        _c4_2 = self.linear_c4(c4_2)
        _c4 = self.diff_c4([_c4_1, _c4_2])
        _c4_up = resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1)
        _c3_2 = self.linear_c3(c3_2)
        _c3 = self.diff_c3([_c3_1, _c3_2])
        _c3_up = resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1)
        _c2_2 = self.linear_c2(c2_2)
        _c2 = self.diff_c2([_c2_1, _c2_2])
        _c2_up = resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1)
        _c1_2 = self.linear_c1(c1_2)
        _c1 = self.diff_c1([_c1_1, _c1_2])

        # Linear Fusion of difference image from all scales
        _c = self.linear_fuse(torch.cat([_c4_up, _c3_up, _c2_up, _c1], dim=1))

        # Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)
        # Residual block
        x = self.dense_2x(x)
        # Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        # Residual block
        x = self.dense_1x(x)

        # Final prediction
        cp = self.change_probability(x)

        outputs.append(cp)

        return outputs


# DA2Net
class DA2Net(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, depths=[3, 3, 4, 3], heads=[4, 4, 4, 4],
                 enc_channels=[144, 288, 576, 1152], decoder_softmax=False, dec_embed_dim=256, checkpoint_path=None):
        super(DA2Net, self).__init__()
        model_cfg = r"D:\program\CD\A_DA2-Net\sam2_configs\sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.embed_dims = enc_channels
        self.depths = depths
        self.embedding_dim = dec_embed_dim
        self.drop_path_rate = 0.1

        # decoder
        self.dec = Decoder(in_channels=self.embed_dims, embedding_dim=self.embedding_dim, output_nc=output_nc,
                           align_corners=False)

    def forward(self, x1, x2):
        # fx1, fx2 = [self.enc(x1), self.enc(x2)]

        x11, x12, x13, x14 = self.encoder(x1)
        x21, x22, x23, x24 = self.encoder(x2)
        fx1 = list()
        fx2 = list()
        fx1.append(x11)
        fx1.append(x12)
        fx1.append(x13)
        fx1.append(x14)
        fx2.append(x21)
        fx2.append(x22)
        fx2.append(x23)
        fx2.append(x24)
        change_map = self.dec(fx1, fx2)

        return change_map



######################################################################################
if __name__ == '__main__':
    model = DA2Net(input_nc=3, output_nc=2, depths=[3, 3, 4, 3], heads=[4, 4, 4, 4],
                    enc_channels=[144, 288, 576, 1152], decoder_softmax=False, dec_embed_dim=256).cuda()
    x1 = torch.ones(4, 3, 256, 256).cuda()
    x2 = torch.ones(4, 3, 256, 256).cuda()
    res = model(x1, x2)
    print(res[0].shape)
