import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from .attn_blocks import SEBlock, CBAModule
from .swin_transformer import PatchEmbed, BasicLayer, PatchMerging


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv_res = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x) + self.conv_res(x)


class ConvEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvEncoder, self).__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x1 = self.conv(x)
        out = self.down(x1)
        return out, x1


class SameConvEncoders(nn.Module):
    def __init__(self, in_chans, out_chans=[16, 32, 64, 128, 256, 512]):
        super(SameConvEncoders, self).__init__()
        self.in_chans = in_chans
        self.conv1 = ConvEncoder(in_chans, out_chans[0])
        self.conv2 = ConvEncoder(out_chans[0], out_chans[1])
        self.conv3 = ConvEncoder(out_chans[1], out_chans[2])
        self.conv4 = ConvEncoder(out_chans[2], out_chans[3])
        self.conv5 = ConvEncoder(out_chans[3], out_chans[4])
        self.conv6 = ConvEncoder(out_chans[4], out_chans[5])

    def forward(self, x):
        if x.size()[1] != self.in_chans:
            x = x.repeat(1, self.in_chans, 1, 1)
        x, x1 = self.conv1(x)
        x, x2 = self.conv2(x)
        x, x3 = self.conv3(x)
        x, x4 = self.conv4(x)
        x, x5 = self.conv5(x)
        x, x6 = self.conv6(x)
        return x1, x2, x3, x4, x5, x6


class SwinEncoders(nn.Module):
    """
    Swin-Transformer Encoder Branch
    Input: 224 x 224 x 3; Output: (7 x 7) x 768
    Swin-Transformer: 2, 2, 6, 2; DownSample: PatchMerging
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, window_size=7,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], norm_layer=nn.LayerNorm,
                 drop_rate=0., mlp_ratio=4., drop_path_rate=0.1, attn_drop_rate=0., patch_norm=True,
                 qkv_bias=True, qk_scale=None, use_checkpoint=False, fused_window_process=False, ape=False):
        super(SwinEncoders, self).__init__()

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution

        # absolute position embedding
        self.ape = ape
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # swin-transformer encoder
        self.in_chans = in_chans
        self.layers = nn.ModuleList()
        num_layers = len(depths)
        for i_layer in range(num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                norm_layer=norm_layer, downsample=PatchMerging if (i_layer<num_layers-1) else None,
                use_checkpoint=use_checkpoint, fused_window_process=fused_window_process)
            self.layers.append(layer)

        # other functions
        self.num_features = int(embed_dim * 2 ** (num_layers - 1))
        self.norm = norm_layer(self.num_features)

        # initial weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        if x.size()[1] != self.in_chans:
            x = x.repeat(1, self.in_chans, 1, 1)
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_trans = []
        for layer in self.layers:
            x_trans.append(x)
            x = layer(x)
        x = self.norm(x)
        return x_trans

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


class SameConvUnifyConvSamp(nn.Module):
    """
    Same Scale Features in same layer captured by Convolution
    Implement: SwinTrans+SE, CNN+CBAM, attention enhancement
    """
    def __init__(self, layer, conv_chans=[64, 128, 256, 512], trans_chans=[96, 192, 384, 768]):
        super(SameConvUnifyConvSamp, self).__init__()
        self.channels = conv_chans[layer - 1] + trans_chans[layer - 1]
        self.SEBlock = SEBlock(trans_chans[layer - 1])
        self.CBAModule = CBAModule(conv_chans[layer - 1])
        self.features = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, trans_chans[layer - 1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(trans_chans[layer - 1]),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(self.channels, trans_chans[layer - 1], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(trans_chans[layer - 1]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_conv, x_trans):
        # resolution/size unify
        B, C, H, W = x_conv.shape
        x_trans = x_trans.reshape(B, -1, H, W)
        x_trans = self.SEBlock(x_trans)
        x_conv = self.CBAModule(x_conv)

        # template setting as convolution and residual connection
        x = torch.cat([x_conv, x_trans], dim=1)
        out = self.features(x)
        out = torch.add(out, self.residual(x))
        return out


class SameConvUnisConvSamp(nn.Module):
    def __init__(self):
        super(SameConvUnisConvSamp, self).__init__()
        self.multiConvUni1 = SameConvUnifyConvSamp(layer=1)
        self.multiConvUni2 = SameConvUnifyConvSamp(layer=2)
        self.multiConvUni3 = SameConvUnifyConvSamp(layer=3)
        self.multiConvUni4 = SameConvUnifyConvSamp(layer=4)

    def forward(self, c1, c2, c3, c4, x_trans):
        t1, t2, t3, t4 = x_trans[0], x_trans[1], x_trans[2], x_trans[3]
        s1 = self.multiConvUni1(c1, t1)
        s2 = self.multiConvUni2(c2, t2)
        s3 = self.multiConvUni3(c3, t3)
        s4 = self.multiConvUni4(c4, t4)
        return s1, s2, s3, s4


class FinalSkipConnection(nn.Module):
    """
    UpSample all SwinTrans to same resolution as CNN Features,
    concat five feature maps, followed by residual block and CBAM spatial attention
    """
    def __init__(self, conv_chans=[16, 32], trans_chans=[96, 192, 384, 768]):
        super(FinalSkipConnection, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.CBAModule1 = CBAModule(conv_chans[1])
        self.CBAModule0 = CBAModule(conv_chans[0])
        self.chans = [conv_chans[0] + sum(trans_chans), conv_chans[1] + sum(trans_chans)]
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.chans[1], conv_chans[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_chans[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_chans[1], conv_chans[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_chans[1]),
            nn.ReLU(inplace=True),
        )
        self.resd1 = nn.Sequential(
            nn.Conv2d(conv_chans[1], conv_chans[1], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(conv_chans[1]),
            nn.ReLU(inplace=True),
        )
        self.conv0 = nn.Sequential(
            nn.Conv2d(self.chans[0], conv_chans[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_chans[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_chans[0], conv_chans[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_chans[0]),
            nn.ReLU(inplace=True),
        )
        self.resd0 = nn.Sequential(
            nn.Conv2d(conv_chans[0], conv_chans[0], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(conv_chans[0]),
            nn.ReLU(inplace=True),
        )

    def forward(self, c1, c2, x_trans):
        B, C, H, W = c1.shape
        t1, t2, t3, t4 = x_trans[0], x_trans[1], x_trans[2], x_trans[3]
        t1, t2 = t1.reshape(B, -1, H // 4, W // 4), t2.reshape(B, -1, H // 8, W // 8)
        t3, t4 = t3.reshape(B, -1, H // 16, W // 16), t4.reshape(B, -1, H // 32, W // 32)
        t11, t12 = self.up(t1), self.up(self.up(t2)),
        t13, t14 = self.up(self.up(self.up(t3))), self.up(self.up(self.up(self.up(t4))))
        s1 = self.CBAModule1(self.conv1(torch.cat([c2, t11, t12, t13, t14], dim=1)) + self.resd1(c2))
        t21, t22, t23, t24 = self.up(t11), self.up(t12), self.up(t13), self.up(t14)
        s2 = self.CBAModule0(self.conv0(torch.cat([c1, t21, t22, t23, t24], dim=1)) + self.resd0(c1))
        return s1, s2


class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.upSample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x_skip, x):
        out = self.upSample(x)
        out = torch.cat([x_skip, out], dim=1)
        out = self.conv(out)
        return out


class SameConvDecoders(nn.Module):
    def __init__(self, trans_chans=[96, 192, 384, 768]):
        super(SameConvDecoders, self).__init__()
        self.chans = trans_chans
        self.decoder1 = Decoder(self.chans[0] + self.chans[1], self.chans[0])
        self.decoder2 = Decoder(self.chans[1] + self.chans[2], self.chans[1])
        self.decoder3 = Decoder(self.chans[2] + self.chans[3], self.chans[2])

    def forward(self, x1, x2, x3, x4):
        out = self.decoder3(x3, x4)
        out = self.decoder2(x2, out)
        out = self.decoder1(x1, out)
        return out


class DuEnNet(nn.Module):
    def __init__(self, in_chans, num_classes, conv_chan=16, trans_chan=96):
        super(DuEnNet, self).__init__()
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.swinEncoder = SwinEncoders(in_chans=self.in_chans)
        self.convEncoder = SameConvEncoders(in_chans=self.in_chans)
        self.multiSkipConnect = SameConvUnisConvSamp()
        self.decoder = SameConvDecoders()
        self.finalSkipConnect = FinalSkipConnection()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_chans = [trans_chan + conv_chan * 2, (trans_chan + conv_chan * 2) // 2 + conv_chan]
        self.up1_conv = ConvBlock(self.up_chans[0], self.up_chans[0] // 2)
        self.up0_conv = ConvBlock(self.up_chans[1], self.up_chans[1] // 2)
        self.output = nn.Conv2d(in_channels=self.up_chans[1] // 2, out_channels=self.num_classes, kernel_size=1,
                                bias=False)

    def forward(self, x):
        B, _, H, W = x.shape
        x_trans = self.swinEncoder(x)
        # print(x_trans[0].shape, x_trans[1].shape, x_trans[2].shape, x_trans[3].shape)
        conv1, conv2, conv3, conv4, conv5, conv6 = self.convEncoder(x)
        # print(conv1.shape, conv2.shape, conv3.shape, conv4.shape)
        skip1, skip2, skip3, skip4 = self.multiSkipConnect(conv3, conv4, conv5, conv6, x_trans)
        # print(skip1.shape, skip2.shape, skip3.shape, skip4.shape)
        out = self.decoder(skip1, skip2, skip3, skip4)
        out = out.reshape(B, -1, H // 4, W // 4)
        m1, m2 = self.finalSkipConnect(conv1, conv2, x_trans)
        out = self.up1_conv(torch.cat([m1, self.up(out)], dim=1))
        out = self.up0_conv(torch.cat([m2, self.up(out)], dim=1))
        out = self.output(out)
        if self.num_classes == 1:
            out = torch.sigmoid(out)
        return out
