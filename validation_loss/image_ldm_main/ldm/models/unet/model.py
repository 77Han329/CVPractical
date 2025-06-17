import torch
import torch.nn as nn

from layers.residual import ResBlock
from layers.residual import zero_module
from layers.residual import TimestepBlock
from layers.pooling import Pool2d, UnPool2d
from layers.t_emb import timestep_embedding
from layers.resize import Upsample, Downsample
from layers.attention import SpatialSelfAttention


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# adapted from https://github.com/facebookresearch/DiT/blob/main/models.py
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
    

class EfficientUNet(nn.Module):
    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 3, 4),
            conv_resample=True,
            dim_head=64,
            num_heads=4,
            use_linear_attn=True,
            use_scale_shift_norm=True,
            pool_factor=-1,
            num_classes=0,
            class_dropout_prob=0.0
    ):
        """
        2D UNet model with attention. It includes down- and up-
        sampling blocks to train an end-to-end high resolution
        diffusion model.

        Args:
            in_channels: channels in the input Tensor.
            model_channels: base channel count for the model.
            out_channels: channels in the output Tensor.
            num_res_blocks: number of residual blocks per downsample.
            attention_resolutions: a collection of downsample rates at which
                attention will take place. Maybe a set, list, or tuple.
                For example, if this contains 4, then at 4x downsampling, attention
                will be used.
            dropout: the dropout probability.
            channel_mult: channel multiplier for each level of the UNet.
            conv_resample: if True, use learned convolutions for upsampling and
                downsampling.
            num_heads: the number of attention heads in each attention layer.
            dim_head: the dimension of each attention head.
            use_linear_attn: If true, applies linear attention in the encoder/decoder.
            use_scale_shift_norm: If True, use ScaleShiftNorm instead of LayerNorm.
            pool_factor: Down-sampling factor for spatial dimensions (w. conv layer)
            num_classes: number of classes for conditioning. If 0, unconditional training.
            class_dropout_prob: probability of dropping out class labels for classifier-free guidance.
        """
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.pool_factor = pool_factor

        # timestep embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # class conditioning
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        if self.num_classes > 0:
            self.y_embedder = LabelEmbedder(num_classes, time_embed_dim, class_dropout_prob)
            uc = " + unconditional" if class_dropout_prob > 0 else ""
            print(f"[EfficientUNet] Class-conditional ({self.num_classes} classes{uc})")
        else:
            self.y_embedder = None

        # if ds factors > 1, it will down-sample the input, otherwise its just identity
        if pool_factor > 1:
            self.pool = Pool2d(in_channels, model_channels, pool_factor=pool_factor)
            starting_channels = model_channels
        else:
            self.pool = nn.Identity()
            starting_channels = in_channels

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                nn.Conv2d(starting_channels, model_channels, 3, padding=1)
            )
        ])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        out_channels=mult * model_channels,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        SpatialSelfAttention(
                            dim=ch,
                            heads=num_heads,
                            dim_head=dim_head,
                            use_linear=use_linear_attn,
                            use_efficient_attn=True
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm
            ),
            SpatialSelfAttention(
                ch,
                heads=num_heads,
                dim_head=dim_head,
                use_linear=False,
                use_efficient_attn=True,
            ),
            ResBlock(
                ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        emb_channels=time_embed_dim,
                        out_channels=model_channels * mult,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        SpatialSelfAttention(
                            ch,
                            heads=num_heads,
                            dim_head=dim_head,
                            use_linear=use_linear_attn,
                            use_efficient_attn=True,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # check whether we first down-sampled the input
        if pool_factor > 1:
            self.out = nn.Sequential(
                nn.GroupNorm(32, ch),
                nn.SiLU(),
                nn.Conv2d(model_channels, model_channels, 3, padding=1)
            )
            self.un_pool = UnPool2d(model_channels, out_channels, pool_factor=pool_factor)
        else:
            self.out = nn.Sequential(
                nn.GroupNorm(32, ch),
                nn.SiLU(),
                zero_module(
                    nn.Conv2d(model_channels, out_channels, 3, padding=1))
            )
            self.un_pool = nn.Identity()

    def forward(self, x, t, context=None, y=None):
        """
        Apply the model to an input batch.

        Args:
            x: an [N x C x ...] Tensor of inputs.
            t: a 1-D batch of timesteps.
            context: an optional [N x C x ...] Tensor of context that gets
                concatenated to the inputs.
            y: (N,) tensor of class labels
        Returns:
            an [N x C x ...] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(t, self.model_channels))

        if context is not None:
            x = torch.cat([x, context], dim=1)

        if self.y_embedder is not None:
            assert y is not None, "Class labels missing!"
            y = self.y_embedder(y, self.training)
            emb = emb + y

        x = self.pool(x)

        hs = []
        h = x

        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)

        h = self.out(h)
        h = self.un_pool(h)

        return h


if __name__ == "__main__":
    class_conditional = False
    concat_context = False

    unet = EfficientUNet(
        in_channels=(6 if concat_context else 3),
        model_channels=64,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=[4],
        dropout=0,
        channel_mult=(1, 2, 4),
        conv_resample=True,
        dim_head=64,
        num_heads=4,
        use_linear_attn=True,
        use_scale_shift_norm=True,
        pool_factor=1,
        num_classes=(10 if class_conditional else 0),
        class_dropout_prob=0.1
    )
    print(f"Params: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}")

    ipt = torch.randn((2, 3, 64, 64))
    cont = torch.randn((2, 3, 64, 64)) if concat_context else None
    t_ = torch.rand((2,))
    y_ = torch.randint(0, 10, (2,)) if class_conditional else None
    out = unet(ipt, t_, cont, y_)
    print("Input:", ipt.shape)                      # (bs, c, h, w)
    print("Output:", out.shape)                     # (bs, c, h, w)
