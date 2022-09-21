# Copyright 2021 Tomoki Hayashi
# MIT License (https://opensource.org/licenses/MIT)
# Adapted by Florian Lux 2021

# This code is based on https://github.com/jik876/hifi-gan.


import copy

import torch
import torch.nn.functional as F
from torch.nn import Conv1d

from Layers.ResidualBlock import HiFiGANResidualBlock as ResidualBlock
from TrainingInterfaces.Spectrogram_to_Wave.Avocodo.AvocodoDiscriminators import MultiCoMBDiscriminator
from TrainingInterfaces.Spectrogram_to_Wave.Avocodo.AvocodoDiscriminators import MultiSubBandDiscriminator


class HiFiGANGenerator(torch.nn.Module):

    def __init__(self,
                 in_channels=80,
                 out_channels=1,
                 channels=512,
                 kernel_size=7,
                 upsample_scales=(8, 6, 4, 4),  # CAREFUL: Avocodo assumes that there are always 4 upsample scales, because it takes intermediate results.
                 upsample_kernel_sizes=(16, 12, 8, 8),
                 resblock_kernel_sizes=(3, 7, 11),
                 resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
                 use_additional_convs=True,
                 bias=True,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.1}):
        """
        Initialize HiFiGANGenerator module.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_scales (list): List of upsampling scales.
            upsample_kernel_sizes (list): List of kernel sizes for upsampling layers.
            resblock_kernel_sizes (list): List of kernel sizes for residual blocks.
            resblock_dilations (list): List of dilation list for residual blocks.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
        """
        super().__init__()

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)

        # define modules
        self.num_upsamples = len(upsample_kernel_sizes)
        self.num_blocks = len(resblock_kernel_sizes)
        self.input_conv = torch.nn.Conv1d(in_channels,
                                          channels,
                                          kernel_size,
                                          1,
                                          padding=(kernel_size - 1) // 2, )
        self.upsamples = torch.nn.ModuleList()
        self.blocks = torch.nn.ModuleList()
        for i in range(len(upsample_kernel_sizes)):
            self.upsamples += [torch.nn.Sequential(getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                                                   torch.nn.ConvTranspose1d(channels // (2 ** i),
                                                                            channels // (2 ** (i + 1)),
                                                                            upsample_kernel_sizes[i],
                                                                            upsample_scales[i],
                                                                            padding=(upsample_kernel_sizes[i] - upsample_scales[i]) // 2, ), )]
            for j in range(len(resblock_kernel_sizes)):
                self.blocks += [ResidualBlock(kernel_size=resblock_kernel_sizes[j],
                                              channels=channels // (2 ** (i + 1)),
                                              dilations=resblock_dilations[j],
                                              bias=bias,
                                              use_additional_convs=use_additional_convs,
                                              nonlinear_activation=nonlinear_activation,
                                              nonlinear_activation_params=nonlinear_activation_params, )]
        self.output_conv = torch.nn.Sequential(
            # NOTE(kan-bayashi): follow official implementation but why
            #   using different slope parameter here? (0.1 vs. 0.01)
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(channels // (2 ** (i + 1)),
                            out_channels,
                            kernel_size,
                            1,
                            padding=(kernel_size - 1) // 2, ), torch.nn.Tanh(), )

        self.out_proj_x1 = Conv1d(512 // 4, 1, 7, 1, padding=3)
        self.out_proj_x2 = Conv1d(512 // 8, 1, 7, 1, padding=3)

        # apply weight norm
        self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, c):
        """
        Calculate forward propagation.
        
        Args:
            c (Tensor): Input tensor (B, in_channels, T).
            
        Returns:
            Tensor: Output tensor (B, out_channels, T).
            Tensor: intermediate result
            Tensor: another intermediate result
        """
        c = self.input_conv(c)
        for i in range(self.num_upsamples):
            c = self.upsamples[i](c)
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks
            if i == 1:
                x1 = self.out_proj_x1(c)
            elif i == 2:
                x2 = self.out_proj_x2(c)
        c = self.output_conv(c)

        return c, x2, x1

    def reset_parameters(self):
        """
        Reset parameters.
        
        This initialization follows the official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py
        """

        def _reset_parameters(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)

        self.apply(_reset_parameters)

    def remove_weight_norm(self):
        """
        Remove weight normalization module from all of the layers.
        """

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """
        Apply weight normalization module from all of the layers.
        """

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def inference(self, c, normalize_before=False):
        """
        Perform inference.
        
        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, in_channels).
            normalize_before (bool): Whether to perform normalization.
            
        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).
        """
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
        if normalize_before:
            c = (c - self.mean) / self.scale
        c = self.forward(c.transpose(1, 0).unsqueeze(0))
        return c.squeeze(0).transpose(1, 0)


class HiFiGANPeriodDiscriminator(torch.nn.Module):

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 period=3,
                 kernel_sizes=[5, 3],
                 channels=32,
                 downsample_scales=[3, 3, 3, 3, 1],
                 max_downsample_channels=1024,
                 bias=True,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.1},
                 use_weight_norm=True,
                 use_spectral_norm=False, ):
        """
        Initialize HiFiGANPeriodDiscriminator module.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            period (int): Period.
            kernel_sizes (list): Kernel sizes of initial conv layers and the final conv layer.
            channels (int): Number of initial channels.
            downsample_scales (list): List of downsampling scales.
            max_downsample_channels (int): Number of maximum downsampling channels.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_spectral_norm (bool): Whether to use spectral norm.
                If set to true, it will be applied to all of the conv layers.
        """
        super().__init__()
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1, "Kernel size must be odd number."
        assert kernel_sizes[1] % 2 == 1, "Kernel size must be odd number."

        self.period = period
        self.convs = torch.nn.ModuleList()
        in_chs = in_channels
        out_chs = channels
        for downsample_scale in downsample_scales:
            self.convs += [torch.nn.Sequential(torch.nn.Conv2d(in_chs,
                                                               out_chs,
                                                               (kernel_sizes[0], 1),
                                                               (downsample_scale, 1),
                                                               padding=((kernel_sizes[0] - 1) // 2, 0), ),
                                               getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), )]
            in_chs = out_chs
            # NOTE(kan-bayashi): Use downsample_scale + 1?
            out_chs = min(out_chs * 4, max_downsample_channels)
        self.output_conv = torch.nn.Conv2d(out_chs,
                                           out_channels,
                                           (kernel_sizes[1] - 1, 1),
                                           1,
                                           padding=((kernel_sizes[1] - 1) // 2, 0), )

        if use_weight_norm and use_spectral_norm:
            raise ValueError("Either use use_weight_norm or use_spectral_norm.")

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # apply spectral norm
        if use_spectral_norm:
            self.apply_spectral_norm()

    def forward(self, x):
        """
        Calculate forward propagation.
        
        Args:
            c (Tensor): Input tensor (B, in_channels, T).
            
        Returns:
            list: List of each layer's tensors.
        """
        # transform 1d to 2d -> (B, C, T/P, P)
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        # forward conv
        outs = []
        for layer in self.convs:
            x = layer(x)
            outs = outs + [x]
        x = self.output_conv(x)
        x = torch.flatten(x, 1, -1)
        outs = outs + [x]

        return outs

    def apply_weight_norm(self):
        """
        Apply weight normalization module from all of the layers.
        """

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):
        """
        Apply spectral normalization module from all of the layers.
        """

        def _apply_spectral_norm(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.spectral_norm(m)

        self.apply(_apply_spectral_norm)


class HiFiGANMultiPeriodDiscriminator(torch.nn.Module):

    def __init__(self,
                 periods=[2, 3, 5, 7, 11],
                 discriminator_params={
                     "in_channels"                : 1,
                     "out_channels"               : 1,
                     "kernel_sizes"               : [5, 3],
                     "channels"                   : 32,
                     "downsample_scales"          : [3, 3, 3, 3, 1],
                     "max_downsample_channels"    : 1024,
                     "bias"                       : True,
                     "nonlinear_activation"       : "LeakyReLU",
                     "nonlinear_activation_params": {"negative_slope": 0.1},
                     "use_weight_norm"            : True,
                     "use_spectral_norm"          : False,
                     }, ):
        """
        Initialize HiFiGANMultiPeriodDiscriminator module.
        
        Args:
            periods (list): List of periods.
            discriminator_params (dict): Parameters for hifi-gan period discriminator module.
                The period parameter will be overwritten.
        """
        super().__init__()
        self.discriminators = torch.nn.ModuleList()
        for period in periods:
            params = copy.deepcopy(discriminator_params)
            params["period"] = period
            self.discriminators += [HiFiGANPeriodDiscriminator(**params)]

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.
        """
        outs = []
        for f in self.discriminators:
            outs = outs + [f(x)]

        return outs


class HiFiGANScaleDiscriminator(torch.nn.Module):

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_sizes=[15, 41, 5, 3],
                 channels=128,
                 max_downsample_channels=1024,
                 max_groups=16,
                 bias=True,
                 downsample_scales=[2, 2, 4, 4, 1],
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.1},
                 use_weight_norm=True,
                 use_spectral_norm=False, ):
        """
        Initialize HiFiGAN scale discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (list): List of four kernel sizes. The first will be used for the first conv layer,
                and the second is for downsampling part, and the remaining two are for output layers.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_spectral_norm (bool): Whether to use spectral norm.
                If set to true, it will be applied to all of the conv layers.
        """
        super().__init__()
        self.layers = torch.nn.ModuleList()

        # check kernel size is valid
        assert len(kernel_sizes) == 4
        for ks in kernel_sizes:
            assert ks % 2 == 1

        # add first layer
        self.layers += [torch.nn.Sequential(torch.nn.Conv1d(in_channels,
                                                            channels,
                                                            # NOTE(kan-bayashi): Use always the same kernel size
                                                            kernel_sizes[0],
                                                            bias=bias,
                                                            padding=(kernel_sizes[0] - 1) // 2, ),
                                            getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), )]

        # add downsample layers
        in_chs = channels
        out_chs = channels
        # NOTE(kan-bayashi): Remove hard coding?
        groups = 4
        for downsample_scale in downsample_scales:
            self.layers += [torch.nn.Sequential(torch.nn.Conv1d(in_chs,
                                                                out_chs,
                                                                kernel_size=kernel_sizes[1],
                                                                stride=downsample_scale,
                                                                padding=(kernel_sizes[1] - 1) // 2,
                                                                groups=groups,
                                                                bias=bias,
                                                                ), getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), )]
            in_chs = out_chs
            # NOTE(kan-bayashi): Remove hard coding?
            out_chs = min(in_chs * 2, max_downsample_channels)
            # NOTE(kan-bayashi): Remove hard coding?
            groups = min(groups * 4, max_groups)

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [torch.nn.Sequential(torch.nn.Conv1d(in_chs,
                                                            out_chs,
                                                            kernel_size=kernel_sizes[2],
                                                            stride=1,
                                                            padding=(kernel_sizes[2] - 1) // 2,
                                                            bias=bias, ),
                                            getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), )]
        self.layers += [torch.nn.Conv1d(out_chs,
                                        out_channels,
                                        kernel_size=kernel_sizes[3],
                                        stride=1,
                                        padding=(kernel_sizes[3] - 1) // 2,
                                        bias=bias, ), ]

        if use_weight_norm and use_spectral_norm:
            raise ValueError("Either use use_weight_norm or use_spectral_norm.")

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # apply spectral norm
        if use_spectral_norm:
            self.apply_spectral_norm()

    def forward(self, x):
        """
        Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of output tensors of each layer.
        """
        outs = []
        for f in self.layers:
            x = f(x)
            outs = outs + [x]

        return outs

    def apply_weight_norm(self):
        """
        Apply weight normalization module from all of the layers.
        """

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):
        """
        Apply spectral normalization module from all of the layers.
        """

        def _apply_spectral_norm(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.spectral_norm(m)

        self.apply(_apply_spectral_norm)


class HiFiGANMultiScaleDiscriminator(torch.nn.Module):

    def __init__(self,
                 scales=3,
                 downsample_pooling="AvgPool1d",
                 # follow the official implementation setting
                 downsample_pooling_params={
                     "kernel_size": 4,
                     "stride"     : 2,
                     "padding"    : 2,
                     },
                 discriminator_params={
                     "in_channels"                : 1,
                     "out_channels"               : 1,
                     "kernel_sizes"               : [15, 41, 5, 3],
                     "channels"                   : 128,
                     "max_downsample_channels"    : 1024,
                     "max_groups"                 : 16,
                     "bias"                       : True,
                     "downsample_scales"          : [2, 2, 4, 4, 1],
                     "nonlinear_activation"       : "LeakyReLU",
                     "nonlinear_activation_params": {"negative_slope": 0.1},
                     },
                 follow_official_norm=False, ):
        """
        Initialize HiFiGAN multi-scale discriminator module.

        Args:
            scales (int): Number of multi-scales.
            downsample_pooling (str): Pooling module name for downsampling of the inputs.
            downsample_pooling_params (dict): Parameters for the above pooling module.
            discriminator_params (dict): Parameters for hifi-gan scale discriminator module.
            follow_official_norm (bool): Whether to follow the norm setting of the official
                implementaion. The first discriminator uses spectral norm and the other
                discriminators use weight norm.
        """
        super().__init__()
        self.discriminators = torch.nn.ModuleList()

        # add discriminators
        for i in range(scales):
            params = copy.deepcopy(discriminator_params)
            if follow_official_norm:
                if i == 0:
                    params["use_weight_norm"] = False
                    params["use_spectral_norm"] = True
                else:
                    params["use_weight_norm"] = True
                    params["use_spectral_norm"] = False
            self.discriminators += [HiFiGANScaleDiscriminator(**params)]
        self.pooling = getattr(torch.nn, downsample_pooling)(**downsample_pooling_params)

    def forward(self, x):
        """
        Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.
        """
        outs = []
        for f in self.discriminators:
            outs = outs + [f(x)]
            x = self.pooling(x)

        return outs


class HiFiGANMultiScaleMultiPeriodDiscriminator(torch.nn.Module):

    def __init__(self,
                 # Multi-scale discriminator related
                 scales=3,
                 scale_downsample_pooling="AvgPool1d",
                 scale_downsample_pooling_params={
                     "kernel_size": 4,
                     "stride"     : 2,
                     "padding"    : 2,
                     },
                 scale_discriminator_params={
                     "in_channels"                : 1,
                     "out_channels"               : 1,
                     "kernel_sizes"               : [15, 41, 5, 3],
                     "channels"                   : 128,
                     "max_downsample_channels"    : 1024,
                     "max_groups"                 : 16,
                     "bias"                       : True,
                     "downsample_scales"          : [4, 4, 4, 4, 1],
                     "nonlinear_activation"       : "LeakyReLU",
                     "nonlinear_activation_params": {"negative_slope": 0.1},
                     },
                 follow_official_norm=True,
                 # Multi-period discriminator related
                 periods=[2, 3, 5, 7, 11],
                 period_discriminator_params={
                     "in_channels"                : 1,
                     "out_channels"               : 1,
                     "kernel_sizes"               : [5, 3],
                     "channels"                   : 32,
                     "downsample_scales"          : [3, 3, 3, 3, 1],
                     "max_downsample_channels"    : 1024,
                     "bias"                       : True,
                     "nonlinear_activation"       : "LeakyReLU",
                     "nonlinear_activation_params": {"negative_slope": 0.1},
                     "use_weight_norm"            : True,
                     "use_spectral_norm"          : False,
                     }, ):
        """
        Initialize HiFiGAN multi-scale + multi-period discriminator module.

        Args:
            scales (int): Number of multi-scales.
            scale_downsample_pooling (str): Pooling module name for downsampling of the inputs.
            scale_downsample_pooling_params (dict): Parameters for the above pooling module.
            scale_discriminator_params (dict): Parameters for hifi-gan scale discriminator module.
            follow_official_norm (bool): Whether to follow the norm setting of the official
                implementaion. The first discriminator uses spectral norm and the other
                discriminators use weight norm.
            periods (list): List of periods.
            period_discriminator_params (dict): Parameters for hifi-gan period discriminator module.
                The period parameter will be overwritten.
        """
        super().__init__()
        self.msd = HiFiGANMultiScaleDiscriminator(scales=scales,
                                                  downsample_pooling=scale_downsample_pooling,
                                                  downsample_pooling_params=scale_downsample_pooling_params,
                                                  discriminator_params=scale_discriminator_params,
                                                  follow_official_norm=follow_official_norm, )
        self.mpd = HiFiGANMultiPeriodDiscriminator(periods=periods,
                                                   discriminator_params=period_discriminator_params, )

    def forward(self, x):
        """
        Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of list of each discriminator outputs,
                which consists of each layer output tensors.
                Multi scale and multi period ones are concatenated.
        """
        msd_outs = self.msd(x)
        mpd_outs = self.mpd(x)
        return msd_outs + mpd_outs


class AvocodoHiFiGANJointDiscriminator(torch.nn.Module):

    def __init__(self,
                 # Multi-scale discriminator related
                 scales=3,
                 scale_downsample_pooling="AvgPool1d",
                 scale_downsample_pooling_params={
                     "kernel_size": 4,
                     "stride"     : 2,
                     "padding"    : 2,
                     },
                 scale_discriminator_params={
                     "in_channels"                : 1,
                     "out_channels"               : 1,
                     "kernel_sizes"               : [15, 41, 5, 3],
                     "channels"                   : 128,
                     "max_downsample_channels"    : 1024,
                     "max_groups"                 : 16,
                     "bias"                       : True,
                     "downsample_scales"          : [4, 4, 4, 4, 1],
                     "nonlinear_activation"       : "LeakyReLU",
                     "nonlinear_activation_params": {"negative_slope": 0.1},
                     },
                 follow_official_norm=True,
                 # Multi-period discriminator related
                 periods=[2, 3, 5, 7, 11],
                 period_discriminator_params={
                     "in_channels"                : 1,
                     "out_channels"               : 1,
                     "kernel_sizes"               : [5, 3],
                     "channels"                   : 32,
                     "downsample_scales"          : [3, 3, 3, 3, 1],
                     "max_downsample_channels"    : 1024,
                     "bias"                       : True,
                     "nonlinear_activation"       : "LeakyReLU",
                     "nonlinear_activation_params": {"negative_slope": 0.1},
                     "use_weight_norm"            : True,
                     "use_spectral_norm"          : False,
                     },
                 # CoMB discriminator related
                 kernels=[[7, 11, 11, 11, 11, 5],
                          [11, 21, 21, 21, 21, 5],
                          [15, 41, 41, 41, 41, 5]],
                 channels=[16, 64, 256, 1024, 1024, 1024],
                 groups=[1, 4, 16, 64, 256, 1],
                 strides=[1, 1, 4, 4, 4, 1],
                 # Sub-Band discriminator related
                 tkernels=[7, 5, 3],
                 fkernel=5,
                 tchannels=[64, 128, 256, 256, 256],
                 fchannels=[32, 64, 128, 128, 128],
                 tstrides=[[1, 1, 3, 3, 1],
                           [1, 1, 3, 3, 1],
                           [1, 1, 3, 3, 1]],
                 fstride=[1, 1, 3, 3, 1],
                 tdilations=[[[5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11]],
                             [[3, 5, 7], [3, 5, 7], [3, 5, 7], [3, 5, 7], [3, 5, 7]],
                             [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]],
                 fdilations=[[1, 2, 3],
                             [1, 2, 3],
                             [1, 2, 3],
                             [2, 3, 5],
                             [2, 3, 5]],
                 tsubband=[6, 11, 16],
                 n=16,
                 m=64,
                 freq_init_ch=128):
        super().__init__()
        self.msd = HiFiGANMultiScaleDiscriminator(scales=scales,
                                                  downsample_pooling=scale_downsample_pooling,
                                                  downsample_pooling_params=scale_downsample_pooling_params,
                                                  discriminator_params=scale_discriminator_params,
                                                  follow_official_norm=follow_official_norm, )
        self.mpd = HiFiGANMultiPeriodDiscriminator(periods=periods,
                                                   discriminator_params=period_discriminator_params, )
        self.mcmbd = MultiCoMBDiscriminator(kernels, channels, groups, strides)
        self.msbd = MultiSubBandDiscriminator(tkernels, fkernel, tchannels, fchannels, tstrides, fstride, tdilations, fdilations, tsubband, n, m, freq_init_ch)

    def forward(self, wave, intermediate_wave_upsampled_twice=None, intermediate_wave_upsampled_once=None):
        """
        Calculate forward propagation.

        Args:
            wave: The predicted or gold waveform
            intermediate_wave_upsampled_twice: the wave before the final upsampling in the generator
            intermediate_wave_upsampled_once: the wave before the second final upsampling in the generator

        Returns:
            List: List of lists of each discriminator outputs,
                which consists of each layer output tensors.
        """
        msd_outs = self.msd(wave)
        mpd_outs = self.mpd(wave)
        mcmbd_outs = self.mcmbd(wave_final=wave,
                                intermediate_wave_upsampled_twice=intermediate_wave_upsampled_twice,
                                intermediate_wave_upsampled_once=intermediate_wave_upsampled_once)
        msbd_outs = self.msbd(wave)
        return msd_outs + mpd_outs + mcmbd_outs + msbd_outs
