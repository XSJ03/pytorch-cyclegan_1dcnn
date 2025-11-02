import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class CNET(nn.Module):
    def __init__(self,
                 name,
                 residual_channels=64,
                 filter_width=3,
                 dilations=[1, 2, 4, 8, 1, 2, 4, 8],
                 input_channels=123,
                 output_channels=48,
                 cond_dim=None,
                 cond_channels=64,
                 postnet_channels=256,
                 do_postproc=True,
                 do_GU=True):

        super(CNET, self).__init__()

        self.name = name
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filter_width = filter_width
        self.dilations = dilations
        self.residual_channels = residual_channels
        self.postnet_channels = postnet_channels
        self.do_postproc = do_postproc
        self.do_GU = do_GU
        self._use_cond = cond_dim is not None

        if self._use_cond:
            self.cond_dim = cond_dim
            self.cond_channels = cond_channels

        self._create_layers()

    def _create_layers(self):
        fw = self.filter_width
        r = self.residual_channels
        s = self.postnet_channels

        # Input layer
        self.input_conv = nn.Conv1d(
            self.input_channels, 2 * r,
            kernel_size=fw, padding=(fw - 1) // 2
        )

        # Conditional embedding
        if self._use_cond:
            self.cond_embed = nn.Conv1d(
                self.cond_dim, self.cond_channels,
                kernel_size=1, padding=0
            )

        # Convolution modules
        self.conv_modules = nn.ModuleList()
        for i, dilation in enumerate(self.dilations):
            module = nn.ModuleDict()

            # Filter and gate convolution
            padding = (dilation * (fw - 1)) // 2
            module['filter_gate'] = nn.Conv1d(
                r, 2 * r,
                kernel_size=fw,
                dilation=dilation,
                padding=padding
            )

            # Skip connection
            if self.do_postproc:
                module['skip'] = nn.Conv1d(r, s, kernel_size=1)

            # Gated unit
            if self.do_GU:
                module['post_filter'] = nn.Conv1d(r, r, kernel_size=1)

            # Conditional convolution
            if self._use_cond:
                module['cond_filter'] = nn.Conv1d(
                    self.cond_channels, 2 * r, kernel_size=1
                )

            self.conv_modules.append(module)

        # Post-processing module
        if self.do_postproc:
            self.postproc_conv1 = nn.Conv1d(
                s, s, kernel_size=fw, padding=(fw - 1) // 2
            )

            if isinstance(self.output_channels, list):
                output_dim = sum(self.output_channels)
            else:
                output_dim = self.output_channels

            self.postproc_conv2 = nn.Conv1d(
                s, output_dim, kernel_size=fw, padding=(fw - 1) // 2
            )
        else:
            # Last layer
            if isinstance(self.output_channels, list):
                output_dim = sum(self.output_channels)
            else:
                output_dim = self.output_channels

            self.last_conv = nn.Conv1d(
                r, output_dim, kernel_size=fw, padding=(fw - 1) // 2
            )

    def _input_layer(self, x):
        # x shape: (batch, seq_len, input_channels)
        x = x.transpose(1, 2)  # (batch, input_channels, seq_len)

        y = self.input_conv(x)

        # Split for tanh and sigmoid gates
        r = self.residual_channels
        y_tanh = torch.tanh(y[:, :r, :])
        y_sigmoid = torch.sigmoid(y[:, r:, :])
        y = y_tanh * y_sigmoid

        return y  # (batch, residual_channels, seq_len)

    def _embed_cond(self, cond_input):
        # cond_input shape: (batch, seq_len, cond_dim)
        cond_input = cond_input.transpose(1, 2)  # (batch, cond_dim, seq_len)

        y = self.cond_embed(cond_input)
        y = torch.tanh(y)

        return y  # (batch, cond_channels, seq_len)

    def _conv_module(self, x, module, dilation, cond_input=None):
        # x shape: (batch, residual_channels, seq_len)

        # Filter and gate convolution
        y = module['filter_gate'](x)

        # Add conditional input if available
        if self._use_cond and cond_input is not None:
            cond_contribution = module['cond_filter'](cond_input)
            y = y + cond_contribution

        # Split for tanh and sigmoid gates
        r = self.residual_channels
        y_tanh = torch.tanh(y[:, :r, :])
        y_sigmoid = torch.sigmoid(y[:, r:, :])
        y = y_tanh * y_sigmoid

        # Skip connection
        if self.do_postproc:
            skip_out = module['skip'](y)
        else:
            skip_out = None

        # Gated unit
        if self.do_GU:
            y = module['post_filter'](y)
            y = y + x  # Residual connection

        return y, skip_out

    def _postproc_module(self, skip_outputs):
        # Sum all skip outputs
        x = torch.stack(skip_outputs, dim=0).sum(dim=0)

        y = self.postproc_conv1(x)
        y = F.relu(y)
        y = self.postproc_conv2(y)

        # Split output if needed
        if isinstance(self.output_channels, list):
            outputs = []
            start = 0
            for channels in self.output_channels:
                outputs.append(y[:, start:start + channels, :])
                start += channels
            return outputs
        else:
            return y

    def _last_layer(self, x):
        y = self.last_conv(x)

        # Split output if needed
        if isinstance(self.output_channels, list):
            outputs = []
            start = 0
            for channels in self.output_channels:
                outputs.append(y[:, start:start + channels, :])
                start += channels
            return outputs
        else:
            return y

    def forward(self, x_input, cond_input=None):
        # x_input shape: (batch, seq_len, input_channels)

        # Embed conditional input if available
        if self._use_cond and cond_input is not None:
            cond_embedded = self._embed_cond(cond_input)
        else:
            cond_embedded = None

        # Input layer
        x = self._input_layer(x_input)

        # Convolution modules
        skip_outputs = []
        for i, (module, dilation) in enumerate(zip(self.conv_modules, self.dilations)):
            x, skip_out = self._conv_module(x, module, dilation, cond_embedded)
            if self.do_postproc and skip_out is not None:
                skip_outputs.append(skip_out)

        # Output processing
        if self.do_postproc:
            output = self._postproc_module(skip_outputs)
        else:
            output = self._last_layer(x)

        # Convert back to (batch, seq_len, channels) format
        if isinstance(output, list):
            output = [out.transpose(1, 2) for out in output]
        else:
            output = output.transpose(1, 2)

        return output

    def get_variable_list(self):
        """返回模型参数列表（兼容性函数）"""
        return list(self.parameters())