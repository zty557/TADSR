import torch
import torch.nn as nn
import torch.nn.functional as F

class PeriodicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super(PeriodicConv2d, self).__init__()
        # 保存原始参数
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self._stride = stride if isinstance(stride, tuple) else (stride, stride)
        self._padding = padding if isinstance(padding, tuple) else (padding, padding)
        self._dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self._groups = groups
        
        # 创建内部Conv2d层，但padding设为0，因为我们会手动处理padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, 
                             padding=0, dilation=dilation, groups=groups, bias=bias, 
                             padding_mode='zeros', device=device, dtype=dtype)

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def kernel_size(self):
        return self._kernel_size

    @property
    def stride(self):
        return self._stride

    @property
    def padding(self):
        return self._padding

    @property
    def dilation(self):
        return self._dilation

    @property
    def groups(self):
        return self._groups

    def forward(self, x):
        # 计算需要的padding大小
        pad_h = self._kernel_size[0] // 2
        pad_w = self._kernel_size[1] // 2

        # 获取输入特征图的尺寸
        batch_size, channels, height, width = x.size()

        # 创建新的特征图，高度增加上下padding
        padded_x = torch.zeros(batch_size, channels, height + 2 * pad_h, width + 2 * pad_w, 
                             device=x.device, dtype=x.dtype)
        
        # 复制原始特征图到中间位置
        padded_x[:, :, pad_h:pad_h+height, pad_w:pad_w+width] = x

        # 处理左右padding（周期延展）
        # 左侧padding使用右侧的值
        padded_x[:, :, pad_h:pad_h+height, :pad_w] = x[:, :, :, -pad_w:]
        # 右侧padding使用左侧的值
        padded_x[:, :, pad_h:pad_h+height, -pad_w:] = x[:, :, :, :pad_w]

        # 使用传统Conv2d进行卷积
        return self.conv(padded_x)

    def extra_repr(self):
        s = ('{_in_channels}, {_out_channels}, kernel_size={_kernel_size}'
             ', stride={_stride}')
        if self._padding != (0,) * len(self._padding):
            s += ', padding={_padding}'
        if self._dilation != (1,) * len(self._dilation):
            s += ', dilation={_dilation}'
        if self._groups != 1:
            s += ', groups={_groups}'
        if self.conv.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                             missing_keys, unexpected_keys, error_msgs):
        # 获取原始权重键名
        weight_key = prefix + 'conv.weight'
        bias_key = prefix + 'conv.bias'
        
        # 检查是否存在权重
        if weight_key in state_dict:
            self.conv.weight.data.copy_(state_dict[weight_key])
            state_dict.pop(weight_key)
        
        # 检查是否存在偏置
        if bias_key in state_dict and self.conv.bias is not None:
            self.conv.bias.data.copy_(state_dict[bias_key])
            state_dict.pop(bias_key)
        
        # 调用父类的加载方法处理其他参数
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                    missing_keys, unexpected_keys, error_msgs)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        # 将内部conv层的权重重命名为标准格式
        for key in list(state_dict.keys()):
            if 'conv.' in key:
                # 保持conv.前缀
                pass
        return state_dict
