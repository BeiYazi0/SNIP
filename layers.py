import torch
from torch import nn
from torch.nn import functional as F


class BaseMaskedLayer(nn.Module):
    def __init__(self, masked=False):
        super(BaseMaskedLayer, self).__init__()
        if not masked:
            device = torch.device("cuda:0")
            if isinstance(self.weight_shape, list):
                self.mask = []
                for weight_shape in self.weight_shape:
                    mask = torch.ones(weight_shape, requires_grad=True, device=device)
                    self.mask.append(mask)
            else:
                self.mask = torch.ones(self.weight_shape, requires_grad=True, device=device)

    def forward(self, *args):
        raise NotImplementedError

    def apply_mask(self, mask):
        self.mask.requires_grad = False
        self.weight.grad.zero_()
        mask = mask.view(self.weight_shape).float()
        self.weight.register_hook(lambda grad: grad * mask)
        self.mask[:] = mask

    def get_grad(self):
        if isinstance(self.mask, list):
            res = []
            for mask in self.mask:
                res.append(mask.grad.view(-1))
            return torch.cat(res)
        return self.mask.grad.view(-1)

    def init_parameters(self):
        nn.init.xavier_normal_(self.weight)

    @property
    def weight_numel(self):
        return self._weight_num


class MaskedLinear(BaseMaskedLayer):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True):
        self.weight_shape = (out_features, in_features)
        self._weight_num = out_features * in_features
        super(MaskedLinear, self).__init__()

        self.weight = nn.Parameter(torch.zeros(self.weight_shape))
        self.init_parameters()
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,)))
        else:
            self.bias = None

    def forward(self, x):
        output = nn.functional.linear(x, self.mask * self.weight, self.bias)
        return output


class MaskedConv2d(BaseMaskedLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):
        self._weight_num = out_channels * in_channels * kernel_size * kernel_size
        self.weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        super(MaskedConv2d, self).__init__()
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.zeros(self.weight_shape))
        self.init_parameters()
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_channels,)))
        else:
            self.bias = None

    def forward(self, x):
        output = nn.functional.conv2d(x, self.mask * self.weight, bias=self.bias, stride=self.stride,
                                      padding=self.padding)
        return output


class MaskedLSTMCell(BaseMaskedLayer):
    def __init__(self, input_size, hidden_size):
        self.weight_shape = [(4, hidden_size, input_size), (4, hidden_size, hidden_size)]
        self._weight_num = 4 * (input_size * hidden_size + hidden_size * hidden_size)
        super(MaskedLSTMCell, self).__init__()
        self.p = 4 * hidden_size * input_size

        self.W_x = nn.Parameter(torch.zeros((4, hidden_size, input_size)))
        self.W_h = nn.Parameter(torch.zeros((4, hidden_size, hidden_size)))
        self.init_parameters()
        self.bias = nn.Parameter(torch.zeros((4, hidden_size)))
        self.state = (None, None)

    def apply_mask(self, mask):
        mask1 = mask[:self.p].view(self.weight_shape[0]).float()
        mask2 = mask[self.p:].view(self.weight_shape[1]).float()

        for mask in self.mask:
            mask.requires_grad = False
        self.W_x.grad.zero_()
        self.W_h.grad.zero_()

        self.W_x.register_hook(lambda grad: grad * mask1)
        self.W_h.register_hook(lambda grad: grad * mask2)
        self.mask[0][:] = mask1
        self.mask[1][:] = mask2

    def init_parameters(self):
        for weight in self.W_x:
            nn.init.xavier_normal_(weight)
        for weight in self.W_h:
            nn.init.xavier_normal_(weight)

    def init_state(self, batch_size, num_hiddens):
        self.state = torch.zeros((batch_size, num_hiddens)).cuda(), torch.zeros((batch_size, num_hiddens)).cuda()

    def reset_state(self):
        (H, C) = self.state
        H.fill_(0.)
        C.fill_(0.)
        self.state = (H, C)

    def forward(self, inputs):
        masked_params = []
        for i in range(4):
            weight_x = self.W_x[i]
            mask = self.mask[0][i]
            masked_params.append((weight_x * mask).transpose(1, 0))  # 转置方便下面实现WT @ X

            weight_h = self.W_h[i]
            mask = self.mask[1][i]
            masked_params.append((weight_h * mask).transpose(1, 0))  # 转置方便下面实现WT @ X

        [W_xi, W_hi, W_xf, W_hf, W_xo, W_ho, W_xc, W_hc] = masked_params
        [b_i, b_f, b_o, b_c] = self.bias

        (H, C) = self.state
        outputs = []  # 各个时间步的输出
        for X in inputs:
            I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
            F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
            O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
            C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
            C = F * C + I * C_tilda
            H = O * torch.tanh(C)
            outputs.append(H)
        self.state = (H.detach(), C.detach())  # Wraps hidden states in new Tensors, to detach them from their history
        return torch.stack(outputs)


class StateSelect(nn.Module):
    def __init__(self):
        super(StateSelect, self).__init__()

    def forward(self, X):
        return X[-1]


class MaskedResidual(BaseMaskedLayer):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(MaskedResidual, self).__init__(True)
        self.conv1 = MaskedConv2d(input_channels, num_channels, 3, padding=1, stride=strides)
        self.conv2 = MaskedConv2d(num_channels, num_channels, 3, padding=1)
        self._weight_num = self.conv1.weight_numel + self.conv2.weight_numel
        self.p = [self.conv1.weight_numel, self._weight_num]
        if use_1x1conv:
            self.conv3 = MaskedConv2d(input_channels, num_channels, 1, stride=strides)
            self._weight_num += self.conv3.weight_numel
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

    def get_grad(self):
        res = [self.conv1.get_grad(), self.conv2.get_grad()]
        if self.conv3:
            res.append(self.conv3.get_grad())
        return torch.cat(res)

    def apply_mask(self, mask):
        self.conv1.apply_mask(mask[:self.p[0]])
        self.conv2.apply_mask(mask[self.p[0]:self.p[1]])
        if self.conv3:
            self.conv3.apply_mask(mask[self.p[1]:])


class MaskedLinear_t(BaseMaskedLayer):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True):
        self.weight_shape = (out_features, in_features)
        self._weight_num = out_features * in_features
        super(MaskedLinear_t, self).__init__()
        del self.mask

        self.weight = nn.Parameter(torch.zeros(self.weight_shape))
        self.init_parameters()
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,)))
        else:
            self.bias = None
        self.grad = None

    def forward(self, x):
        output = nn.functional.linear(x, self.weight, self.bias)
        return output

    def save_grad(self):
        self.grad = self.weight.grad.detach()

    def get_grad(self):
        return (self.grad * self.weight).view(-1)

    def apply_mask(self, mask):
        mask = mask.view(self.weight_shape).float()
        with torch.no_grad():
            self.weight *= mask


class MaskedConv2d_t(BaseMaskedLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):
        self._weight_num = out_channels * in_channels * kernel_size * kernel_size
        self.weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        super(MaskedConv2d_t, self).__init__()
        del self.mask
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.zeros(self.weight_shape))
        self.init_parameters()
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_channels,)))
        else:
            self.bias = None
        self.grad = None

    def forward(self, x):
        output = nn.functional.conv2d(x, self.weight, bias=self.bias, stride=self.stride,
                                      padding=self.padding)
        return output

    def save_grad(self):
        self.grad = self.weight.grad.detach()

    def get_grad(self):
        return (self.grad * self.weight).view(-1)

    def apply_mask(self, mask):
        mask = mask.view(self.weight_shape).float()
        with torch.no_grad():
            self.weight *= mask
