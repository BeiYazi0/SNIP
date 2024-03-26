import torch
from torch import nn

from layers import BaseMaskedLayer, MaskedConv2d, MaskedLinear, MaskedLSTMCell, StateSelect, MaskedResidual, MaskedConv2d_t, MaskedLinear_t


class BaseModel:
    def __init__(self, net):
        self.net = net
        self.sparsity_ratio = 0

    def __call__(self, x):
        return self.net(x)

    def __getitem__(self, index):
        return self.net[index]

    def get_masks_grad(self):
        weight_num = 0
        mask_grad = []
        for layer in self.net:
            if isinstance(layer, BaseMaskedLayer):
                mask_grad.append(torch.abs(layer.get_grad()))
                weight_num += layer.weight_numel

        return weight_num, mask_grad

    def apply_masks(self, inputs, loss, k_ratio):
        self.net.train()
        X, y = inputs
        l = loss(self.net(X), y)
        l.backward()

        weight_num, masks_grad = self.get_masks_grad()

        keep_weight_num = int((1 - k_ratio) * weight_num)
        score = torch.cat(masks_grad)
        score_sum = torch.sum(score)
        score.div_(score_sum)
        print(torch.max(score))

        threshold = torch.topk(score, keep_weight_num)[0][-1] * score_sum
        masks = [mask_grad >= threshold for mask_grad in masks_grad]

        self.sparsity_ratio = 1 - torch.sum(torch.cat(masks)) / weight_num

        i = 0
        for layer in self.net:
            if isinstance(layer, BaseMaskedLayer):
                layer.apply_mask(masks[i])
                i += 1

    def load_state_dict(self, state):
        self.net.load_state_dict(state)

    def state_dict(self):
        return self.net.state_dict()

    def parameters(self):
        return self.net.parameters()

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()


class LeNet_300_100(BaseModel):
    def __init__(self, ):
        net = nn.Sequential(
            MaskedLinear(28 * 28, 300), nn.ReLU(),
            MaskedLinear(300, 100), nn.ReLU(),
            MaskedLinear(100, 10))
        super(LeNet_300_100, self).__init__(net.cuda(torch.device("cuda:0")))


class LeNet_5_Caffe(BaseModel):
    def __init__(self):
        net = nn.Sequential(
            MaskedConv2d(1, 20, 5, padding=0, bias=True), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MaskedConv2d(20, 50, 5, padding=0, bias=True), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            MaskedLinear(50 * 4 * 4, 500), nn.ReLU(),
            MaskedLinear(500, 10))
        super(LeNet_5_Caffe, self).__init__(net.cuda(torch.device("cuda:0")))


class AlexNet(BaseModel):
    def __init__(self, fc_size):
        net = nn.Sequential(
            MaskedConv2d(3, 96, kernel_size=11, stride=2, padding=1), nn.ReLU(),
            MaskedConv2d(96, 256, kernel_size=5, stride=2, padding=1), nn.ReLU(),
            MaskedConv2d(256, 384, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            MaskedConv2d(384, 384, kernel_size=3, stride=2,  padding=1), nn.ReLU(),
            MaskedConv2d(384, 256, kernel_size=3, stride=2,  padding=1), nn.ReLU(),
            nn.Flatten(),
            MaskedLinear(256, fc_size), nn.ReLU(),
            MaskedLinear(fc_size, fc_size), nn.ReLU(),
            MaskedLinear(fc_size, 10))

        super(AlexNet, self).__init__(net.cuda(torch.device("cuda:0")))


class MaskedLSTM(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        net = nn.Sequential()
        for i in range(num_layers):
            net.add_module(f"lstm{i}", MaskedLSTMCell(input_size, hidden_size))
            input_size = hidden_size
        net.add_module("select", StateSelect())  # 选取最后一个时间步的状态作为linear的输入
        net.add_module("fc", MaskedLinear(hidden_size, output_size))

        super(MaskedLSTM, self).__init__(net.cuda(torch.device("cuda:0")))

    def init_lstm_state(self, batch_size, num_hiddens):
        for layer in self.net:
            if isinstance(layer, MaskedLSTMCell):
                layer.init_state(batch_size, num_hiddens)

    def reset_state(self):
        for layer in self.net:
            if isinstance(layer, MaskedLSTMCell):
                layer.reset_state()


class MaskedVgg(BaseModel): # VGG-D
    def __init__(self, conv_arch=((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))):  # vgg16
        net = self._vgg(conv_arch)
        super(MaskedVgg, self).__init__(net.cuda(torch.device("cuda:0")))

    @staticmethod
    def _vgg_block(num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(MaskedConv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return layers

    def _vgg(self, conv_arch):
        conv_blks = []
        in_channels = 3
        # 卷积层部分
        for (num_convs, out_channels) in conv_arch:
            conv_blks.extend(self._vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        return nn.Sequential(
            *conv_blks, nn.Flatten(),
            # 全连接层部分
            MaskedLinear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            MaskedLinear(512, 10))


class MaskedWideResNet(BaseModel):
    def __init__(self, widen_factor):
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        # 1st conv before any network block
        conv1 = MaskedConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        conv = []
        for i in range(1, 4):
            conv.extend(self._resnet_block(nChannels[i - 1], nChannels[i], 2))
        net = nn.Sequential(
            conv1, nn.BatchNorm2d(nChannels[0]), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *conv,
            nn.BatchNorm2d(64 * widen_factor), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(), MaskedLinear(512, 10))
        super(MaskedWideResNet, self).__init__(net.cuda(torch.device("cuda:0")))

    @staticmethod
    def _resnet_block(input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(MaskedResidual(input_channels, num_channels,
                                          use_1x1conv=True, strides=2))
            else:
                blk.append(MaskedResidual(num_channels, num_channels))
        return blk


class LeNet_5_Caffe_t(BaseModel):
    def __init__(self):
        net = nn.Sequential(
            MaskedConv2d_t(1, 20, 5, padding=0, bias=True), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MaskedConv2d_t(20, 50, 5, padding=0, bias=True), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            MaskedLinear_t(50 * 4 * 4, 500), nn.ReLU(),
            MaskedLinear_t(500, 10))
        super(LeNet_5_Caffe_t, self).__init__(net.cuda(torch.device("cuda:0")))
        self.i = 0
        self.premasks = None
        self.curmasks = None

    def save_grad(self):
        for layer in self.net:
            if isinstance(layer, MaskedLinear_t) or isinstance(layer, MaskedConv2d_t):
                layer.save_grad()

    def remask(self, k_ratio, save=False):
        weight_num, masks_grad = self.get_masks_grad()

        keep_weight_num = int((1 - k_ratio) * weight_num)
        score = torch.cat(masks_grad)
        score_sum = torch.sum(score)
        score.div_(score_sum)

        threshold = torch.topk(score, keep_weight_num)[0][-1] * score_sum
        masks = [mask_grad >= threshold for mask_grad in masks_grad]

        if save:
            self.curmasks = masks

        self.sparsity_ratio = 1 - torch.sum(torch.cat(masks)) / weight_num

        i = 0
        for layer in self.net:
            if isinstance(layer, BaseMaskedLayer):
                layer.apply_mask(masks[i])
                i += 1

    def cal_mask_trans(self):
        if self.i == 0:
            self.i = 1
            self.premasks = self.curmasks
            return
        cnt, num = 0, 0
        for pre, cur in zip(self.premasks, self.curmasks):
            cnt += torch.sum(pre != cur)
            num += pre.numel()
        print("trans: %f" % (cnt/num))
        self.premasks = self.curmasks


class MaskedVgg_t(BaseModel): # VGG-D
    def __init__(self, conv_arch=((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))):  # vgg16
        net = self._vgg(conv_arch)
        super(MaskedVgg_t, self).__init__(net.cuda(torch.device("cuda:0")))
        self.i = 0
        self.premasks = None
        self.curmasks = None

    @staticmethod
    def _vgg_block(num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(MaskedConv2d_t(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return layers

    def _vgg(self, conv_arch):
        conv_blks = []
        in_channels = 3
        # 卷积层部分
        for (num_convs, out_channels) in conv_arch:
            conv_blks.extend(self._vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        return nn.Sequential(
            *conv_blks, nn.Flatten(),
            # 全连接层部分
            MaskedLinear_t(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            MaskedLinear_t(512, 10))

    def save_grad(self):
        for layer in self.net:
            if isinstance(layer, MaskedLinear_t) or isinstance(layer, MaskedConv2d_t):
                layer.save_grad()

    def remask(self, k_ratio, save=False):
        weight_num, masks_grad = self.get_masks_grad()

        keep_weight_num = int((1 - k_ratio) * weight_num)
        score = torch.cat(masks_grad)
        score_sum = torch.sum(score)
        score.div_(score_sum)

        threshold = torch.topk(score, keep_weight_num)[0][-1] * score_sum
        masks = [mask_grad >= threshold for mask_grad in masks_grad]

        if save:
            self.curmasks = masks

        self.sparsity_ratio = 1 - torch.sum(torch.cat(masks)) / weight_num

        i = 0
        for layer in self.net:
            if isinstance(layer, BaseMaskedLayer):
                layer.apply_mask(masks[i])
                i += 1

    def cal_mask_trans(self):
        if self.i == 0:
            self.i = 1
            self.premasks = self.curmasks
            return
        cnt, num = 0, 0
        for pre, cur in zip(self.premasks, self.curmasks):
            cnt += torch.sum(pre != cur)
            num += pre.numel()
        print("trans: %f" % (cnt/num))
        self.premasks = self.curmasks
