import torch
from torch import nn

from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from torch.optim import lr_scheduler

from dataload import load_minst, load_cifar_10, load_minst_random
from models import LeNet_300_100, LeNet_5_Caffe, AlexNet, MaskedLSTM, MaskedVgg, MaskedWideResNet, LeNet_5_Caffe_t, \
    MaskedVgg_t, MaskedGRU
from train import train_net, train_rnn, train_net_t, train_show_loss

torch.manual_seed(2019)


def set_axes(axes, xlabel, ylabel, xlim, ylim, legend, xscale, yscale):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-r', '-b', '-g', '-m', '-C1', '-C5'), figsize=(4, 3.5), axes=None, twins=False, ylim2=None):
    backend_inline.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize
    axes = axes if axes else plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    if twins:
        ax2 = axes.twinx()
        ax2.set_ylim(ylim2)
        ax2.set_ylabel(ylabel[1])
    i = 0
    ax = axes
    f = []
    for x, y, fmt in zip(X, Y, fmts):
        if twins and (i > 0):
            ax = ax2
        if len(x):
            h, = ax.plot(x, y, fmt)
        else:
            h, = ax.plot(y, fmt)
        f.append(h)
        i += 1
    set_axes(axes, xlabel, ylabel[0], xlim, ylim, legend, xscale, yscale)


class Tester:
    @staticmethod
    def test_topk():
        a = torch.arange(16).reshape(4, 4).view(-1)
        _, index = torch.topk(a, 4)

        mask = torch.zeros_like(a)
        mask[index] = 1
        mask.view((4, 4))
        print(mask)

    @staticmethod
    def test_lenet_300_100_muti():
        path = "models/lenet_300_100"
        epoch, batch_size, lr = 60, 100, 0.1

        loss = nn.CrossEntropyLoss()

        data_iter = load_minst(batch_size, flatten=True)

        test_err = []
        t_ratios = []
        k_ratios = torch.arange(0.4, 1., 0.1)
        for k_ratio in k_ratios:
            net = LeNet_300_100()
            trainer = torch.optim.SGD(net.parameters(), weight_decay=0.0005, lr=lr, momentum=0.9)
            scheduler = lr_scheduler.StepLR(trainer, step_size=25000, gamma=0.1)
            err = train_net(net, loss, trainer, data_iter, epoch, path, k_ratio, scheduler)
            print("true sparsity: %f%%    test_err: %.2f%%" % (net.sparsity_ratio * 100, err * 100))
            t_ratios.append(net.sparsity_ratio)
            test_err.append(err)

        plot(torch.tensor(t_ratios) * 100, torch.tensor(test_err) * 100, xlabel='Sparsity (%)', ylabel=['Error (%)'])
        # plt.show()
        plt.savefig("res/lenet-300-100.png")

    @staticmethod
    def test_lenet_5_muti():
        path = "models/lenet_5"
        epoch, batch_size, lr = 40, 100, 0.1

        loss = nn.CrossEntropyLoss()

        data_iter = load_minst(batch_size, flatten=False)

        test_err = []
        t_ratios = []
        k_ratios = torch.arange(0.2, 1., 0.1)
        for k_ratio in k_ratios:
            net = LeNet_5_Caffe()
            trainer = torch.optim.SGD(net.parameters(), weight_decay=0.0005, lr=lr, momentum=0.9)
            scheduler = lr_scheduler.StepLR(trainer, step_size=25000, gamma=0.1)
            err = train_net(net, loss, trainer, data_iter, epoch, path, k_ratio, scheduler)
            print("true sparsity: %f%%    test_err: %.2f%%" % (net.sparsity_ratio * 100, err * 100))
            t_ratios.append(net.sparsity_ratio)
            test_err.append(err)

        plot(torch.tensor(t_ratios) * 100, torch.tensor(test_err) * 100, ylim=[0.6, 1.2], xlabel='Sparsity (%)',
             ylabel=['Error (%)'])
        plt.show()
        # plt.savefig("res/lenet-5.png")

    @staticmethod
    def test_lenet_300_100(k_ratio):
        path = "models/lenet_300_100"
        epoch, batch_size, lr = 250, 100, 0.1

        loss = nn.CrossEntropyLoss()

        data_iter = load_minst(batch_size, flatten=True)

        net = LeNet_300_100()
        trainer = torch.optim.SGD(net.parameters(), weight_decay=0.0005, lr=lr, momentum=0.9)
        scheduler = lr_scheduler.StepLR(trainer, step_size=25000, gamma=0.1)
        err = train_net(net, loss, trainer, data_iter, epoch, path, k_ratio, scheduler)
        print("true sparsity: %f%%    test_err: %.2f%%" % (net.sparsity_ratio * 100, err * 100))

    @staticmethod
    def test_lenet_5(k_ratio):
        path = "models/lenet_5"
        epoch, batch_size, lr = 250, 100, 0.1

        loss = nn.CrossEntropyLoss()

        data_iter = load_minst(batch_size, flatten=False)

        net = LeNet_5_Caffe()
        trainer = torch.optim.SGD(net.parameters(), weight_decay=0.0005, lr=lr, momentum=0.9)
        scheduler = lr_scheduler.StepLR(trainer, step_size=25000, gamma=0.1)
        err = train_net(net, loss, trainer, data_iter, epoch, path, k_ratio, scheduler)
        print("true sparsity: %f%%    test_err: %.2f%%" % (net.sparsity_ratio * 100, err * 100))

    @staticmethod
    def test_masked_lstm(hidden_size):
        path = "models/lstm_s"
        epoch, batch_size, lr = 100, 100, 0.1

        loss = nn.CrossEntropyLoss().cuda(torch.device("cuda:0"))

        data_iter = load_minst(batch_size, flatten=False, rnn=True)

        net = MaskedLSTM(28, hidden_size, 2, 10)
        net.init_lstm_state(batch_size, hidden_size)
        trainer = torch.optim.SGD(net.parameters(), weight_decay=0.0005, lr=lr, momentum=0.9)
        scheduler = lr_scheduler.StepLR(trainer, step_size=25000, gamma=0.1)
        err = train_rnn(net, loss, trainer, data_iter, epoch, path, 0.95, scheduler)
        print("true sparsity: %f%%    test_err: %.2f%%" % (net.sparsity_ratio * 100, err * 100))

    @staticmethod
    def test_masked_gru(hidden_size):
        path = "models/gru"
        epoch, batch_size, lr = 100, 100, 0.1

        loss = nn.CrossEntropyLoss().cuda(torch.device("cuda:0"))

        data_iter = load_minst(batch_size, flatten=False, rnn=True)

        net = MaskedGRU(28, hidden_size, 1, 10)
        net.init_gru_state(batch_size, hidden_size)
        trainer = torch.optim.SGD(net.parameters(), weight_decay=0.0005, lr=lr, momentum=0.9)
        scheduler = lr_scheduler.StepLR(trainer, step_size=25000, gamma=0.1)
        err = train_rnn(net, loss, trainer, data_iter, epoch, path, 0.95, scheduler)
        print("true sparsity: %f%%    test_err: %.2f%%" % (net.sparsity_ratio * 100, err * 100))

    @staticmethod
    def test_alexnet(fc_size):
        path = "models/alexnet"
        epoch, batch_size, lr = 250, 128, 0.1

        loss = nn.CrossEntropyLoss().cuda(torch.device("cuda:0"))

        data_iter = load_cifar_10(batch_size)

        net = AlexNet(fc_size)
        trainer = torch.optim.SGD(net.parameters(), weight_decay=0.0005, lr=lr, momentum=0.9)
        scheduler = lr_scheduler.StepLR(trainer, step_size=30000, gamma=0.1)
        err = train_net(net, loss, trainer, data_iter, epoch, path, 0.9, scheduler)
        print("true sparsity: %f%%    test_err: %.2f%%" % (net.sparsity_ratio * 100, err * 100))

    @staticmethod
    def test_masked_vggD():
        path = "models/vgg"
        epoch, batch_size, lr = 300, 128, 0.1

        loss = nn.CrossEntropyLoss().cuda(torch.device("cuda:0"))

        data_iter = load_cifar_10(batch_size)

        net = MaskedVgg()
        trainer = torch.optim.SGD(net.parameters(), weight_decay=0.0005, lr=lr, momentum=0.9)
        scheduler = lr_scheduler.StepLR(trainer, step_size=30000, gamma=0.1)
        err = train_net(net, loss, trainer, data_iter, epoch, path, 0.95, scheduler)
        print("true sparsity: %f%%    test_err: %.2f%%" % (net.sparsity_ratio * 100, err * 100))

    @staticmethod
    def test_masked_wideres(wide_f):
        path = "models/wideres"
        epoch, batch_size, lr = 300, 128, 0.1

        loss = nn.CrossEntropyLoss().cuda(torch.device("cuda:0"))

        data_iter = load_cifar_10(batch_size)

        net = MaskedWideResNet(wide_f)
        trainer = torch.optim.SGD(net.parameters(), weight_decay=0.0005, lr=lr, momentum=0.9)
        scheduler = lr_scheduler.StepLR(trainer, step_size=30000, gamma=0.1)
        err = train_net(net, loss, trainer, data_iter, epoch, path, 0.95, scheduler)
        print("true sparsity: %f%%    test_err: %.2f%%" % (net.sparsity_ratio * 100, err * 100))

    @staticmethod
    def test_LeNet_5_Caffe_t(k_ratio):
        path = "models/lenet_5_t"
        epoch, batch_size, lr = 250, 100, 0.1

        loss = nn.CrossEntropyLoss().cuda()

        data_iter = load_minst(batch_size, flatten=False)

        net = LeNet_5_Caffe_t()
        trainer = torch.optim.SGD(net.parameters(), weight_decay=0.0005, lr=lr, momentum=0.9)
        scheduler = lr_scheduler.StepLR(trainer, step_size=25000, gamma=0.1)
        err = train_net_t(net, loss, trainer, data_iter, epoch, path, k_ratio, scheduler)
        print("true sparsity: %f%%    test_err: %.2f%%" % (net.sparsity_ratio * 100, err * 100))

    @staticmethod
    def test_MaskedVgg_t():
        path = "models/VGG_t"
        epoch, batch_size, lr = 300, 128, 0.1

        loss = nn.CrossEntropyLoss().cuda(torch.device("cuda:0"))

        data_iter = load_cifar_10(batch_size)

        net = MaskedVgg_t()
        trainer = torch.optim.SGD(net.parameters(), weight_decay=0.0005, lr=lr, momentum=0.9)
        scheduler = lr_scheduler.StepLR(trainer, step_size=30000, gamma=0.1)
        err = train_net_t(net, loss, trainer, data_iter, epoch, path, 0.95, scheduler)
        print("true sparsity: %f%%    test_err: %.2f%%" % (net.sparsity_ratio * 100, err * 100))

    @staticmethod
    def test_loss():
        epoch, batch_size, lr = 60, 100, 0.01

        loss = nn.CrossEntropyLoss()

        true_iter, random_iter = load_minst_random(batch_size)
        steps = len(true_iter) * epoch

        print('true labels')
        net = LeNet_5_Caffe()
        trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        tl = train_show_loss(net, loss, trainer, true_iter, epoch)

        print('true labels(pruned)')
        net = LeNet_5_Caffe()
        net.apply_masks(next(iter(true_iter)), loss, 0.99)
        trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        tlp = train_show_loss(net, loss, trainer, true_iter, epoch)

        print('random labels')
        net = LeNet_5_Caffe()
        trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        rl = train_show_loss(net, loss, trainer, random_iter, epoch)

        print('random labels + reg')
        net = LeNet_5_Caffe()
        trainer = torch.optim.SGD(net.parameters(), weight_decay=0.0005, lr=lr, momentum=0.9)
        rld = train_show_loss(net, loss, trainer, random_iter, epoch)

        print('random labels(pruned)')
        net = LeNet_5_Caffe()
        net.apply_masks(next(iter(random_iter)), loss, 0.99)
        trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        rlp = train_show_loss(net, loss, trainer, random_iter, epoch)

        plot(torch.arange(steps) / 1000, [tl, tlp, rl, rld, rlp], xlabel='Iteration (×10^3)', ylabel=['Loss'],
             legend=['true labels', 'true labels(pruned)', 'random labels', 'random labels + reg',
                     'random labels(pruned)'])
        plt.show()
