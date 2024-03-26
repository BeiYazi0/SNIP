import torch


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = (y_hat.type(y.dtype) == y)
    return torch.sum(cmp)


def evaluate_accuracy(net, data_iter, rnn=False):
    net.eval()  # 设置为评估模式
    # 正确预测的数量，总预测的数量
    device = torch.device("cuda:0")
    metric = torch.zeros(2, device=device)
    with torch.no_grad():
        for X, y in data_iter:
            if rnn:
                X = X.permute(1, 0, 2)
                net.reset_state()
            metric[0] += accuracy(net(X), y)
            metric[1] += y.numel()
    return metric[0] / metric[1]


def train_net(net, loss, trainer, data_iter, epochs, path, k_ratio, scheduler=None):
    train_iter, val_iter, test_iter = data_iter

    net.apply_masks(next(iter(train_iter)), loss, k_ratio)
    print("true sparsity: %f%%" % (net.sparsity_ratio * 100))

    val_acc_best = 0
    for epoch in range(epochs):
        net.train()
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)

            trainer.zero_grad()  # 清除了优化器中的grad
            l.backward()  # 通过进行反向传播来计算梯度
            trainer.step()  # 通过调用优化器来更新模型参数
            if scheduler:
                scheduler.step()

        val_acc = evaluate_accuracy(net, val_iter)
        if val_acc > val_acc_best:
            val_acc_best = val_acc
            torch.save(net.state_dict(), path)

        print("epoch: %d    val_acc: %.2f%%" % (epoch + 1, val_acc * 100))

    if val_acc < val_acc_best:
        net.load_state_dict(torch.load(path))

    return 1 - evaluate_accuracy(net, test_iter)


def train_rnn(net, loss, trainer, data_iter, epochs, path, k_ratio, scheduler=None):
    train_iter, val_iter, test_iter = data_iter

    X, y = next(iter(train_iter))
    net.apply_masks((X.permute(1, 0, 2), y), loss, k_ratio)
    print("true sparsity: %f%%" % (net.sparsity_ratio * 100))

    val_acc_best = 0
    for epoch in range(epochs):
        net.train()
        for X, y in train_iter:
            net.reset_state()
            y_hat = net(X.permute(1, 0, 2))
            l = loss(y_hat, y)

            trainer.zero_grad()  # 清除了优化器中的grad
            l.backward()  # 通过进行反向传播来计算梯度
            trainer.step()  # 通过调用优化器来更新模型参数
            if scheduler:
                scheduler.step()

        val_acc = evaluate_accuracy(net, val_iter, rnn=True)
        if val_acc > val_acc_best:
            val_acc_best = val_acc
            torch.save(net.state_dict(), path)

        print("epoch: %d    val_acc: %.2f%%" % (epoch + 1, val_acc * 100))

    if val_acc < val_acc_best:
        net.load_state_dict(torch.load(path))

    return 1 - evaluate_accuracy(net, test_iter, rnn=True)


def train_net_t(net, loss, trainer, data_iter, epochs, path, k_ratio, scheduler=None):
    train_iter, val_iter, test_iter = data_iter

    # net.apply_masks(next(iter(train_iter)), loss, k_ratio)
    # print("true sparsity: %f%%" % (net.sparsity_ratio * 100))

    val_acc_best = 0
    for epoch in range(epochs):
        net.train()
        i = -1
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            i += 1

            trainer.zero_grad()  # 清除了优化器中的grad
            l.backward()  # 通过进行反向传播来计算梯度
            net.save_grad()
            trainer.step()  # 通过调用优化器来更新模型参数
            if i == 0:
                net.remask(k_ratio, True)
            else:
                net.remask(k_ratio)
            if scheduler:
                scheduler.step()

        net.cal_mask_trans()

        val_acc = evaluate_accuracy(net, val_iter)
        if val_acc > val_acc_best:
            val_acc_best = val_acc
            torch.save(net.state_dict(), path)

        print("epoch: %d    val_acc: %.2f%%" % (epoch + 1, val_acc * 100))

    if val_acc < val_acc_best:
        net.load_state_dict(torch.load(path))

    return 1 - evaluate_accuracy(net, test_iter)