import pandas as pd
import torch
from matplotlib import ticker
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import normalize
import seaborn as sns
import sys

learningrateinput = float(sys.argv[1])
secondhyddenlayerinput = int(sys.argv[2])

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 18
torch.set_default_tensor_type(torch.DoubleTensor)
from sklearn.metrics import r2_score


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)


def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


Sourcedata = pd.read_excel(
    "D:/Doctor_Thesis/database and website/code/3_Different ML model training and selection with selected feature/ANNoptimize/dataset_origin.xlsx",
    sheet_name="Sheet2",
)
# standardScaler = MaxAbsScaler()
# standardScaler.fit(Sourcedata)
Sourcedata = normalize(Sourcedata)
X = torch.from_numpy(Sourcedata[:, :-1])
# print(X.size(1))
y = torch.from_numpy(Sourcedata[:, -1])


class simpleNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(simpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Activation_Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Activation_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.Tanh())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.Sigmoid())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# def weight_init(net):
# for m in net.modules():
#     if isinstance(m, nn.Conv2d):
#         init.xavier_uniform(m.weight)
#         if m.bias:
#             init.constant(m.bias, 0.5)
#     elif isinstance(m, nn.BatchNorm2d):
#         init.constant(m.weight, 1)
#         init.constant(m.bias, 0.5)
#     elif isinstance(m, nn.Linear):
#         init.normal_(m.weight)
# if m.bias:
#     init.constant(m.bias, 0.5)


# 2. 初始化网络结构


def get_kfold_data(k, i, X, y):
    fold_size = X.shape[0] // k
    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        X_valid, y_valid = X[val_start:val_end], y[val_start:val_end]
        X_train = torch.cat((X[0:val_start], X[val_end:]), dim=0)
        y_train = torch.cat((y[0:val_start], y[val_end:]), dim=0)
    else:
        X_valid, y_valid = X[val_start:], y[val_start:]
        X_train = X[0:val_start]
        y_train = y[0:val_start]
    return X_train, y_train, X_valid, y_valid


# def weight_init(m):
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)
#         nn.init.constant_(m.bias, 0.5)
#     # 也可以判断是否为conv2d，使用相应的初始化方式
#     elif isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#      # 是否为批归一化层
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)


def traink(
    X_train,
    y_train,
    X_val,
    y_val,
    epoch,
    learning_rate,
    number_in_second_layer,
    number_of_net,
):
    loss_func = torch.nn.MSELoss()

    net = Activation_Net(X_train.size(1), number_in_second_layer, number_in_second_layer, 1)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # net.apply(weight_init)

    # for parameters in net.parameters():
    #     print(parameters)    # print(net)
    loss11 = []
    loss22 = []
    loss33 = []
    loss44 = []
    # pred1 = []
    # test1 = []
    loss111 = pd.DataFrame()
    # prediction1 = ()
    # prediction2 = ()
    # loss1 = ()
    for t in range(epoch):

        prediction1 = net(X_train)
        prediction1 = prediction1.squeeze(-1)
        loss1 = loss_func(prediction1, y_train)
        # print("Epoch {}, loss: {}".format(t, loss1))
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        if t % 5 == 0:  # 每100步打印一次损失
            # plt.cla()
            prediction1 = net(X_train).squeeze(-1)
            prediction2 = net(X_val).squeeze(-1)
            # pred1.append(prediction2.data.numpy())
            # test1.append(y_val.data.numpy())
            loss11.append(loss_func(prediction1, y_train).data.numpy())
            loss22.append(loss_func(prediction2, y_val).data.numpy())
            loss33.append(mape(prediction1.data.numpy(), y_train.data.numpy()))
            loss44.append(mape(prediction2.data.numpy(), y_val.data.numpy()))
            # print("Epoch:{}, loss2:{:.6f}".format(t, loss22))
            # plt.scatter(X_val.data.numpy(), prediction2.data.numpy())
            # plt.pause(0.05)
    # print(net)
    torch.save(net, "net%s.pkl" % number_of_net)
    prediction2 = net(X_val).squeeze(-1).data.numpy()
    prediction2 = net(X_val).squeeze(-1).data.numpy()
    pred = pd.DataFrame([prediction2, y_val.data.numpy()])
    # print(pred)
    loss111 = pd.concat([loss111, pd.DataFrame(loss11)], axis=1)
    loss222 = pd.concat([loss111, pd.DataFrame(loss22)], axis=1)
    loss333 = pd.concat([loss222, pd.DataFrame(loss33)], axis=1)
    loss444 = pd.concat([loss333, pd.DataFrame(loss44)], axis=1)

    return loss444, pred


def k_fold(k, X, y, learning_rate):
    # train_loss_sum, valid_loss_sum = 0, 0
    # train_acc_sum, valid_acc_sum = 0, 0
    # loss111 = pd.DataFrame()
    # loss222 = pd.DataFrame()
    # loss333 = pd.DataFrame()
    # loss444 = pd.DataFrame()

    for j in range(secondhyddenlayerinput, secondhyddenlayerinput + 1, 1):
        loss_combined = pd.DataFrame()
        pred = pd.DataFrame()
        loss_curve = pd.DataFrame()
        for i in range(k):
            # print("*" * 25, "第", i + 1, "折", "*" * 25)
            data = get_kfold_data(k, i, X, y)  # 获取k折交叉验证的训练和验证数据
            # print("X_train.data.numpy()")
            # print(data[0])
            # print("y_train.data.numpy()")
            # print(data[1])
            loss444, pred1 = traink(*data, 5000, learning_rate, j, i)
            loss444.columns = [
                "mse in train",
                "mse in test",
                "mape in train",
                "MAPE in test",
            ]
            # print(pred1)
            pred = pd.concat([pred, pred1], axis=1)
            # loss44 = loss444.iloc[-1:, -1:]

            # pred111 = pred.iloc[:, -1:]
            loss_combined = pd.concat(
                [loss_combined, pd.DataFrame(loss444.iloc[-1:, -1:])], axis=0
            )
            loss_curve = pd.concat([loss_curve, loss444.iloc[50:, -1:]], axis=1)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        pred = pred.T
        pred.columns = ["predicted normalized EMY", "actual normalized EMY"]
        # print(pred)

        sns.scatterplot(
            data=pred,
            x=pred["actual normalized EMY"],
            y=pred["predicted normalized EMY"],
        )
        ax.set_xlabel("Actual normalized EMY")
        ax.set_ylabel("Predicted normalized EMY")
        plt.gcf().text(
            0.6,
            0.8,
            "R$^2$ = %.4f"
            % (
                r2_score(
                    pred["actual normalized EMY"], pred["predicted normalized EMY"]
                )
            ),
            fontsize=18,
        )
        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(left=0.165)
        plt.subplots_adjust(right=0.95)
        plt.subplots_adjust(top=0.95)
        sns.lineplot(x=[0, 1], y=[0, 1])
        fig.savefig("ANN predicted.svg")

        # fig2, ax2 = plt.subplots()
        # column2 = np.arange(1, 11, 1)
        # print(loss_curve)
        # loss_curve.columns = column2
        # ax2 = sns.lineplot(data=loss_curve)

        # print("第二层网格数%i" % j)
        print(loss_combined.mean())
        # print(loss_combined)
        # loss_combined = loss_combined.T
        # loss_combined.columns = np.arange(1,11,1)
        # print(loss_combined)
        fig3, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(
            loss_combined, x=np.arange(1, 11, 1), y=loss_combined["MAPE in test"]
        )
        ax.set_xlabel("Round in CV of ANN")
        ax.set_ylim([0, 0.18])
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        from matplotlib.ticker import MultipleLocator

        y_major_locator = MultipleLocator(0.02)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.gcf().text(
            0.4,
            0.9,
            "mean MAPE = %.2f%%" % (100 * loss_combined.mean()),
            fontsize=18,
        )
        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(left=0.165)
        plt.subplots_adjust(right=0.95)
        plt.subplots_adjust(top=0.95)
        fig3.savefig("ANN error.svg")
        loss_combined.to_csv("ANN ERROR list.csv")

    # print(
    #     # "train_loss:{:.5f}, train_acc:{:.3f}%".format(loss11, loss22)
    #     pd.DataFrame(loss11).T,
    #     "\n--------\n",
    #     pd.DataFrame(loss22).T,
    # )

    # print(loss111.T)
    # print(loss222.min(axis=0))
    # print(loss444.min(axis=0))
    # print(loss222.min(axis=0).mean())
    # print(loss444.min(axis=0).mean())
    # print(loss222.T)
    # print(loss333.T)
    # print(loss444.T)
    return


# loss1 = pd.DataFrame()
# loss2 = pd.DataFrame()
# loss3 = pd.DataFrame()
# loss4 = pd.DataFrame()

k_fold(10, X, y, learningrateinput)
# i=0.025,mapemean = 24.82
# loss1 = pd.concat([loss1, pd.DataFrame(loss111)], axis=1)
# loss2 = pd.concat([loss1, pd.DataFrame(loss222)], axis=1)
# loss3 = pd.concat([loss2, pd.DataFrame(loss333)], axis=1)
# loss4 = pd.concat([loss3, pd.DataFrame(loss444)], axis=1).T
# loss4.index = ["mse in train", "mse in test", "mape in train", "mape in test"]
# loss4.to_csv("ANN lossall.csv")
# print(loss1)
# print(loss2)
# print(loss3)
# print(loss4)
# fig, ax = plt.subplots(2, 2, figsize=(12, 6))
# ax1 = sns.barplot(data=loss1, ax=ax[0][0])
# ax1.set_ylabel("mse in train")
# ax2 = sns.barplot(data=loss2, ax=ax[0][1])
# ax2.set_ylabel("mse in test")
# ax3 = sns.barplot(data=loss3, ax=ax[1][0])
# ax3.set_ylabel("mape in train")
# ax4 = sns.barplot(data=loss4, ax=ax[1][1])
# ax4.set_ylabel("mape in test")
# plt.savefig("ANN error.png")
# print(loss1.T.mean())
# print(loss2.T.mean())
# print(loss3.T.mean())
# print(loss4.T.mean())


#########3.7
