import matplotlib.pyplot as plt
import pandas as pd

import scipy.io
import sys
import os

import torch
import random
from torch.optim import LBFGS
from tqdm import tqdm

import torch.nn.functional as F

from pinnsformer_main.util import *
from improved_pinn_conv import PINNs_CONV

sys.path.append(os.path.abspath('D:\PycharmProjects\pythonProject\PYPOWER-master'))
from pypower.api import makePTDF


# 设置随机种子
def set_seed(seed=0):
    """
    设置随机种子以确保结果可重现

    Args:
        seed: 随机种子值
    """
    # 设置Python的随机种子
    random.seed(seed)

    # 设置NumPy的随机种子
    np.random.seed(seed)

    # 设置PyTorch的随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU

    # 设置CuDNN以确保确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"随机种子已设置为: {seed}")


# 设置随机种子（放在所有随机操作之前）
seed = 0
set_seed(seed)

mat = scipy.io.loadmat('ED_RTS79_data_for_training_PINN_test_5000_imp.mat')
baseMVA = np.array(mat['baseMVA']).T
Ng = int(mat['Ng'])
Nb = int(mat['Nb'])
Nl = int(mat['Nl'])
Nt = int(mat['Nt'])
gen = mat['gen']
bus = mat['bus']
branch = mat['branch']
G = mat['Ag']
prob_failure = mat['prob_failure']
failure_matrix = mat['pg_state']
pg_cost = mat['cost']
Set_RE = np.array(mat['Set_RE']).astype(int)
Pgmin = mat['Pgmin']
ramp = mat['Ramp_max']
T_up_down = mat['T_on_off'].astype(int)
EDNS_vector_mean = mat['EDNS_vector_mean']
EDNS_vector_std = mat['EDNS_vector_std']
LOLP_vector_mean = mat['LOLP_vector_mean']
LOLP_vector_std = mat['LOLP_vector_std']
dataX = mat['dataX']
dataY = mat['dataY'].reshape(-1, 1)
dataZ = mat['dataZ'].reshape(-1, 1)
dataYx0 = mat['dataYx0']
del mat

# print(bus[:,2].sum())

# 神经网络网架结构设计只考虑发电机故障
is_all_one = np.all(failure_matrix == 1, axis=1)  # 检查每行是否全为1
failure_matrix = failure_matrix[~is_all_one, :]
prob_failure = prob_failure[~is_all_one]

N_failure_considered = 2000
# 组合索引和概率
indexed_probs = list(enumerate(prob_failure))
# 按概率从大到小排序
sorted_probs = sorted(indexed_probs, key=lambda x: x[1], reverse=True)
selected_indices = [idx for idx, prob in sorted_probs[:N_failure_considered]]

failure_matrix = failure_matrix[selected_indices, :]
prob_failure = prob_failure[selected_indices]

# prob_failure = prob_failure / prob_failure.sum()

Fmax = branch[:,5]

bus[:,0] = bus[:,0] - 1
branch[:,0] = branch[:,0] - 1
branch[:,1] = branch[:,1] - 1

bus_ptdf = np.zeros([Nb, 4])
bus_ptdf [:,0] = bus[:,0] # BUS_I
bus_ptdf [:,1] = bus[:,1] # BUS_TYPE
bus_ptdf [:,3] = bus[:,2] # REF

branch_ptdf = np.zeros([Nl, 11])
branch_ptdf [:,0] = branch[:,0] # F_BUS
branch_ptdf [:,1] = branch[:,1] # T_BUS
branch_ptdf [:,3]= branch[:,3] # BR_X
branch_ptdf [:,8]= np.ones(Nl) # TAP
branch_ptdf [:,9]= np.zeros(Nl) # SHIFT
branch_ptdf [:,10]= np.ones(Nl) # BR_STATUS

PTDF = makePTDF(baseMVA, bus_ptdf, branch_ptdf)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

dataX_train = dataX
dataY_train = dataY
dataZ_train = dataZ
dataYx0_train = dataYx0
N_train = dataX.shape[0]

dataX_train = torch.tensor(dataX_train.squeeze() , dtype=torch.float, requires_grad=True).to(device)
dataY_train = torch.tensor(dataY_train.squeeze() , dtype=torch.float).to(device)
dataZ_train = torch.tensor(dataZ_train.squeeze() , dtype=torch.float).to(device)
dataYx0_train = torch.tensor(dataYx0_train.squeeze() , dtype=torch.float).to(device)

x = dataX_train
y = dataY_train
z = dataZ_train
u_x = dataYx0_train

# 读取验证集
mat = scipy.io.loadmat('ED_RTS79_data_for_testing_PINN_test_750_imp.mat')
EDNS_vector_mean_vali = mat['EDNS_vector_mean']
EDNS_vector_std_vali = mat['EDNS_vector_std']
LOLP_vector_mean_vali = mat['LOLP_vector_mean']
LOLP_vector_std_vali = mat['LOLP_vector_std']
dataX_vali = mat['dataX']
dataY_vali = mat['dataY'].reshape(-1, 1)
dataZ_vali= mat['dataZ'].reshape(-1, 1)
del mat

dataX_vali = torch.tensor(dataX_vali.squeeze() , dtype=torch.float, requires_grad=True).to(device)
x_vali = dataX_vali
real_edns_vali = EDNS_vector_mean_vali + dataY_vali * EDNS_vector_std_vali
real_lolp_vali = LOLP_vector_mean_vali + dataZ_vali * LOLP_vector_std_vali

def plot_loss_curves(loss_track):
    # 解压损失数据
    epochs = range(1, len(loss_track) + 1)
    loss_res = [item[0] for item in loss_track]
    loss_bc = [item[1] for item in loss_track]
    loss_sum = [item[2] for item in loss_track]

    plt.figure(figsize=(15, 5))

    # 1. 绘制普通坐标系曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_res, 'b.-', label='LOLP loss (target)')
    plt.plot(epochs, loss_bc, 'r.-', label='Boundary Loss (bc)')
    plt.plot(epochs, loss_sum, 'g.-', label='Total Loss (sum)')

    plt.title('Training Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # 2. 绘制对数坐标系曲线（纵轴对数）
    plt.subplot(1, 2, 2)
    plt.semilogy(epochs, loss_res, 'b.-', label='LOLP Loss (target)')
    plt.semilogy(epochs, loss_bc, 'r.-', label='Boundary Loss (bc)')
    plt.semilogy(epochs, loss_sum, 'g.-', label='Total Loss (sum)')

    plt.title('Logarithmic Scale Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.grid(True, which="both", ls="--")
    plt.legend()

    plt.tight_layout()
    plt.show()

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss <= self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

# 应用 He 初始化
def init_weights(m):
    if isinstance(m, nn.Linear): # 判断是否为线性层
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0) # 偏置通常初始化为0

# EDNS网络
hidden_dim = 128
model = PINNs_CONV(in_dim=Nb+Ng, hidden_dim=hidden_dim, out_dim=1, failure_matrix=failure_matrix, prob_failure=prob_failure, type=1).to(device)
model.apply(init_weights)
print(model)

optim = LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=50,
    tolerance_grad=1e-32,
    tolerance_change=1e-32,
    history_size=100,
    line_search_fn='strong_wolfe'
)

early_stopper = EarlyStopping(patience=50)

import time

start = time.time()  # 开始时间

loss_track = []

for i in tqdm(range(1000)):
    def closure():
        model.train()
        pred_x1, pred_x2, pred_x3 = model(x)

        pred_u_x = torch.autograd.grad(pred_x1, x, grad_outputs=torch.ones_like(pred_x1), retain_graph=True, create_graph=True)[0]

        loss_res = F.l1_loss(pred_x1, y)
        loss_bc = 10 * torch.mean(torch.sum(torch.abs(pred_u_x - u_x), dim=1))
        loss_ae = 0.01 * torch.mean(torch.sum((pred_x2 - pred_x3)**2, dim=1))
        # torch.mean(torch.sum((pred_u_x - u_x)**2, dim=1))

        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)**2

        lambda_l2 = 1e-3

        loss_sum = loss_res + loss_bc + loss_ae + lambda_l2 * l2_reg

        loss_track.append([loss_res.item(), loss_bc.item(), loss_sum.item()])

        optim.zero_grad()
        loss_sum.backward()
        return loss_sum

    loss_sum = optim.step(closure)

    with torch.no_grad():
        model.eval()
        pred_x1_vali, _, _ = model(x_vali)
        pred_edns_vali = pred_x1_vali.cpu().detach().numpy() * EDNS_vector_std + EDNS_vector_mean
    error_edns_vali  = np.mean(np.abs(real_edns_vali - pred_edns_vali.T) / real_edns_vali)

    early_stopper(val_loss=error_edns_vali)

    if early_stopper.early_stop:
        break

    # if error_edns_vali < 0.01:
    #     break

    print('\nLoss Res: {:4f}, loss_bc: {:4f}, Loss_sum: {:4f}'.format(loss_track[-1][0], loss_track[-1][1], loss_track[-1][2]))
    print('Train Loss: {:4f}'.format(loss_track[-1][2]))
    print('Vali EDNS Error: {:4f}'.format(error_edns_vali*100))

end = time.time()    # 结束时间
cpu_time = end - start

loss_res = np.array([item[0] for item in loss_track])
loss_bc = np.array([item[1] for item in loss_track])
loss_sum = np.array([item[2] for item in loss_track])

plot_loss_curves(loss_track)

torch.save(model.state_dict(), 'D:\PycharmProjects\pythonProject\RTS79_pinn-based-uc_and_ed20260404\ed_pinns_EDNS_test_5000_seed0.pt')

model = PINNs_CONV(in_dim=Nb+Ng, hidden_dim=hidden_dim, out_dim=1, failure_matrix=failure_matrix, prob_failure=prob_failure, type=1).to(device)
model.load_state_dict(torch.load('D:\PycharmProjects\pythonProject\RTS79_pinn-based-uc_and_ed20260404\ed_pinns_EDNS_test_5000_seed0.pt'))

with torch.no_grad():
    train_pred, _, _ = model(x)
    train_pred = EDNS_vector_std * train_pred.cpu().detach().numpy() + EDNS_vector_mean

train_real = EDNS_vector_std * y.cpu().detach().numpy() + EDNS_vector_mean

train_error = np.mean(np.abs(train_real-train_pred) / train_real)

print('train error: {:4f}'.format(train_error * 100))

mat = scipy.io.loadmat('ED_RTS79_data_for_testing_PINN_test_750_imp.mat')
EDNS_vector_mean_test = mat['EDNS_vector_mean']
EDNS_vector_std_test = mat['EDNS_vector_std']
LOLP_vector_mean_test = mat['LOLP_vector_mean']
LOLP_vector_std_test = mat['LOLP_vector_std']
dataX_test = mat['dataX']
dataY_test = mat['dataY'].reshape(-1, 1)
dataZ_test = mat['dataZ'].reshape(-1, 1)
del mat

x_test = torch.tensor(dataX_test.squeeze(), dtype=torch.float).to(device)

with torch.no_grad():
    test_pred, _, _ = model(x_test)
    test_pred = EDNS_vector_std * test_pred.cpu().detach().numpy() + EDNS_vector_mean

test_real = EDNS_vector_std_test * dataY_test + EDNS_vector_mean_test

test_error = np.mean(np.abs(test_real-test_pred.T) / test_real)

print('test error: {:4f}'.format(test_error * 100))

df_train = pd.DataFrame({
    'train_real': train_real.squeeze(),
    'train_pred': train_pred.squeeze(),
})

df_test = pd.DataFrame({
    'test_real': test_real.squeeze(),
    'test_pred': test_pred.squeeze()
})

df_train_loss_track = pd.DataFrame({
    'loss_res': loss_res.squeeze(),
    'loss_bc': loss_bc.squeeze(),
})

df_time = pd.DataFrame({
    'cpu_time': [cpu_time]
})

# 2. 使用ExcelWriter保存到不同sheet
with pd.ExcelWriter('RTS79_pinn_training_EDNS_output_test_5000_seed0.xlsx') as writer:
    df_train.to_excel(writer, sheet_name='训练集', index=False)
    df_test.to_excel(writer, sheet_name='测试集', index=False)
    df_train_loss_track.to_excel(writer, sheet_name='损失函数', index=False)
    df_time.to_excel(writer, sheet_name='计算时间', index=False)