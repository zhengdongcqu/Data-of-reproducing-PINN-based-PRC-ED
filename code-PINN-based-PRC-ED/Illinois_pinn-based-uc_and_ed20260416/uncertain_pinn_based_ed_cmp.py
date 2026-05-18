import numpy as np
import sys
import os
import scipy.io
import pandas as pd
sys.path.append(os.path.abspath('D:\PycharmProjects\pythonProject\PYPOWER-master'))
from pypower.api import makePTDF
import gurobipy as gp
from gurobipy import GRB

mat = scipy.io.loadmat('D:\PycharmProjects\pythonProject\Illinois_pinn-based-uc_and_ed20260416\ED_illinois200_data_for_training_PINN_2000.mat')
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
up_cost = mat['up_cost']
shut_cost = mat['shut_cost']
Set_RE = np.array(mat['Set_RE']).T.astype(int)
Set_TH = np.array(mat['Set_TH']).T.astype(int)
Set_Fast = np.array(mat['Set_Fast']).T.astype(int)
Pgmax = mat['Pgmax']
Pgmin = mat['Pgmin']
PL = mat['PL']
UG_da = mat['UG_da']
Pgmax_da = mat['Pgmax_da']
ramp = mat['Ramp_max']
T_up_down = mat['T_on_off'].astype(int)
EDNS_vector = mat['EDNS_vector']
LOLP_vector = mat['LOLP_vector']
EDNS_vector_mean = mat['EDNS_vector_mean']
EDNS_vector_std = mat['EDNS_vector_std']
LOLP_vector_mean = mat['LOLP_vector_mean']
LOLP_vector_std = mat['LOLP_vector_std']
EDNS_Target = mat['EDNS_target']
LOLP_Target = mat['LOLP_target']
dataX = mat['dataX']
dataY = mat['dataY'].reshape(-1, 1)
dataZ = mat['dataZ'].reshape(-1, 1)
dataYx0 = mat['dataYx0']
del mat
Set_not_Fast = np.setdiff1d(Set_TH, Set_Fast)

failure_matrix0 = failure_matrix
prob_failure0 = prob_failure

# 神经网络网架结构设计只考虑发电机故障
is_all_one = np.all(failure_matrix == 1, axis=1)  # 检查每行是否全为1
failure_matrix = failure_matrix[~is_all_one, :]
prob_failure = prob_failure[~is_all_one]

# load_shedding_events_num = load_shedding_events_num[~is_all_one]

# xita = 0.99
# def extract_indices_by_probability(prob_failure):
#     # 组合索引和概率
#     indexed_probs = list(enumerate(prob_failure))
#     # 按概率从大到小排序
#     sorted_probs = sorted(indexed_probs, key=lambda x: x[1], reverse=True)
#
#     total_prob = prob_failure0[0]
#     selected_indices = []
#
#     for idx, prob in sorted_probs:
#         if total_prob > xita:
#             break
#         total_prob += prob
#         selected_indices.append(idx)
#         # 添加精度检查避免浮点误差
#         if abs(total_prob - xita) < 1e-9:
#             break
#
#     return selected_indices
#
# selected_indices = extract_indices_by_probability(prob_failure)

N_failure_considered = 2000
# 组合索引和概率
indexed_probs = list(enumerate(prob_failure))
# 按概率从大到小排序
sorted_probs = sorted(indexed_probs, key=lambda x: x[1], reverse=True)
selected_indices = [idx for idx, prob in sorted_probs[:N_failure_considered]]

failure_matrix = failure_matrix[selected_indices, :]
prob_failure = prob_failure[selected_indices]

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

from conventional_pinn_conv import PINNs_CONV
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

hidden_dim = 128

model = PINNs_CONV(in_dim=Nb+Ng, hidden_dim=hidden_dim, out_dim=2).to(device)
model.load_state_dict(torch.load('D:\PycharmProjects\pythonProject\Illinois_pinn-based-uc_and_ed20260416\ed_conventional_pinns_test_2000_seed2.pt'))
for param in model.parameters():
    param.requires_grad = False

mat = scipy.io.loadmat('rolling_ed_K4_illinois200.mat')
rep_WT_Forecast_Error_squeezed = mat['rep_WT_Forecast_Error']
rep_WT_Forecast_Error = rep_WT_Forecast_Error_squeezed.tolist() \
    if isinstance(rep_WT_Forecast_Error_squeezed, np.ndarray) else list(rep_WT_Forecast_Error_squeezed)
rep_prob_Forecast_Error = np.array(mat['rep_prob_Forecast_Error']).T
del mat

Ns = rep_prob_Forecast_Error.shape[0]

units = Ng # the total number of thermal units
##Difining the model
m = gp.Model("reliability_constrained ED")
##Defining the decision variables
thermal_output = m.addMVar((units, Nt), 0, vtype=GRB.CONTINUOUS, name='thermal_output')
thermal_reserve = m.addMVar((units, Nt), 0, vtype=GRB.CONTINUOUS, name='thermal_reserve')
thermal_status = m.addMVar((units, Nt), vtype=GRB.BINARY, name='thermal_status')
u = m.addMVar((units, Nt), vtype=GRB.BINARY, name='thermal_up')
v = m.addMVar((units, Nt), vtype=GRB.BINARY, name='thermal_down')

yita_EDNS = m.addMVar((1, Nt), 0, vtype=GRB.CONTINUOUS)
yita_LOLP = m.addMVar((1, Nt), 0, vtype=GRB.CONTINUOUS)

## seting the objective function
M = 1e4
total_cost = gp.quicksum(
    pg_cost[i] * (thermal_output[i, t] + thermal_reserve[i, t]) * baseMVA + u[i, t] * up_cost[i] + v[i, t] * shut_cost[i] + M * yita_EDNS[:, t] + M * yita_LOLP[:,t]
    for i in range(units)  # 外层循环：机组 i
    for t in range(Nt)  # 内层循环：时段 t
)

m.setObjective(total_cost, GRB.MINIMIZE)

m.addConstrs((yita_EDNS[:, t] >= 0 for t in range(Nt)))
m.addConstrs((yita_LOLP[:, t] >= 0 for t in range(Nt)))
for t in range(Nt):
    ## Constranits of the operating
    m.addConstrs((thermal_output[i, t] + thermal_reserve[i, t] >= Pgmin[i, t] * thermal_status[i, t] for i in range(units)),name='generation_min')
    m.addConstrs((thermal_output[i, t] + thermal_reserve[i, t] <= Pgmax[i, t] * thermal_status[i, t] for i in range(units)),name='generation_max')

    m.addConstrs((thermal_output[i, t] + thermal_reserve[i, t] <= np.minimum(Pgmax[i, t], Pgmax_da[i, t] + ramp[i]) for i in range(units)),name='generation_da_max')
    m.addConstrs((np.maximum(0, Pgmax_da[i, t] - ramp[i]) <= thermal_output[i, t] + thermal_reserve[i, t] for i in range(units)), name='generation_da_min')

    ## Constranits of the load demanding
    m.addConstr(gp.quicksum(thermal_output[i, t] for i in range(units)) == PL[:,t].sum(), name='power_balance')

    ## Constranits of the reserve
    m.addConstr(gp.quicksum(thermal_reserve[i, t] for i in range(units)) >= 0.15 * PL[:, t].sum(), name='reserve_requirement1')
    m.addConstr(gp.quicksum(thermal_reserve[i, t] for i in range(units)) <= 0.25 * PL[:, t].sum(), name='reserve_requirement2')
    m.addConstrs((thermal_reserve[i, t] <= ramp[i] for i in range(units)),name='reserve_ramp')

    ## Constranits of the power flow
    m.addConstr(PTDF @ (G @ thermal_output[:,t] - PL[:,t]) <= Fmax, name='power_flow_upper_bound')
    m.addConstr(PTDF @ (G @ thermal_output[:,t] - PL[:,t]) >= -Fmax, name='power_flow_lower_bound')

# m.addConstrs(thermal_reserve[Set_RE[i,:].item()-1, :]==0 for i in range(Set_RE.shape[0]))
m.addConstrs(thermal_status[Set_RE[i,:].item()-1, :]==1 for i in range(Set_RE.shape[0]))
m.addConstrs(thermal_status[Set_not_Fast[i]-1, :] == UG_da[Set_not_Fast[i]-1, :] for i in range(Set_not_Fast.shape[0])) # 固定机组组合方案

for i in range(units):
    ## Constranits of the ramp
    m.addConstrs((thermal_output[i, t+1] - thermal_output[i, t] <= ramp[i] for t in range(0, Nt-1)), name='ramp_up')
    m.addConstrs((thermal_output[i, t+1] - thermal_output[i, t] >= -ramp[i] for t in range(0, Nt-1)), name='ramp_down')

    m.addConstrs((u[i, t] >= thermal_status[i, t] - thermal_status[i, t-1] for t in range(1, Nt)), name='up_indicators')
    m.addConstr(u[i, 0] >= thermal_status[i, 0], name='up_indicators_t0')

    m.addConstrs((v[i, t] >= thermal_status[i, t-1] - thermal_status[i, t] for t in range(1, Nt)), name='down_indicators')
    m.addConstr(v[i, 0] >= 1 - thermal_status[i, 0], name='down_indicators_t0')

    for t in range(Nt - T_up_down[i].item() + 1):
        # 最小开机时间约束
        m.addConstr(gp.quicksum(thermal_status[i, j] for j in range(t, t + T_up_down[i].item())) >= T_up_down[i] * u[i, t], name=f'min_on_time_i{i}_t{t}')
        # 最小停机时间约束
        m.addConstr(gp.quicksum(1 - thermal_status[i, j] for j in range(t, t + T_up_down[i].item())) >= T_up_down[i]* v[i, t], name=f'min_off_time_i{i}_t{t}')

import time

Set_EDNS = 100 * np.ones([Nt,3])
Set_LOLP = 100 * np.ones([Nt,3])

Set_EDNS[:,2] = EDNS_Target
Set_LOLP[:,2] = LOLP_Target

start = time.time()  # 开始时间
k = 0
while True:

    m.optimize()
    if k == 1:
        Pg_avai_before_dispatching = thermal_output.X + thermal_reserve.X
        total_cost_before = m.objVal
        Set_EDNS[:,0] = Set_EDNS[:,1]
        Set_LOLP[:,0] = Set_LOLP[:,1]

    k += 1

    EDNS_satisfy_or_not = (Set_EDNS[:, 1].reshape(-1, 1) - EDNS_Target) < 1e-5
    LOLP_satisfy_or_not = (Set_LOLP[:, 1].reshape(-1, 1) - LOLP_Target) < 1e-5

    if (max(Set_EDNS[:,1].reshape(-1,1) - EDNS_Target) < 1e-5 and max(Set_LOLP[:,1].reshape(-1,1) - LOLP_Target) < 1e-5) or k > 50:
        total_cost_final = m.objVal
        break

    RATE = np.sum(thermal_output.X + thermal_reserve.X, 0) / np.sum(PL, 0)
    for t in range(Nt):
        thermal_status_current = thermal_status[:, t].X
        thermal_output_current = thermal_output[:, t].X
        thermal_reserve_current = thermal_reserve[:, t].X

        # 创建一个数组来存储所有场景的x
        x_all = []
        for s in range(Ns):
            rep_WT_Forecast_Error_s = np.array(rep_WT_Forecast_Error[s]).squeeze(0)
            x = (thermal_output_current + thermal_reserve_current).copy()
            tmp1 = (1 + rep_WT_Forecast_Error_s[:, t]) * Pgmax[Set_RE - 1, t].squeeze(1)
            tmp2 = x[Set_RE - 1].squeeze(1)
            min_result = np.minimum(tmp1, tmp2).reshape(-1, 1)
            x[Set_RE - 1] = min_result
            x = np.concatenate([x, PL[:, t]])
            x_all.append(x.reshape(1, -1))

        # 将所有场景的x堆叠成一个张量
        x_all = np.vstack(x_all)  # 形状: (Ns, Ng+Nb)
        x_tensor = torch.FloatTensor(x_all).requires_grad_(True).to(device)

        # 一次性调用两个神经网络
        EDNS_t_s, LOLP_t_s = model(x_tensor)

        # 计算梯度
        x_grad_EDNS_tensor = torch.autograd.grad(
            EDNS_t_s.sum(), x_tensor,
            grad_outputs=torch.ones_like(EDNS_t_s.sum()),
            retain_graph=True,
            create_graph=True
        )[0]

        x_grad_LOLP_tensor = torch.autograd.grad(
            LOLP_t_s.sum(), x_tensor,
            grad_outputs=torch.ones_like(LOLP_t_s.sum()),
            retain_graph=True,
            create_graph=True
        )[0]

        # 反标准化
        EDNS_t_s = EDNS_vector_std * EDNS_t_s.to(device).cpu().detach().numpy().squeeze() + EDNS_vector_mean
        LOLP_t_s = LOLP_vector_std * LOLP_t_s.to(device).cpu().detach().numpy().squeeze() + LOLP_vector_mean

        # 计算加权平均值
        EDNS_t = np.dot(rep_prob_Forecast_Error.T, EDNS_t_s.reshape(-1, 1))
        LOLP_t = np.dot(rep_prob_Forecast_Error.T, LOLP_t_s.reshape(-1, 1))

        # 计算梯度加权平均值
        x_grad_EDNS_weighted = EDNS_vector_std * (
                    rep_prob_Forecast_Error.reshape(1, -1) @ x_grad_EDNS_tensor.cpu().detach().numpy())
        x_grad_LOLP_weighted = LOLP_vector_std * (
                    rep_prob_Forecast_Error.reshape(1, -1) @ x_grad_LOLP_tensor.cpu().detach().numpy())

        Set_EDNS[t, 1] = EDNS_t.item()
        Set_LOLP[t, 1] = LOLP_t.item()

        # Benders Cuts
        if EDNS_t > EDNS_Target:
            m.addConstr(
                EDNS_t - EDNS_Target +
                x_grad_EDNS_weighted[:, 0:Ng].reshape(-1, 1).T @
                (thermal_output[:, t] + thermal_reserve[:, t] - (thermal_output_current + thermal_reserve_current))
                <= yita_EDNS[:, t]
            )

        if LOLP_t > LOLP_Target:
            m.addConstr(
                LOLP_t - LOLP_Target +
                x_grad_LOLP_weighted[:, 0:Ng].reshape(-1, 1).T @
                (thermal_output[:, t] + thermal_reserve[:, t] - (thermal_output_current + thermal_reserve_current))
                <= yita_LOLP[:, t]
            )

end = time.time()    # 结束时间
cpu_time = end - start
print(f"耗时: {cpu_time:.4f} 秒")

result_thermal_output = thermal_output.X
result_thermal_reserve = thermal_reserve.X

# 验证
# 先用发电系统进行验证
Pg_avai = result_thermal_output + result_thermal_reserve
x1 = np.sum(PL.T, 1)
x2 = np.matmul(Pg_avai.T, failure_matrix0.T)
x0 = x1.reshape(-1, 1) - x2
x0[x0 < 0] = 0
flag = x0.copy()
flag[flag > 0] = 1
ends_gs = np.matmul(x0, prob_failure0)
lolp_gs = np.matmul(flag, prob_failure0)
# 预测
x = np.concatenate([Pg_avai, PL])
x = torch.FloatTensor(x.T).requires_grad_(True).to(device)
EDNS_test, LOLP_test = model(x)
EDNS_test = EDNS_vector_std * EDNS_test.to(device).cpu().detach().numpy().squeeze() + EDNS_vector_mean
LOLP_test = LOLP_vector_std * LOLP_test.to(device).cpu().detach().numpy().squeeze() + LOLP_vector_mean
CMP_EDNS = np.column_stack((EDNS_test.T, ends_gs))
CMP_LOLP = np.column_stack((LOLP_test.T, lolp_gs))

CMP_EDNS_max = np.max(CMP_EDNS, axis=0)  # 对每一行求平均
CMP_LOLP_max = np.max(CMP_LOLP, axis=0)

Set_EDNS[:, 1] = EDNS_test.T.squeeze()
Set_LOLP[:, 1] = LOLP_test.T.squeeze()

delta_Pg_avai = Pg_avai - Pg_avai_before_dispatching

# 准备时段和机组名称
hours = [f'时段{t + 1}' for t in range(Nt)]
units_names = [f'机组{i + 1}' for i in range(units)]

# 创建各结果的数据框
df_Pg_avai = pd.DataFrame(Pg_avai.T, columns=units_names, index=hours)
df_Pg_avai_before_dispatching = pd.DataFrame(Pg_avai_before_dispatching.T, columns=units_names, index=hours)
df_delta_Pg_avai = pd.DataFrame(delta_Pg_avai.T, columns=units_names, index=hours)

# 准备时段和机组名称
hours = [f'时段{t+1}' for t in range(Nt)]
units_names = [f'机组{i+1}' for i in range(units)]

# 创建各结果的数据框
df_Pg_avai = pd.DataFrame(Pg_avai, columns=hours, index=units_names)
df_Pg_avai_before_dispatching = pd.DataFrame(Pg_avai_before_dispatching, columns=hours, index=units_names)
df_delta_Pg_avai = pd.DataFrame(delta_Pg_avai, columns=hours, index=units_names)

# 创建EDNS和LOLP数据框
df_edns = pd.DataFrame({
    '迭代前EDNS': 100 * Set_EDNS[:, 0],
    '迭代后EDNS': 100 * Set_EDNS[:, 1],
    '目标EDNS': 100 * Set_EDNS[:, 2]
}, index=hours)

df_lolp = pd.DataFrame({
    '迭代前LOLP': Set_LOLP[:, 0],
    '迭代后LOLP': Set_LOLP[:, 1],
    '目标LOLP': Set_LOLP[:, 2]
}, index=hours)

df_time = pd.DataFrame({
    '计算时间': [cpu_time]
})

filename = 'illinois200_uncertain_pinn_based_ed_results_K4_cmp.xlsx'

try:
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 保存机组相关结果
        df_Pg_avai.to_excel(writer, sheet_name='机组可用容量')
        df_Pg_avai_before_dispatching.to_excel(writer, sheet_name='调度前机组可用容量')
        df_delta_Pg_avai.to_excel(writer, sheet_name='机组可用容量差值')

        # 保存可靠性指标
        df_edns.to_excel(writer, sheet_name='EDNS结果')
        df_lolp.to_excel(writer, sheet_name='LOLP结果')

        # 保存计算时间
        df_time.to_excel(writer, sheet_name='计算时间')

    print(f"优化结果已成功保存到Excel文件: {filename}")
    print("包含以下工作表:")
    sheets = ['机组可用容量', '调度前机组可用容量', '机组可用容量差值', 'EDNS结果', 'LOLP结果']
    for sheet in sheets:
        print(f"- {sheet}")

except Exception as e:
    print(f"保存Excel文件时发生错误: {e}")

# m.write('output.sol')