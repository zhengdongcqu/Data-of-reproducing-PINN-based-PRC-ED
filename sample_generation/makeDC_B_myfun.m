function [B] = makeDC_B_myfun(baseMVA, bus, branch)
%MAKEB   Builds the DC power flow matrices B 建立直流潮流计算的B阵
%   [B] = makeDC_B(baseMVA, bus, branch) 
%   matrices B used in the DC power
%   flow. Does appropriate conversions to p.u.

%% define named indices into bus, branch matrices
[PQ, PV, REF, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, ...
	VA, BASE_KV, VMAX, VMIN, PGMAX,PGMIN,QGMAX,QGMIN,PG,QG,LOLP,LOLE,LOLF,LOLD,EDNS,EENS,...
    LAM_P,LAM_Q,MU_VMAX,MU_VMIN,MU_PMAX,MU_PMIN,MU_QMAX,MU_QMIN] = idx_bus;
[GEN_BUS, GEN_PMAX, GEN_PMIN, GEN_QMAX, GEN_QMIN, GEN_STATUS, ...
	GEN_LAMDA,GEN_MTTF,GEN_PG,GEN_QG,GEN_PFAILURE] = idx_gen;
[F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE, TAP, SHIFT_ANGLE, BR_STATUS, BR_LAMDA,...
        BR_MTTF,TCSC_X, BR_PFAILURE, PF, QF, PT, QT, MU_SF, MU_ST, KT_MAXANGLE_TPST,KT_MINANGLE_TPST,KT_MAX_X_TCSC,KT_MIN_X_TCSC] = idx_brch;

[NEWTON,FDFP_XB,FDFP_BX,DCPF] = idx_powerflow;

%% constants
nb = size(bus, 1);			%% number of buses
nl = size(branch, 1);		%% number of lines

%%-----  form Bp (B prime)  ----形成B前的简化
temp_branch = branch;						%% modify a copy of branch
temp_bus = bus;								%% modify a copy of bus
temp_bus(:, BS) = zeros(nb, 1);				%% zero out shunts at buses
temp_branch(:, BR_B) = zeros(nl, 1);		%% zero out line charging shunts
temp_branch(:, TAP) = ones(nl, 1);			%% cancel out taps
temp_branch(:, BR_R) = zeros(nl, 1);		%% zero out line resistance

B = -imag( makeYbus_myfun(baseMVA, temp_bus, temp_branch) );

return;
