clear all
clc
close all
delete(gcp('nocreate')); 

[PQ, PV, REF, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, ...
    VA, BASE_KV, VMAX, VMIN, PGMAX,PGMIN,QGMAX,QGMIN,PG,QG,LOLP,LOLE,LOLF,LOLD,EDNS,EENS,...
    LAM_P,LAM_Q,MU_VMAX,MU_VMIN,MU_PMAX,MU_PMIN,MU_QMAX,MU_QMIN] = idx_bus;
[GEN_BUS, GEN_PMAX, GEN_PMIN, GEN_QMAX, GEN_QMIN, GEN_STATUS, ...
    GEN_LAMDA,GEN_MTTF,GEN_PG,GEN_QG,GEN_PFAILURE] = idx_gen;
[F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE, TAP, SHIFT_ANGLE, BR_STATUS, BR_LAMDA,...
    BR_MTTF,TCSC_X, BR_PFAILURE, PF, QF, PT, QT, MU_SF, MU_ST, KT_MAXANGLE_TPST,KT_MINANGLE_TPST,KT_MAX_X_TCSC,KT_MIN_X_TCSC] = idx_brch;
[NEWTON,FDFP_XB,FDFP_BX,DCPF] = idx_powerflow;

casename =  'case_illinois200';
case_illinois200=loadcase(casename);
baseMVA=case_illinois200.baseMVA;
bus=case_illinois200.bus;
gen=case_illinois200.gen;
gencost=case_illinois200.gencost;
branch=case_illinois200.branch;

Ng=size(gen,1);
Nb=size(bus,1);
Nl=size(branch,1);

bus(:,3)=bus(:,3)/baseMVA;
gen(:,9)=gen(:,9)/baseMVA;
branch(:,6)=branch(:,6)/baseMVA;

Set_Fast=[29,30];
N_Fast=length(Set_Fast);

% lambda_line=0.001+(0.005-0.001)*rand(Nl,1);
% lambda_gen=0.01+(0.05-0.01)*rand(Ng,1);

load('lambda_gen_illinois200.mat')
load('lambda_line_illinois200.mat')

Set_RE=[25:28];
N_RE=length(Set_RE);

Set_TH=setdiff([1:Ng]',Set_RE);
N_TH=length(Set_TH);

sum(gen(Set_RE,9))/sum(gen(:,9))
sum(gen(Set_TH,9))/sum(bus(:,3))

for i=1:Nb
    pgmax=sum(gen(gen(:,1)==i,9));
    for j=1:Nl
        if ismember(i,branch(j,1:2))
            if branch(j,6)<pgmax
                branch(j,6)=pgmax;
            end
        end
    end
end

%%  construct the DC optimal power flow problem of DC OPA, and use gurobi to solve it.
A=zeros(size(bus,1),size(branch,1));%A用于列写节点平衡约束，A1用于列写直流潮流约束  %可以修改，从而提高其可迁移性
Ag=zeros(size(bus,1),size(gen,1));
for i=1:size(bus,1)
    for j=1:size(branch,1)
        if branch(j,1)==i
            A(i,j)=1;
        elseif branch(j,2)==i
            A(i,j)=-1;
        end
    end
    for j=1:size(gen,1)
        if i==gen(j,1)
            Ag(i,j)=1;
        end
    end
end

rate=1./gen(:,9);
rate=rate/max(rate);
cost=10+50*rate;
cost(Set_RE)=min(cost)/2;
up_cost=500+500*gen(:,9)/max(gen(:,9));
shut_cost=up_cost/2;
T_on_off=ceil(6*gen(:,9)/max(gen(:,9)));
T_on_off(Set_Fast)=1;

Nt=24;

Pgmin=repmat(0.20*gen(:,9),1,Nt);
Pgmin(Set_RE,:)=0;

Ramp_max=0.50*gen(:,9);

Fmax=branch(:,6);

Set_root_gen={};
Set_root_line={};
Set_root_line_A={}; % 用于生成A矩阵

kmax_gen=3;
kmax_line=1;
n_root=2;
n_root_line=n_root;
for k=1:kmax_gen
    tmp=nchoosek([1:Ng],k);
    for i=1:length(tmp)
        Set_root_gen{n_root,1}=tmp(i,:);
        n_root=n_root+1;
    end
end

for k=1:kmax_line
    tmp=nchoosek([1:Nl],k);
    for i=1:length(tmp)
        Set_root_gen{n_root,1}=[];
        Set_root_line{n_root,1}=tmp(i,:);
        Set_root_line_A{n_root_line,1}=tmp(i,:);
        n_root=n_root+1;
        n_root_line=n_root_line+1;
    end
end

failure_set={Set_root_gen,Set_root_line};
N_failure=size(failure_set{1,1},1);

prob_failure=zeros(N_failure,1);
pg_state=ones(N_failure,Ng);
for root=1:N_failure
    gen_failure_num=failure_set{1,1}{root,1};
    line_failure_num=failure_set{1,2}{root,1};
    prob_failure(root)=prod(lambda_gen(gen_failure_num))*prod(1-lambda_gen(setdiff([1:Ng]',gen_failure_num)))*...
        prod(lambda_line(line_failure_num))*prod(1-lambda_line(setdiff([1:Nl]',line_failure_num)));
    pg_state(root,gen_failure_num)=0;
end

prob_failure_descend = sort(prob_failure, 'descend');

N_line_failure=size(Set_root_line_A,1);
Set_A=cell(N_line_failure,1);

state_map_A = containers.Map('KeyType', 'char', 'ValueType', 'double');
for root=1:N_line_failure
    tebranch = branch;
    line_failure_num=Set_root_line_A{root,1};
    if ~isempty(line_failure_num)
        tebranch(line_failure_num,RATE) = 0;
        tebranch(line_failure_num,BR_X) = 1e8;
    end
    state_key=mat2str(line_failure_num);
    state_map_A(state_key)=root;

    G = sparse(gen(:,1),1:Ng,ones(1,Ng),Nb,Ng);
    
    Inb = eye(Nl);
    Inl = eye(Nb);
    Ing = eye(Ng);
    B = makeDC_B_myfun(baseMVA,bus,tebranch);
    
    from = branch(:, F_BUS);
    to = branch(:, T_BUS);
    Cf = sparse(1:Nl,from,1./tebranch(:,BR_X), Nl, Nb);
    Ct = sparse(1:Nl,to,1./tebranch(:,BR_X), Nl, Nb);
    
    B1 = Cf-Ct;%下标相同的相减，如果没有对应的就等于本身,就是支路导纳矩阵乘以电纳
    Fmax =  tebranch(:,RATE);
    
    % x=[相角，发电机出力，切负荷，辅助变量]
    % y=[负荷，线路容量，线路容量，发电容量，负荷]
    
    A_tmp = [-B,G,Inl,zeros(Nb,2*Nl+Ng+Nb);%功率平衡约束
        
    B1,zeros(Nl,Ng+Nb),Inb,zeros(Nl,Nl+Ng+Nb);%Inb为辅助变量，线路容量约束
    
    -B1,zeros(Nl,Ng+Nb+Nl),Inb,zeros(Nl,Ng+Nb);%Inb为辅助变量，线路容量约束
    
    zeros(Ng,Nb),Ing,zeros(Ng,Nb+2*Nl),Ing,zeros(Ng,Nb);%Ing为辅助变量，发电容量约束
    
    zeros(Nb,Nb+Ng),Inl,zeros(Nb,2*Nl+Ng),Inl];%Inl为辅助变量，切负荷约束

    Set_A{root,1}=A_tmp;
end

%% 拉丁超立方抽样
Ns_each_t=1000;
Ns=Nt*Ns_each_t;
n_dim=N_Fast+Ng+Nb+1;
n_level=5;

edges = linspace(0, 1, n_level+1);
intervals = cell(1, n_level);
for i = 1:n_level
    intervals{i} = [edges(i), edges(i+1)];
end

x=[];
while(1)
    x_tmp=zeros(n_level,n_dim);
    for k=1:n_dim
        level=randperm(n_level); %每个维度都重新摇
        for j=1:n_level
            x_tmp(j,k)=intervals{level(j)}(1)+rand*(intervals{level(j)}(2)-intervals{level(j)}(1));
        end
    end
    x=[x;x_tmp];
    if size(x,1)>=Ns
        break
    end
end

x=x';

% K=4; % 典型场景数量
% 
% [wind_data,wind_data_raw] = xlsread('风电出力历史数据.xlsx');
% wind_data = wind_data(:,end)/max(wind_data(:,end));
% hourly_wind_output = mean(reshape(wind_data, 4, []), 1);
% hourly_wind_output = reshape(hourly_wind_output, 24, [])'; 
% [~, rep_hourly_wind_output] = kmeans(hourly_wind_output,K);
% % plot([1:24]',rep_hourly_wind_output)
% 
% [load_demand_data,load_demand_data_raw] = xlsread('澳大利亚电力负荷与价格预测数据.xlsx');
% load_demand_data = 0.50 + 0.50 *(load_demand_data(:,end)-min(load_demand_data(:,end)))/(max(load_demand_data(:,end))-min(load_demand_data(:,end)));
% hourly_load_demand = mean(reshape(load_demand_data, 2, []), 1);
% hourly_load_demand = reshape(hourly_load_demand, 24, [])'; 
% [~, rep_hourly_load_demand] = kmeans(hourly_load_demand,K);
% % plot([1:24]',rep_hourly_load_demand)
% 
% Set_Pgmax=cell(K,1);
% for k=1:K
%     Pgmax=repmat(gen(:,9),1,Nt);
%     Pgmax(Set_RE,:)=repmat(rep_hourly_wind_output(k,:),N_RE,1).*repmat(gen(Set_RE,9),1,Nt);
%     Set_Pgmax{k}=Pgmax;
% end
% Pgmax=Set_Pgmax{4};
% 
% Set_PL=cell(K,1);
% for k=1:K
%     PL=zeros(Nb,Nt);
%     for i=1:Nb
%         PL(i,:)=bus(i,3)*rep_hourly_load_demand(k,:);
%     end
%     Set_PL{k}=PL;
% end
% PL=Set_PL{1};
% 
% PG_all=sdpvar(Ng,Nt,'full');%发电机中标总容量
% PG=sdpvar(Ng,Nt,'full');%发电机中标容量
% RG_KKX=sdpvar(Ng,Nt,'full');
% Theta=sdpvar(Nb,Nt,'full');
% UG=binvar(Ng,Nt,'full');%机组状态变量
% ZG=binvar(Ng,Nt,'full');%开机变量
% WG=binvar(Ng,Nt,'full');%关机变量
% 
% Constraints_Clearing0=[];
% 
% %功率平衡约束
% Constraints_Clearing0=[Constraints_Clearing0,PG+RG_KKX==PG_all];
% 
% Constraints_Clearing0=[Constraints_Clearing0,0<=PG];
% Constraints_Clearing0=[Constraints_Clearing0,PG(Set_TH,:)<=Pgmax(Set_TH,:)];
% Constraints_Clearing0=[Constraints_Clearing0,PG(Set_RE,:)<=Pgmax(Set_RE,:)];%风机参与能量市场
% 
% Constraints_Clearing0=[Constraints_Clearing0,0<=RG_KKX];
% Constraints_Clearing0=[Constraints_Clearing0,RG_KKX(Set_TH,:)<=Pgmax(Set_TH,:)];
% Constraints_Clearing0=[Constraints_Clearing0,0==RG_KKX(Set_RE,:)];%风电不能当备用,不参与备用市场
% 
% %火电机组约束
% Constraints_Clearing0=[Constraints_Clearing0,Pgmin(Set_TH,:).*UG(Set_TH,:)<=PG_all(Set_TH,:)];
% Constraints_Clearing0=[Constraints_Clearing0,PG_all(Set_TH,:)<=Pgmax(Set_TH,:).*UG(Set_TH,:)];
% 
% %能量市场平衡约束
% Constraints_Clearing0=[Constraints_Clearing0,[sum(PL,1)'-sum(PG,1)'==0]];
% 
% %备用市场约束
% Constraints_Clearing0=[Constraints_Clearing0,sum(RG_KKX,1)'>=0.15*sum(PL,1)'];
% 
% Constraints_Clearing0=[Constraints_Clearing0,[UG(Set_RE,:)==1]]; %假设风机一直处于开机状态
% Constraints_Clearing0=[Constraints_Clearing0,[UG(Set_Fast,:)==0]]; %快速启停机组只供应急时使用
% Constraints_Clearing0=[Constraints_Clearing0,[UG([22,23],:)==0]];
% 
% %启停机逻辑
% for t=1:Nt
%     if t==1
%         Constraints_Clearing0=[Constraints_Clearing0,[UG(:,t)==ZG(:,t)-WG(:,t)]];
%     else
%         Constraints_Clearing0=[Constraints_Clearing0,[UG(:,t)-UG(:,t-1)==ZG(:,t)-WG(:,t)]];
%     end
% end
% 
% %发电机最小开停机时间约束
% for g=1:Ng
%     for t=2:Nt
%         if t+T_on_off(g)-1<=Nt
%             Constraints_Clearing0=[Constraints_Clearing0,-UG(g,t-1)+UG(g,t)-UG(g,t:t+T_on_off(g)-1)<=0,...
%                 UG(g,t-1)-UG(g,t)+UG(g,t:t+T_on_off(g)-1)<=1];
%         end
%     end
% end
% 
% %爬坡约束
% for t=1:Nt-1
%     Constraints_Clearing0=[Constraints_Clearing0,[PG(Set_TH,t+1)-PG(Set_TH,t)<=Ramp_max(Set_TH).*UG(Set_TH,t)+(1-UG(Set_TH,t)).*gen(Set_TH,9)]];
%     Constraints_Clearing0=[Constraints_Clearing0,[PG(Set_TH,t)-PG(Set_TH,t+1)<=Ramp_max(Set_TH).*UG(Set_TH,t+1)+(1-UG(Set_TH,t+1)).*gen(Set_TH,9)]];
% end
% 
% %爬坡约束
% Constraints_Clearing0=[Constraints_Clearing0,[RG_KKX<=repmat(Ramp_max,1,Nt)]];
% 
% num_phjd=find(bus(:,2)==3);
% for t=1:Nt
%     Constraints_Clearing0=[Constraints_Clearing0,[PL(:,t)-Ag*PG(:,t)+A*(diag(1./branch(:,4))*(A'*Theta(:,t)))==0]];
%     Constraints_Clearing0=[Constraints_Clearing0,[-pi<=Theta(:,t)<=pi,Theta(num_phjd,t)==0]];
%     Constraints_Clearing0=[Constraints_Clearing0,[-branch(:,6)<=diag(1./branch(:,4))*(A'*Theta(:,t))<=branch(:,6)]];
% end
% 
% ops_clearing=sdpsettings('solver', 'gurobi', 'verbose', 1);
% sol=solvesdp(Constraints_Clearing0,...
%     sum(sum(repmat(up_cost(Set_TH),1,Nt).*ZG(Set_TH,:),1))+sum(sum(repmat(shut_cost(Set_TH),1,Nt).*WG(Set_TH,:),1))...
%     +sum(sum(repmat(cost,1,Nt).*PG_all*baseMVA,1)),ops_clearing);
% 
% UG_da=value(UG); % 日前确定机组组合方案
% Pgmax_da=value(PG_all);
% 
% da_result=cell(2,1);
% da_result{1}=UG_da;
% da_result{2}=Pgmax_da;
% da_result{3}=PL;
% da_result{4}=Pgmax;
% 
% (sum(Pgmax_da,1)-sum(PL,1))./sum(PL,1)
% 
% save('illinois200_da_result_TL.mat','da_result')

load('illinois200_da_result_TL.mat')
UG_da=da_result{1};
Pgmax_da=da_result{2};
PL=da_result{3};
Pgmax=da_result{4};
(sum(Pgmax_da,1)-sum(PL,1))./sum(PL,1)

stairs(sum(PL,1)*100)
hold on
stairs(sum(Pgmax_da(Set_RE,:),1)*100)

num_down=setdiff(find(~any(UG_da,2)==1),Set_Fast');
idx=[];
for root=1:N_failure
    gen_failure_num=failure_set{1,1}{root,1};
    line_failure_num=failure_set{1,2}{root,1};
    if ~isempty(intersect(gen_failure_num,num_down))
        idx=[idx;root];
    end
end
prob_failure_up=prob_failure;
pg_state_up=pg_state;
prob_failure_up(idx)=[];
pg_state_up(idx,:)=[];

rate_min=1.15;
rate_max=1.25;

PLMAX = zeros(Nb, Ns);
PGMAX = zeros(Ng, Ns);
EDNS_vector_gs_matrix=zeros(Nt, Ns_each_t);
LOLP_vector_gs_matrix=zeros(Nt, Ns_each_t);
EDNS_vector_gs=zeros(Ns,1);
LOLP_vector_gs=zeros(Ns,1);
test = zeros(Ns,1);

s=0;
for t=1:Nt
    for s_each_t = 1:Ns_each_t
        s=s+1;
        PLMAX(:, s) = PL(:, t);
        PGMAX(Set_TH, s) = Pgmax_da(Set_TH, t);
        PGMAX(Set_RE, s) = Pgmax_da(Set_RE, t);
        x_da_t=UG_da(:,t);

        num_fast=Set_Fast(find(x(1:N_Fast, s)>0.5));
        x_da_t(num_fast)=1;

        PGMAX(Set_TH, s) = Pgmax_da(Set_TH, t) + x_da_t(Set_TH) .* (-Ramp_max(Set_TH) + 2 * Ramp_max(Set_TH) .* x(N_Fast+1:N_Fast+N_TH, s));
        PGMAX(Set_TH, s) = max(PGMAX(Set_TH, s),x_da_t(Set_TH).*Pgmin(Set_TH,t));
        PGMAX(Set_TH, s) = min(PGMAX(Set_TH, s),x_da_t(Set_TH).*Pgmax(Set_TH,t));

        rate = rate_min + x(end, s) * (rate_max - rate_min);

        N=0;
        while(1)
            N=N+1;
            if sum(PGMAX(:, s))/sum(PL(:, t))<rate
                delta = rate*sum(PL(:, t))-sum(PGMAX(:, s));
            elseif sum(PGMAX(:, s))/sum(PL(:, t))>rate
                delta = rate*sum(PL(:, t))-sum(PGMAX(:, s));
            else
                break
            end
            num = Set_TH(find(x_da_t(Set_TH)==1 & PGMAX(Set_TH, s)<=Pgmax(Set_TH, t) & PGMAX(Set_TH, s)>=Pgmin(Set_TH, t)));
            adjust = delta / length(num);
            PGMAX(num, s) = PGMAX(num, s)+adjust;
            PGMAX(num, s) = max(PGMAX(num, s),Pgmin(num, t));
            PGMAX(num, s) = min(PGMAX(num, s),Pgmax(num, t));
            if N>=100 || abs(sum(PGMAX(:, s))/sum(PL(:, t))-rate)<=0.001
                break
            end
        end

        test(s) = sum(PGMAX(:, s)) / sum(PL(:, t));

        PGMAX(Set_RE, s) = min(Pgmax(Set_RE, t) .* (1 - 0.30 + 0.60 * x(N_Fast+N_TH+1:N_Fast+Ng, s)),gen(Set_RE,9));
        PLMAX(:, s) = PL(:, t).* (1 - 0.10 + 0.20 * x(N_Fast+Ng+1:end-1, s));

        load_shedding_gs=sum(PLMAX(:, s))-pg_state*PGMAX(:, s);
        load_shedding_gs(load_shedding_gs<0)=0;
        EDNS_gs=load_shedding_gs'*prob_failure;
        LOLP_gs=(load_shedding_gs>0)'*prob_failure;

        EDNS_vector_gs_matrix(t,s_each_t)=EDNS_gs;
        LOLP_vector_gs_matrix(t,s_each_t)=LOLP_gs;
        EDNS_vector_gs(s)=EDNS_gs;
        LOLP_vector_gs(s)=LOLP_gs;
    end
end

EDNS_quantile_5 = quantile(EDNS_vector_gs_matrix, 0.05, 2);
LOLP_quantile_5 = quantile(LOLP_vector_gs_matrix, 0.05, 2);

EDNS_quantile_15 = quantile(EDNS_vector_gs_matrix, 0.15, 2);
LOLP_quantile_15 = quantile(LOLP_vector_gs_matrix, 0.15, 2);

EDNS_quantile_25 = quantile(EDNS_vector_gs_matrix, 0.25, 2);
LOLP_quantile_25 = quantile(LOLP_vector_gs_matrix, 0.25, 2);

num=find(test>1.15 & test<1.25);
EDNS_target = quantile(EDNS_vector_gs(num), 0.25);
LOLP_target = quantile(LOLP_vector_gs(num), 0.25);

% % SMOTE
x0=x';

% scatter([1:length(EDNS_vector_gs)],EDNS_vector_gs)
% scatter([1:length(LOLP_vector_gs)],LOLP_vector_gs)

EDNS_vector_gs_mean=mean(EDNS_vector_gs);
EDNS_vector_gs_std=std(EDNS_vector_gs);
LOLP_vector_gs_mean=mean(LOLP_vector_gs);
LOLP_vector_gs_std=std(LOLP_vector_gs);

data=[(EDNS_vector_gs-EDNS_vector_gs_mean)/EDNS_vector_gs_std, (LOLP_vector_gs-LOLP_vector_gs_mean)/LOLP_vector_gs_std];

K_smote=10;
[idx_smote,ctr_smote]=kmeans(data,K_smote);

N_threshold=round(Ns/K_smote);

while(1)
    cluster_counts = accumarray(idx_smote, 1)'
    num_minority=find(cluster_counts<N_threshold);
    N_num_minority=length(num_minority);

    if isempty(num_minority)
        break
    end

    x_add=[];
    for i=1:N_num_minority
        num_minority_sample=find(idx_smote==num_minority(i));
        N_num_minority_sample=length(num_minority_sample);

        x_others=x0;
        x_others(num_minority_sample,:)=[];

        K0=10;
        if N_num_minority_sample>K0
            C=x0(num_minority_sample(randperm(N_num_minority_sample,K0)),:);
            K1=K0;
        else
            C=x0(num_minority_sample,:);
            K1=N_num_minority_sample;
        end

        dis=pdist2(x_others,C,'cosine');

        for j=1:K1
            dis_j=dis(:,j);
            K2= 100;
            [~, min_indices] = mink(dis_j, K2);
            x_add=[x_add;C(j,:)+rand(K2,n_dim).*(x_others(min_indices,:)-C(j,:))];
        end
    end
    x0=[x0;x_add];

    x_tmp=x_add';
    N_add=size(x_tmp,2);

    PLMAX_add = zeros(Nb, Nt*N_add);
    PGMAX_add = zeros(Ng, Nt*N_add);
    test_add = zeros(N_add,1);
    for s=1:N_add
        t=randperm(Nt,1);

        PLMAX_add(:, s) = PL(:, t);
        PGMAX_add(Set_TH, s) = Pgmax_da(Set_TH, t);
        PGMAX_add(Set_RE, s) = Pgmax_da(Set_RE, t);
        x_da_t=UG_da(:,t);

        num_fast=Set_Fast(find(x_tmp(1:N_Fast, s)>0.5));
        x_da_t(num_fast)=1;

        PGMAX_add(Set_TH, s) = Pgmax_da(Set_TH, t) + x_da_t(Set_TH) .* (-Ramp_max(Set_TH) + 2 * Ramp_max(Set_TH) .* x_tmp(N_Fast+1:N_Fast+N_TH, s));
        PGMAX_add(Set_TH, s) = max(PGMAX_add(Set_TH, s),x_da_t(Set_TH).*Pgmin(Set_TH,t));
        PGMAX_add(Set_TH, s) = min(PGMAX_add(Set_TH, s),x_da_t(Set_TH).*Pgmax(Set_TH,t));

        rate = rate_min + x_tmp(end, s) * (rate_max - rate_min);

        N=0;
        while(1)
            N=N+1;
            if sum(PGMAX_add(:, s))/sum(PL(:, t))<rate
                delta = rate*sum(PL(:, t))-sum(PGMAX_add(:, s));
            elseif sum(PGMAX_add(:, s))/sum(PL(:, t))>rate
                delta = rate*sum(PL(:, t))-sum(PGMAX_add(:, s));
            else
                break
            end
            num = Set_TH(find(x_da_t(Set_TH)==1 & PGMAX_add(Set_TH, s)<=Pgmax(Set_TH, t) & PGMAX_add(Set_TH, s)>=Pgmin(Set_TH, t)));
            adjust = delta / length(num);
            PGMAX_add(num, s) = PGMAX_add(num, s)+adjust;
            PGMAX_add(num, s) = max(PGMAX_add(num, s),Pgmin(num, t));
            PGMAX_add(num, s) = min(PGMAX_add(num, s),Pgmax(num, t));
            if N>=100 || abs(sum(PGMAX_add(:, s))/sum(PL(:, t))-rate)<=0.001
                break
            end
        end

        test_add(s) = sum(PGMAX_add(:, s)) / sum(PL(:, t));

        PGMAX_add(Set_RE, s) = min(Pgmax(Set_RE, t) .* (1 - 0.30 + 0.60 * x_tmp(N_Fast+N_TH+1:N_Fast+Ng, s)),gen(Set_RE,9));
        PLMAX_add(:, s) = PL(:, t).* (1 - 0.10 + 0.20 * x_tmp(N_Fast+Ng+1:end-1, s));

        load_shedding_gs=sum(PLMAX_add(:, s))-pg_state*PGMAX_add(:, s);
        load_shedding_gs(load_shedding_gs<0)=0;
        EDNS_gs=load_shedding_gs'*prob_failure;
        LOLP_gs=(load_shedding_gs>0)'*prob_failure;

        dis=pdist2([(EDNS_gs-EDNS_vector_gs_mean)/EDNS_vector_gs_std,(LOLP_gs-LOLP_vector_gs_mean)/LOLP_vector_gs_std],ctr_smote);
        [~,idx_min]=min(dis);
        if ismember(idx_min,num_minority)
            idx_smote=[idx_smote;idx_min];
            PLMAX=[PLMAX,PLMAX_add(:,s)];
            PGMAX=[PGMAX,PGMAX_add(:,s)];
            EDNS_vector_gs=[EDNS_vector_gs;EDNS_gs];
            LOLP_vector_gs=[LOLP_vector_gs;LOLP_gs];
        end
    end
end

% 聚类
N_need=200;
k_clusters=ceil(N_need/K_smote);

rep_idx=[];
for k=1:K_smote
    k
    %% 1. 数据准备与标准化
    num=find(idx_smote==k);

    data=[PGMAX(:,num);PLMAX(:,num)]';
    data_normalized = zscore(data);

    %% 2. PCA降维
    cov_matrix = cov(data_normalized);
    [V, D] = eig(cov_matrix);
    eigenvalues = diag(D);
    [sorted_eval, idx] = sort(eigenvalues, 'descend');
    sorted_evec = V(:, idx);

    % 选择主成分数量
    cumulative_variance = cumsum(sorted_eval) / sum(sorted_eval);
    k_pca = find(cumulative_variance >= 0.95, 1); % 保留95%方差的主成分
    selected_evec = sorted_evec(:, 1:k_pca);
    reduced_data = data_normalized * selected_evec; % 投影到主成分空间

    %% 3. K-Medoids聚类
    [idx,C] = kmeans(reduced_data, k_clusters,'Distance', 'cityblock');
    D = pdist2(reduced_data, C, 'cityblock');
    closest_points_idx = zeros(k_clusters, 1);
    for w = 1:k_clusters
        % 找到属于第i簇的所有点
        cluster_points = find(idx == w);
        if ~isempty(cluster_points)
            [~, min_idx] = min(D(cluster_points, w));
            closest_points_idx(w) = cluster_points(min_idx(1));
        end
    end
    rep_idx=[rep_idx;num(closest_points_idx)];
end

Ns=length(rep_idx);
PGMAX_train=PGMAX(:,rep_idx);
PLMAX_train=PLMAX(:,rep_idx);

df_LOLP_handles = cell(1, N);
for root=1:N_failure
    root
    load_shedding_matrix_gs=sum(PLMAX,1)-pg_state(root,:)*PGMAX;
    load_shedding_matrix_gs(load_shedding_matrix_gs<0)=0;
    load_shedding_mean = mean(load_shedding_matrix_gs);  % 失负荷量均值
    sigma = std(load_shedding_matrix_gs);                % 失负荷量标准差
    
    x0 = load_shedding_mean + 3 * sigma;
    func = @(k)(1-tanh(k*(x0-load_shedding_mean)).^2)*k-0.001; % 防止梯度爆炸
    k = fzero(func, 1);
    
    % f_LOLP = @(x) tanh(k * (x - load_shedding_mean))+0.5;
    df_LOLP_handles{root} = @(x) (1-tanh(k * (x - load_shedding_mean)).^2)*k;
end

xita=1e-8;

EDNS_vector=zeros(Ns,1);
LOLP_vector=zeros(Ns,1);
partial_derivative_matrix_EDNS=zeros(Ns,Ng+Nb);
partial_derivative_matrix_LOLP=zeros(Ns,Ng+Nb);

load_shedding_events_num=zeros(N_failure,1);

N_parpool=80;%调用的处理器个数
local = parcluster('local');
local.NumWorkers = N_parpool;
pool = local.parpool(N_parpool);
N_batch=N_parpool;

tic
for batch=1:ceil(N_failure/N_batch)
    [batch,ceil(N_failure/N_batch)]
    if batch*N_batch<=N_failure
        failure_num=[(batch-1)*N_batch+1:batch*N_batch];
    else
        failure_num=[(batch-1)*N_batch+1:N_failure];
    end
    failure_set_batch{1,1}=failure_set{1,1}(failure_num);
    failure_set_batch{1,2}=failure_set{1,2}(failure_num);
    prob_failure_batch=prob_failure(failure_num);
    df_LOLP_handles_batch=df_LOLP_handles(failure_num);
    
    load_shedding_matrix=zeros(length(failure_num),Ns);
    Set_LM_f=cell(length(failure_num),1);

    parfor root=1:length(failure_num)
        failure_set_local=failure_set_batch;
        gen_failure_num=failure_set_local{1,1}{root,1};
        line_failure_num=failure_set_local{1,2}{root,1};
        loadshedding_f=zeros(Ns,1);
        LM_f=zeros(Ns,Ng+Nb);

        Nmax=100;
        Set_B_inv=cell(1,Nmax);
        Set_LM=cell(1,Nmax);
        N_LM=0;
        for s=1:Ns
            PLMAX_local=PLMAX_train;
            PGMAX_local=PGMAX_train;

            state_map_A_local=state_map_A;
            Set_A_local=Set_A;

            tmp=state_map_A_local(mat2str(line_failure_num));
            A=Set_A_local{tmp,1};

            PLmax=PLMAX_local(:,s);
            PGmax=PGMAX_local(:,s);
            tebranch = branch;
            if ~isempty(line_failure_num)
                tebranch(line_failure_num,RATE)=0;
            end
            Fmax=tebranch(:,RATE);
            PGmax_tmp=PGmax;
            PGmax_tmp(gen_failure_num)=0;
            b=[PLmax;Fmax;Fmax;PGmax_tmp;PLmax];%约束的上下限，因为都转化为了等式约束,上下限一样
            param=struct();
            param.MSK_IPAR_OPTIMIZER = 'MSK_OPTIMIZER_FREE_SIMPLEX';
            flag_LMSE=0;
            if N_LM>=1
                for k=1:N_LM
                    num=N_LM-k+1;
                    if Set_B_inv{num}*b>-xita
                        LM=Set_LM{num};
                        loadshedding_f(s)=LM*b;
                        tmp=[LM(Nb+2*Nl+1:Nb+2*Nl+Ng),LM(1:Nb)+LM(Nb+2*Nl+Ng+1:end)];
                        tmp(gen_failure_num)=0;
                        LM_f(s,:)=tmp;
                        flag_LMSE=1;
                        break
                    end
                end
            end
            if flag_LMSE==0 || s==1
                c=[zeros(1,Nb+Ng),ones(1,Nb),zeros(1,2*Nl+Ng+Nb)]';%切负荷系数
                prob=struct();
                prob.a=A;
                prob.c=c';
                prob.blc=b;
                prob.buc=b;
                prob.blx=zeros(size(A,2),1);
                prob.bux=[];
                [r,res] = mosekopt('minimize info  echo(0)',prob,param);
                index_AB=find(res.sol.bas.skx(:,1)=='B');
                index_AN = setdiff(1:size(A,2),index_AB);
                cB = c(index_AB,1);
                cN = c(index_AN,1);
                AB = A(:,index_AB);
                B_inv=AB\eye(size(A,1));
                LM=cB'*B_inv;
                loadshedding_f(s)=res.sol.bas.pobjval;
                tmp=[LM(Nb+2*Nl+1:Nb+2*Nl+Ng),LM(1:Nb)+LM(Nb+2*Nl+Ng+1:end)];
                tmp(gen_failure_num)=0;
                LM_f(s,:)=tmp;
                if N_LM+1<=Nmax
                    N_LM=N_LM+1;
                    Set_B_inv{N_LM}=sparse(B_inv);
                    Set_LM{N_LM}=LM;
                else
                    Set_B_inv(1)=[];
                    Set_LM(1)=[];
                    Set_B_inv{end+1}=sparse(B_inv);
                    Set_LM{end+1}=LM;
                end
            end
        end
        load_shedding_matrix(root,:)=loadshedding_f';
        Set_LM_f{root}=sparse(LM_f);
    end
    flag=double(load_shedding_matrix>0);
    load_shedding_events_num(failure_num,:)=load_shedding_events_num(failure_num,:)+sum(flag,2);
    EDNS_vector=EDNS_vector+load_shedding_matrix'*prob_failure_batch;
    LOLP_vector=LOLP_vector+flag'*prob_failure_batch;
    for k=1:length(failure_num)
        partial_derivative_matrix_EDNS=partial_derivative_matrix_EDNS+Set_LM_f{k}*prob_failure_batch(k);
        partial_derivative_matrix_LOLP=partial_derivative_matrix_LOLP+...
            repmat(df_LOLP_handles_batch{k}(load_shedding_matrix(k,:))',1,Ng+Nb).*Set_LM_f{k}*prob_failure_batch(k);
    end
end

% load_shedding_matrix_gs=repmat(sum(PLMAX_train,1),N_failure,1)-pg_state*PGMAX_train;
% load_shedding_matrix_gs(load_shedding_matrix_gs<0)=0;
% EDNS_vector_gs=load_shedding_matrix_gs'*prob_failure;
% LOLP_vector_gs=(load_shedding_matrix_gs>0)'*prob_failure;

dataX = [PGMAX_train;PLMAX_train]';
EDNS_vector_std=std(EDNS_vector);
EDNS_vector_mean=mean(EDNS_vector);

LOLP_vector_std=std(LOLP_vector);
LOLP_vector_mean=mean(LOLP_vector);

dataY = (EDNS_vector-EDNS_vector_mean)/EDNS_vector_std;
dataYx0 = partial_derivative_matrix_EDNS/EDNS_vector_std;

dataZ = (LOLP_vector-LOLP_vector_mean)/LOLP_vector_std;
dataZx0 = partial_derivative_matrix_LOLP/LOLP_vector_std;

sampling_time=toc

delete(gcp('nocreate'));

clear PGMAX
clear PLMAX
clear EDNS_vector_gs
clear LOLP_vector_gs
clear EDNS_vector_gs_matrix
clear LOLP_vector_gs_matrix

save('ED_illinois200_data_for_testing_PINN_200_TL.mat')


