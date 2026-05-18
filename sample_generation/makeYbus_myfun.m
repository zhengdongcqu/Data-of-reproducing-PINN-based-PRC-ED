function [Ybus, Yf, Yt,dYf_dAngle,dYt_dAngle,dYf_dTCSC,dYt_dTCSC] = makeYbus(baseMVA, bus, branch)
%MAKEYBUS   Builds the bus admittance matrix and branch admittance matrices.
%   [Ybus, Yf, Yt] = makeYbus(baseMVA, bus, branch) returns the full
%   bus admittance matrix (i.e. for all buses) and the matrices Yf and Yt
%   which, when multiplied by a complex voltage vector, yield the vector
%   currents injected into each line from the "from" and "to" buses
%   respectively of each line. Does appropriate conversions to p.u.

%   MATPOWER Version 2.0
%   by Ray Zimmerman, PSERC Cornell    12/19/97
%   Copyright (c) 1996, 1997 by Power System Engineering Research Center (PSERC)
%   See http://www.pserc.cornell.edu/ for more info.

%% constants
j = sqrt(-1);
nb = size(bus, 1);			%% number of buses
nl = size(branch, 1);		%% number of lines

%% define named indices into bus, branch matrices
[PQ, PV, REF, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, ...
	VA, BASE_KV, VMAX, VMIN, PGMAX,PGMIN,QGMAX,QGMIN,PG,QG,LOLP,LOLE,LOLF,LOLD,EDNS,EENS,...
    LAM_P,LAM_Q,MU_VMAX,MU_VMIN,MU_PMAX,MU_PMIN,MU_QMAX,MU_QMIN] = idx_bus;
[GEN_BUS, GEN_PMAX, GEN_PMIN, GEN_QMAX, GEN_QMIN, GEN_STATUS, ...
	GEN_LAMDA,GEN_MTTF,GEN_PG,GEN_QG,GEN_PFAILURE] = idx_gen;
[F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE, TAP, SHIFT_ANGLE, BR_STATUS, BR_LAMDA,...
        BR_MTTF,TCSC_X, BR_PFAILURE, PF, QF, PT, QT, MU_SF, MU_ST, KT_MAXANGLE_TPST,KT_MINANGLE_TPST,KT_MAX_X_TCSC,KT_MIN_X_TCSC] = idx_brch;


%% check that bus numbers are equal to indices to bus (one set of bus numbers)
if any(bus(:, BUS_I) ~= [1:nb]')
	error('buses must appear in order by bus number')
end

%% for each branch, compute the elements of the branch admittance matrix where
%%
%%		| If |   | Yff  Yft |   | Vf |;
%%		|    | = |          | * |    |;
%%		| It |   | Ytf  Ytt |   | Vt |;
%%
%%stat = branch(:, BR_STATUS);					%% ones at in-service branches
%%Ys = stat ./ (branch(:, BR_R) + j *( branch(:, BR_X)-branch(:,TCSC_X)));%% series admittance
Ys = 1.0./ (branch(:, BR_R) + j *( branch(:, BR_X)-branch(:,TCSC_X)));%% series admittance
%% TCSC_X is a positive value for capacitive chracteristic and negative value for reactive characteristic
%%Bc = stat .* branch(:, BR_B);							%% line charging susceptance
Bc = branch(:, BR_B);							%% line charging susceptance
tap = ones(nl, 1);								%% default tap ratio = 1
i = find(branch(:, TAP));						%% indices of non-zero tap ratios
tap(i) = branch(i, TAP);						%% assign non-zero tap ratios
tap = tap .* exp(j* branch(:, SHIFT_ANGLE));	%% add phase shifters
Ytt = Ys + j*Bc/2;
Yff = Ytt ./ (tap .* conj(tap));
Yft = - Ys ./ conj(tap);
Ytf = - Ys ./ tap;

%% for each branch, compute the elements of partial of the branch admittance matrix w.r.t. shifting angle
%%
%%		| dIf_dAngle |   | dYff_dAngle    Yft_dAngle |   | Vf |;
%%		|            | = |                           | * |    |;
%%		| dIt_dAngle |   | Ytf_dAngle     Ytt_dAngle |   | Vt |;
%%      Note:tap = abs(tap).*exp(j*Angle);

dYff_dAngle = zeros(size(branch,1),1);
dYtt_dAngle = zeros(size(branch,1),1);
dYft_dAngle = - (Ys.*j) ./ conj(tap);
dYtf_dAngle = (Ys.*j)./tap;




dYs_dTCSC = -2.*real(Ys).*imag(Ys)+...
             j.*( -2.*power(imag(Ys),2)+1.0./(  power(branch(:, BR_R),2)+power(branch(:, BR_X)-branch(:,TCSC_X),2) ) ); 
dYff_dTCSC = dYs_dTCSC./(tap .* conj(tap));
dYtt_dTCSC = dYs_dTCSC;
dYft_dTCSC = -dYs_dTCSC./conj(tap);
dYtf_dTCSC = -dYs_dTCSC./tap;

%% compute shunt admittance
%% if Ps is the real power consumed by the shunt at V = 1.0 p.u.
%% and Qs is the reactive power injected by the shunt at V = 1.0 p.u.
%% then Ps - j Qs = V * conj(Ys * V) = conj(Ys) = Gs - jBs,
%% i.e. Ys = Ps + j Qs, so ...
Ys = (bus(:, GS) + j * bus(:, BS)) / baseMVA;	%% vector of shunt admittances

%% build Ybus
f = branch(:, F_BUS);							%% list of "from" buses
t = branch(:, T_BUS);							%% list of "to" buses
Cf = sparse(f, 1:nl, ones(nl, 1), nb, nl);		%% connection matrix for line & from buses
Ct = sparse(t, 1:nl, ones(nl, 1), nb, nl);		%% connection matrix for line & to buses
Ybus = spdiags(Ys, 0, nb, nb) + ...				%% shunt admittance
	Cf * spdiags(Yff, 0, nl, nl) * Cf' + ...	%% Yff term of branch admittance
	Cf * spdiags(Yft, 0, nl, nl) * Ct' + ...	%% Yft term of branch admittance
	Ct * spdiags(Ytf, 0, nl, nl) * Cf' + ...	%% Ytf term of branch admittance
	Ct * spdiags(Ytt, 0, nl, nl) * Ct';			%% Ytt term of branch admittance

%% Build Yf and Yt such that Yf * V is the vector of complex branch currents injected
%% at each branch's "from" bus, and Yt is the same for the "to" bus end
if nargout > 1
	i = [[1:nl]'; [1:nl]'];		%% double set of row indices	
	Yf = sparse(i, [f; t], [Yff; Yft]);
	Yt = sparse(i, [f; t], [Ytf; Ytt]);
    if nargout > 3
        dYf_dAngle = sparse(i, [f; t], [dYff_dAngle; dYft_dAngle]);
        dYt_dAngle = sparse(i, [f; t], [dYtf_dAngle; dYtt_dAngle]);
    end
    if nargout > 5
       dYf_dTCSC = sparse(i, [f; t], [dYff_dTCSC; dYft_dTCSC]);
       dYt_dTCSC = sparse(i, [f; t], [dYtf_dTCSC; dYtt_dTCSC]);
        
    end    
end

return;
