%% ========================================================================
%  comparison.m
%  从 test.m 提取的“核心优化流程”版本：
%  - 保留：LFM 构造、勒让德窗优化（FA + 精修）、核心指标计算
%  - 去掉：敏感度分析、绘图、与其他窗函数比较输出
% =========================================================================

clear; clc;
rng(80);

%% 信号参数
B = 600e6;
T = 1.7e-6;
fs = 720e6;

N = round(T * fs);
t = (-N/2:N/2-1)' / fs;
f = (-N/2:N/2-1)' * (fs/N);
k = B / T;
s_LFM = exp(1j * pi * k * t.^2);

% 仅用于优化目标基线（不做对比展示）
idx_band_ref = abs(f) <= B/2;
N_band_ref = sum(idx_band_ref);
W_hamming_centered_ref = zeros(N, 1);
W_hamming_centered_ref(idx_band_ref) = hamming(N_band_ref);
W_hamming_centered_ref = W_hamming_centered_ref / (max(W_hamming_centered_ref) + eps);
W_hamming_ref = ifftshift(W_hamming_centered_ref);
s_hamming_ref = ifft(fft(s_LFM) .* W_hamming_ref);
[R_hamming_ref, lag_hamming_ref] = xcorr(s_hamming_ref);
R_hamming_ref = safe_normalize(abs(R_hamming_ref));
[PSLR_hamming_ref, MW_hamming_ref, PAPR_hamming_ref] = compute_metrics_single(R_hamming_ref, lag_hamming_ref, s_hamming_ref);

%% 萤火虫算法参数
dim = 5;
nFireflies = 30;
maxIter = 150;
gamma = 1;
beta0 = 1;
alpha = 0.2;
lambda_MW = 7;
lambda_PAPR = 8;
lambda_PSLR = 80;
MW_target = 1.0;
PAPR_target = 1.0;
PSLR_margin = 1.2;
PSLR_target = PSLR_hamming_ref - PSLR_margin;
PSLR_floor = -30;

lb = -5 * ones(1, dim);
ub = 5 * ones(1, dim);

nRestarts = 4;
global_best_fitness = inf;
global_best_b = zeros(1, dim);

for restart = 1:nRestarts
    fireflies = lb + (ub - lb) .* rand(nFireflies, dim);
    fitness = zeros(nFireflies, 1);

    for i = 1:nFireflies
        fitness(i) = fitness_func(fireflies(i,:), s_LFM, fs, B, ...
            lambda_MW, lambda_PAPR, lambda_PSLR, MW_target, PAPR_target, PSLR_target, PSLR_floor);
    end

    for iter = 1:maxIter
        [fitness, idx] = sort(fitness);
        fireflies = fireflies(idx, :);

        for i = 1:nFireflies
            for j = 1:nFireflies
                if fitness(j) < fitness(i)
                    r = norm(fireflies(i,:) - fireflies(j,:));
                    beta = beta0 * exp(-gamma * r^2);
                    fireflies(i,:) = fireflies(i,:) + ...
                        beta * (fireflies(j,:) - fireflies(i,:)) + ...
                        alpha * (rand(1,dim) - 0.5) .* (ub - lb);
                    fireflies(i,:) = max(min(fireflies(i,:), ub), lb);
                    fitness(i) = fitness_func(fireflies(i,:), s_LFM, fs, B, ...
                        lambda_MW, lambda_PAPR, lambda_PSLR, MW_target, PAPR_target, PSLR_target, PSLR_floor);
                end
            end
        end
    end

    [fitness, idx] = sort(fitness);
    fireflies = fireflies(idx, :);
    if fitness(1) < global_best_fitness
        global_best_fitness = fitness(1);
        global_best_b = fireflies(1,:);
    end
end

b_opt = global_best_b;

% 精修前指标
PSLR_before = compute_PSLR(b_opt, s_LFM, fs, B);

[R_lfm_ref, lag_lfm_ref] = xcorr(s_LFM);
R_lfm_ref = safe_normalize(abs(R_lfm_ref));
[~, MW_lfm, PAPR_lfm] = compute_metrics_single(R_lfm_ref, lag_lfm_ref, s_LFM);

MW_factor_hamming = MW_hamming_ref / MW_lfm;
PAPR_factor_hamming = PAPR_hamming_ref / PAPR_lfm;

delta_MW = 0.2;
delta_PAPR = 0.2;
MW_target_refined = MW_factor_hamming + delta_MW;
PAPR_target_refined = PAPR_factor_hamming + delta_PAPR;

obj_fun = @(b) compute_PSLR(b, s_LFM, fs, B);
options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'interior-point', ...
    'OptimalityTolerance', 1e-4, 'ConstraintTolerance', 1e-4, ...
    'StepTolerance', 1e-8, 'MaxFunctionEvaluations', 2000);

base_margin = PSLR_margin;
max_attempts = 5;
margin_step = 0.2;
accepted = false;

b_refined = b_opt;
fval_refined = PSLR_before;

for attempt = 1:max_attempts
    used_margin = max(0, base_margin - (attempt-1) * margin_step);
    PSLR_target_refined = PSLR_hamming_ref - used_margin;

    nonlcon_stageA = @(b) compute_constraints_v2(b, s_LFM, fs, B, MW_target_refined, PAPR_target_refined, 1e6, MW_lfm, PAPR_lfm);
    [b_feasible, ~, exitflag_A, ~] = fmincon(obj_fun, b_opt, [], [], [], [], lb, ub, nonlcon_stageA, options);
    [cA_try, ~] = nonlcon_stageA(b_feasible);

    nonlcon = @(b) compute_constraints_v2(b, s_LFM, fs, B, MW_target_refined, PAPR_target_refined, PSLR_target_refined, MW_lfm, PAPR_lfm);
    [b_try, fval_try, exitflag_try, ~] = fmincon(obj_fun, b_feasible, [], [], [], [], lb, ub, nonlcon, options);
    [c_try, ~] = nonlcon(b_try);

    if exitflag_try > 0 && all(c_try <= 1e-6) && (fval_try <= PSLR_before)
        accepted = true;
        b_refined = b_try;
        fval_refined = fval_try;
        break;
    elseif exitflag_A > 0 && all(cA_try(1:2) <= 1e-6) && ~accepted
        [pslr_A, ~, ~] = evaluate_metrics(b_feasible, s_LFM, fs, B);
        if pslr_A <= PSLR_before - 0.2
            accepted = true;
            b_refined = b_feasible;
            fval_refined = pslr_A;
            break;
        end
    end
end

if ~accepted
    b_refined = b_opt;
    fval_refined = PSLR_before;
end

b_opt = b_refined;

% 最终仅输出我方优化结果（无对比、无绘图）
[W_opt, ~] = legendre_window(b_opt, fs, B, N);
s_w_opt = ifft(fft(s_LFM) .* W_opt);
[R_opt, lag_opt] = xcorr(s_w_opt);
R_opt = safe_normalize(abs(R_opt));
[PSLR_opt, MW_opt, PAPR_opt] = compute_metrics_single(R_opt, lag_opt, s_w_opt);

fprintf('\n=================== Optimized Legendre Window Result ===================\n');
fprintf('PSLR = %.2f dB\n', PSLR_opt);
fprintf('MW   = %.2e\n', MW_opt);
fprintf('PAPR = %.2f\n', PAPR_opt);
fprintf('Final objective (PSLR) = %.2f dB\n', fval_refined);
fprintf('=======================================================================\n');


%% ========================================================================
%  Algorithm 2 (Wang 2023 pulse-compression window) + 对比
%% ========================================================================
params_alg2 = struct();
params_alg2.rho = 0.01;
params_alg2.Tmax = 800;
params_alg2.psi = 1e-4;
params_alg2.tau = 1.9;
params_alg2.seed = 80;
params_alg2.solver_preference = 'proj-grad';
params_alg2.verbose_every = 0;

% tau 自动扫（粗扫）
tau_list = [1.5, 1.9, 2.3];
coarse_tmax = 300;

alg2_stats = struct('tau', cell(numel(tau_list),1), 'pslr', [], 'mw', [], 'papr', [], 'w', []);
for i_tau = 1:numel(tau_list)
    params_tmp = params_alg2;
    params_tmp.tau = tau_list(i_tau);
    params_tmp.Tmax = coarse_tmax;
    [w_tmp, info_tmp] = design_window_alg2_wang2023(s_LFM, params_tmp);

    % Alg2 输出 w_tmp 为时域复权向量，直接作用于回波/匹配滤波模板
    s_tmp = w_tmp .* s_LFM;
    W_tmp_freq = fft(w_tmp);
    W_tmp_freq = W_tmp_freq / (max(abs(W_tmp_freq)) + eps);
    [R_tmp, lag_tmp] = xcorr(s_tmp);
    R_tmp = safe_normalize(abs(R_tmp));
    [pslr_tmp, mw_tmp, papr_tmp] = compute_metrics_single(R_tmp, lag_tmp, s_tmp);

    alg2_stats(i_tau).tau = tau_list(i_tau);
    alg2_stats(i_tau).pslr = pslr_tmp;
    alg2_stats(i_tau).mw = mw_tmp;
    alg2_stats(i_tau).papr = papr_tmp;
    alg2_stats(i_tau).w = w_tmp;
    alg2_stats(i_tau).info = info_tmp;
end

% 选择tau：优先PSLR最优且 MW/PAPR 相比 W_opt 不超过 +10%
idx_ok = [];
for i_tau = 1:numel(tau_list)
    if alg2_stats(i_tau).mw <= 1.10*MW_opt && alg2_stats(i_tau).papr <= 1.10*PAPR_opt
        idx_ok(end+1) = i_tau; %#ok<AGROW>
    end
end
if ~isempty(idx_ok)
    [~, rel_idx] = min([alg2_stats(idx_ok).pslr]);
    best_idx = idx_ok(rel_idx);
    tau_note = 'constraint-aware selection';
else
    [~, best_idx] = min([alg2_stats.pslr]);
    tau_note = 'PSLR-only fallback';
end

params_alg2.tau = alg2_stats(best_idx).tau;
[w_alg2, info_alg2] = design_window_alg2_wang2023(s_LFM, params_alg2);

% Alg2 输出 w_alg2 为时域复权向量，直接施加到 LFM
s_alg2 = w_alg2 .* s_LFM;
W_alg2_freq = fft(w_alg2);
W_alg2_freq = W_alg2_freq / (max(abs(W_alg2_freq)) + eps);
[R_alg2, lag_alg2] = xcorr(s_alg2);
R_alg2 = safe_normalize(abs(R_alg2));
[PSLR_alg2, MW_alg2, PAPR_alg2] = compute_metrics_single(R_alg2, lag_alg2, s_alg2);

fprintf('\n=================== Legendre vs Wang-Alg2 vs Hamming ===================\n');
fprintf('Selected tau for Alg2 = %.2f (%s)\n', params_alg2.tau, tau_note);
fprintf('Method\t\t\tPSLR(dB)\tMW\t\tPAPR\n');
fprintf('Legendre (W_opt)\t%.2f\t\t%.2e\t%.2f\n', PSLR_opt, MW_opt, PAPR_opt);
fprintf('Wang Alg2\t\t%.2f\t\t%.2e\t%.2f\n', PSLR_alg2, MW_alg2, PAPR_alg2);
fprintf('Hamming baseline\t%.2f\t\t%.2e\t%.2f\n', PSLR_hamming_ref, MW_hamming_ref, PAPR_hamming_ref);
fprintf('Alg2 iterations = %d, final residual = %.3e\n', info_alg2.iters, info_alg2.history.residual(end));
fprintf('Alg2 w-update solver = %s\n', info_alg2.solver_name);
fprintf('=======================================================================\n');

%% ========================================================================
%  算法对比图：窗函数频谱图 / 自相关函数图 / 主瓣范围自相关图
%% ========================================================================
W_legendre_center = fftshift(W_opt);
W_alg2_center = fftshift(W_alg2_freq);
W_hamming_center = fftshift(W_hamming_ref);

% 图1：窗函数频谱图（dB）
figure(1);
f_MHz = f / 1e6;
plot(f_MHz, 20*log10(abs(W_legendre_center)+eps), 'g-', 'LineWidth', 1.6); hold on;
plot(f_MHz, 20*log10(abs(W_alg2_center)+eps), 'b-.', 'LineWidth', 1.4);
plot(f_MHz, 20*log10(abs(W_hamming_center)+eps), 'r--', 'LineWidth', 1.3);
xlabel('Frequency (MHz)'); ylabel('Magnitude (dB)');
legend('Legendre (W_{opt})', 'Wang Alg2', 'Hamming', 'Location', 'best');
grid on;

S_lfm = abs(fftshift(fft(s_LFM)));
[~, idx_pk_spec] = max(S_lfm);
th_spec = max(S_lfm) * 10^(-3/20);
left_spec = find(S_lfm(1:idx_pk_spec) < th_spec, 1, 'last');
if isempty(left_spec), left_spec = 1; end
right_rel_spec = find(S_lfm(idx_pk_spec:end) < th_spec, 1, 'first');
if isempty(right_rel_spec), right_spec = length(S_lfm); else, right_spec = idx_pk_spec + right_rel_spec - 1; end
xlim([f(left_spec), f(right_spec)] / 1e6);
ylim([-80, 5]);

% 图2：自相关函数图（dB）
[R_lfm, lag] = xcorr(s_LFM);
R_lfm = safe_normalize(abs(R_lfm));

figure(2);
plot(lag/fs*1e6, 20*log10(R_lfm+eps), 'k-', 'LineWidth', 1.2); hold on;
plot(lag_opt/fs*1e6, 20*log10(R_opt+eps), 'g-', 'LineWidth', 1.6);
plot(lag_alg2/fs*1e6, 20*log10(R_alg2+eps), 'b-.', 'LineWidth', 1.4);
plot(lag_hamming_ref/fs*1e6, 20*log10(R_hamming_ref+eps), 'r--', 'LineWidth', 1.3);
xlabel('Time (us)'); ylabel('Normalized Magnitude (dB)');
legend('Original LFM', 'Legendre (W_{opt})', 'Wang Alg2', 'Hamming', 'Location', 'best');
grid on; xlim([-0.5 0.5]); ylim([-80 5]);

% 图3：主瓣范围内自相关函数图（线性幅度）
[~, idx_peak] = max(R_lfm);
half_width = 60;
range_idx = max(1, idx_peak-half_width) : min(length(R_lfm), idx_peak+half_width);
t_corr = lag / fs * 1e6;

figure(3);
plot(t_corr(range_idx), R_lfm(range_idx), 'k-', 'LineWidth', 1.2); hold on;
plot(t_corr(range_idx), R_opt(range_idx), 'g-', 'LineWidth', 1.6);
plot(t_corr(range_idx), R_alg2(range_idx), 'b-.', 'LineWidth', 1.4);
plot(t_corr(range_idx), R_hamming_ref(range_idx), 'r--', 'LineWidth', 1.3);
xlabel('Delay (us)'); ylabel('Normalized Magnitude');
legend('Original LFM', 'Legendre (W_{opt})', 'Wang Alg2', 'Hamming', 'Location', 'best');
grid on;

%% ========================================================================
%  函数定义
%% ========================================================================

function [W, f] = legendre_window(b, fs, B, N)
    f = (-N/2:N/2-1)' * (fs/N);
    f_norm = 2 * f / B;
    idx = abs(f) <= B/2;

    W_centered = zeros(N,1);
    if any(idx)
        f_seg = f_norm(idx);
        W_seg_raw = zeros(size(f_seg));
        for m = 1:length(b)
            n = 2*(m-1);
            P = legendreP(n, f_seg);
            W_seg_raw = W_seg_raw + b(m) * P;
        end
        W_seg = exp(W_seg_raw - max(W_seg_raw));
        W_centered(idx) = W_seg;
    end

    W = ifftshift(W_centered);
    W = W / (max(W) + eps);
end

function y = safe_normalize(x)
    x = x(:);
    m = max(abs(x));
    if ~isfinite(m) || m <= eps
        y = zeros(size(x));
    else
        y = x / m;
    end
end

function P = legendreP(n, x)
    if n == 0
        P = ones(size(x));
    elseif n == 1
        P = x;
    else
        P_prev2 = ones(size(x));
        P_prev1 = x;
        for k = 2:n
            P_curr = ((2*k-1)*x.*P_prev1 - (k-1)*P_prev2) / k;
            P_prev2 = P_prev1;
            P_prev1 = P_curr;
        end
        P = P_curr;
    end
end

function [PSLR, MW, PAPR] = compute_metrics_single(R, lag, s)
    R = R(:);
    lag = lag(:);
    [~, idx_peak] = max(R);

    th_3dB = 10^(-3/20);
    left_3dB = find(R(1:idx_peak) < th_3dB, 1, 'last');
    if isempty(left_3dB)
        left_3dB = 1;
    end
    right_3dB_rel = find(R(idx_peak:end) < th_3dB, 1, 'first');
    if isempty(right_3dB_rel)
        right_3dB = length(R);
    else
        right_3dB = idx_peak + right_3dB_rel - 1;
    end
    MW = lag(right_3dB) - lag(left_3dB);

    left_null = idx_peak;
    for i = idx_peak:-1:2
        if R(i-1) > R(i) && R(i) < R(i+1)
            left_null = i;
            break;
        end
    end

    right_null = idx_peak;
    for i = idx_peak:length(R)-2
        if R(i) > R(i+1) && R(i+1) < R(i+2)
            right_null = i+1;
            break;
        end
    end

    th_20dB = 10^(-20/20);
    if left_null == idx_peak
        left_20 = find(R(1:idx_peak) < th_20dB, 1, 'last');
        if ~isempty(left_20)
            left_null = left_20;
        else
            left_null = left_3dB;
        end
    end
    if right_null == idx_peak
        right_20_rel = find(R(idx_peak:end) < th_20dB, 1, 'first');
        if ~isempty(right_20_rel)
            right_null = idx_peak + right_20_rel - 1;
        else
            right_null = right_3dB;
        end
    end

    sidelobe_region = [R(1:left_null-1); R(right_null+1:end)];
    if isempty(sidelobe_region)
        PSLR = -100;
    else
        PSLR = 20*log10(max(sidelobe_region));
    end

    PAPR = max(abs(s).^2) / mean(abs(s).^2);
end

function [pslr, mw, papr] = evaluate_metrics(b, s_LFM, fs, B)
    N = length(s_LFM);
    W = legendre_window(b, fs, B, N);
    s_w = ifft(fft(s_LFM) .* W);
    [R, lag] = xcorr(s_w);
    R = safe_normalize(abs(R));
    [pslr, mw, papr] = compute_metrics_single(R, lag, s_w);
end

function fitness = fitness_func(b, s_LFM, fs, B, lambda_MW, lambda_PAPR, lambda_PSLR, MW_target, PAPR_target, PSLR_target, PSLR_floor)
    try
        N = length(s_LFM);
        W = legendre_window(b, fs, B, N);
        s_w = ifft(fft(s_LFM) .* W);
        [R, lag] = xcorr(s_w);
        R = safe_normalize(abs(R));
        [PSLR, MW, PAPR] = compute_metrics_single(R, lag, s_w);

        [R_lfm, lag_lfm] = xcorr(s_LFM);
        R_lfm = safe_normalize(abs(R_lfm));
        [~, MW_lfm, PAPR_lfm] = compute_metrics_single(R_lfm, lag_lfm, s_LFM);

        MW_factor = MW / MW_lfm;
        PAPR_factor = PAPR / PAPR_lfm;

        penalty_MW = lambda_MW * max(0, MW_factor - MW_target);
        penalty_PAPR = lambda_PAPR * max(0, PAPR_factor - PAPR_target);
        penalty_PSLR = lambda_PSLR * max(0, PSLR - PSLR_target);
        penalty_floor = 1000 * max(0, PSLR - PSLR_floor);

        fitness = PSLR + 1.2*penalty_MW + penalty_PAPR + 1.2*penalty_PSLR + penalty_floor;
        if ~isscalar(fitness)
            fitness = fitness(1);
        end
    catch
        fitness = 1000;
    end
end

function pslr = compute_PSLR(b, s_LFM, fs, B)
    N = length(s_LFM);
    W = legendre_window(b, fs, B, N);
    s_w = ifft(fft(s_LFM) .* W);
    [R, lag] = xcorr(s_w);
    R = safe_normalize(abs(R));
    [pslr, ~, ~] = compute_metrics_single(R, lag, s_w);
end

function [c, ceq] = compute_constraints_v2(b, s_LFM, fs, B, MW_target, PAPR_target, PSLR_target, MW_lfm, PAPR_lfm)
    try
        N = length(s_LFM);
        W = legendre_window(b, fs, B, N);
        s_w = ifft(fft(s_LFM) .* W);
        [R, lag] = xcorr(s_w);
        R = safe_normalize(abs(R));
        [PSLR, MW, PAPR] = compute_metrics_single(R, lag, s_w);

        MW_factor = MW / max(MW_lfm, eps);
        PAPR_factor = PAPR / max(PAPR_lfm, eps);
        c = [MW_factor - MW_target; PAPR_factor - PAPR_target; PSLR - PSLR_target];
        if any(~isfinite(c))
            c = [1e3; 1e3; 1e3];
        end
    catch
        c = [1e3; 1e3; 1e3];
    end
    ceq = [];
end


function [w_alg2, info] = design_window_alg2_wang2023(s_LFM, params)
    if ~isfield(params,'rho'), params.rho = 0.01; end
    if ~isfield(params,'Tmax'), params.Tmax = 4500; end
    if ~isfield(params,'psi'), params.psi = 1e-4; end
    if ~isfield(params,'tau'), params.tau = 1.9; end
    if ~isfield(params,'seed'), params.seed = 80; end
    if ~isfield(params,'solver_preference'), params.solver_preference = 'proj-grad'; end
    if ~isfield(params,'verbose_every'), params.verbose_every = 0; end

    rng(params.seed);
    x = s_LFM(:);
    N = length(x);
    Xdiag = conj(x);

    [R0, lag0] = xcorr(x);
    R0 = safe_normalize(abs(R0));
    [left_null, right_null] = get_mainlobe_bounds(R0, lag0);
    Omega_idx = [1:left_null-1, right_null+1:length(R0)];
    Omega_m = lag0(Omega_idx);
    M = numel(Omega_m);

    a0 = Xdiag .* x;
    am_mat = zeros(N, M);
    for k = 1:M
        sx = linear_shift_zero_pad(x, Omega_m(k));
        am_mat(:,k) = Xdiag .* sx;
    end

    rho = params.rho;
    psi = params.psi;

    % 性能优化：Rm 与其步长估计在整个 ADMM 迭代中不变，提前缓存避免每轮重复构造/谱分解
    qcqp_cache = prepare_w_qcqp_cache_alg2(a0, am_mat, rho);
    qcqp_cache = prepare_w_qcqp_cache_alg2(a0, am_mat, rho);


    w = ones(N,1);
    y = w' * a0;
    z = (w' * am_mat).';
    u = abs(y);
    v = max(abs(z));
    u_bar = max(u, eps);
    v_bar = max(v, 0);

    lambda = 0;
    kappa = zeros(M,1);
    theta = 0;
    zeta = 0;

    history.obj = nan(params.Tmax,1);
    history.residual = nan(params.Tmax,1);
    history.u = nan(params.Tmax,1);
    history.v = nan(params.Tmax,1);

    for it = 1:params.Tmax
        % Step-1: (y,z,u,v) projection update
        y_hat = (w' * a0) - lambda / rho;
        z_hat = (w' * am_mat).' - kappa / rho;
        u_hat = real(u_bar - theta / rho);
        v_hat = real(v_bar - zeta / rho);

        [y_vec, u] = project_geq_complex(y_hat, u_hat);
        y = y_vec(1);
        [z, v] = project_leq_complex(z_hat, v_hat);

        % Step-2: (u_bar,v_bar) update via fast cubic-candidate selection (with 1D fallback)
        [u_bar, v_bar] = update_uvbar_fast(u, v, theta, zeta, rho);

        % Step-3: w update (QCQP with similarity constraint)
        [w, solver_name] = update_w_qcqp(a0, am_mat, y, z, lambda, kappa, x, params.tau, rho, w, params.solver_preference, qcqp_cache);

        % Dual updates
        ry = y - (w' * a0);
        rz = z - (w' * am_mat).';
        ru = u - u_bar;
        rv = v - v_bar;

        lambda = lambda + rho * ry;
        kappa = kappa + rho * rz;
        theta = theta + rho * ru;
        zeta = zeta + rho * rv;

        res_y = abs(ry);
        res_z = sum(abs(rz));
        residual = res_y + res_z;

        history.obj(it) = real(v_bar / max(u_bar, eps));
        history.residual(it) = residual;
        history.u(it) = real(u_bar);
        history.v(it) = real(v_bar);

        if params.verbose_every > 0 && (it == 1 || mod(it, params.verbose_every) == 0)
            fprintf('[Alg2] it=%4d residual=%.3e v/u=%.6f solver=%s\n', it, residual, history.obj(it), solver_name);
        end

        if residual <= psi
            break;
        end
    end

    history.obj = history.obj(1:it);
    history.residual = history.residual(1:it);
    history.u = history.u(1:it);
    history.v = history.v(1:it);

    w_alg2 = w;
    info.iters = it;
    info.solver_name = solver_name;
    info.Omega_m = Omega_m;
    info.history = history;
end

function [left_null, right_null] = get_mainlobe_bounds(R, lag)
    dummy = ones(numel(R),1);
    [~, ~, ~] = compute_metrics_single(R, lag, dummy); %#ok<ASGLU>

    R = R(:);
    [~, idx_peak] = max(R);

    left_null = idx_peak;
    for i = idx_peak:-1:2
        if R(i-1) > R(i) && R(i) < R(i+1)
            left_null = i;
            break;
        end
    end

    right_null = idx_peak;
    for i = idx_peak:length(R)-2
        if R(i) > R(i+1) && R(i+1) < R(i+2)
            right_null = i+1;
            break;
        end
    end

    th_3dB = 10^(-3/20);
    left_3dB = find(R(1:idx_peak) < th_3dB, 1, 'last');
    if isempty(left_3dB), left_3dB = 1; end
    right_3dB_rel = find(R(idx_peak:end) < th_3dB, 1, 'first');
    if isempty(right_3dB_rel), right_3dB = length(R); else, right_3dB = idx_peak + right_3dB_rel - 1; end

    th_20dB = 10^(-20/20);
    if left_null == idx_peak
        left_20 = find(R(1:idx_peak) < th_20dB, 1, 'last');
        if ~isempty(left_20), left_null = left_20; else, left_null = left_3dB; end
    end
    if right_null == idx_peak
        right_20_rel = find(R(idx_peak:end) < th_20dB, 1, 'first');
        if ~isempty(right_20_rel), right_null = idx_peak + right_20_rel - 1; else, right_null = right_3dB; end
    end
end

function sx = linear_shift_zero_pad(x, m)
    x = x(:);
    N = length(x);
    sx = zeros(N,1);
    if m > 0
        if m < N
            sx(1+m:end) = x(1:end-m);
        end
    elseif m < 0
        m2 = -m;
        if m2 < N
            sx(1:end-m2) = x(1+m2:end);
        end
    else
        sx = x;
    end
end

function [y_proj, u_star] = project_geq_complex(y_hat_vec, u_hat)
    y_hat_vec = y_hat_vec(:);
    r = abs(y_hat_vec);
    n = numel(r);
    [rs,~] = sort(r, 'ascend');
    rs = [rs; inf];
    csum = [0; cumsum(rs(1:n))];

    best_f = inf;
    u_star = max(u_hat,0);
    for k = 0:n
        lb = 0;
        ub = inf;
        if k >= 1, lb = rs(k); end
        if k < n, ub = rs(k+1); end
        u_cand = (csum(k+1) + u_hat) / (k + 1);
        u_cand = min(max(u_cand, lb), ub);
        f = sum((max(r, u_cand) - r).^2) + (u_cand - u_hat).^2;
        if f < best_f
            best_f = f;
            u_star = u_cand;
        end
    end
    u_star = max(real(u_star), 0);
    y_proj = max(r, u_star) .* exp(1j*angle(y_hat_vec));
end

function [z_proj, v_star] = project_leq_complex(z_hat_vec, v_hat)
    z_hat_vec = z_hat_vec(:);
    r = abs(z_hat_vec);
    n = numel(r);
    [rd,~] = sort(r, 'descend');
    rd = [rd; 0];
    csum = [0; cumsum(rd(1:n))];

    best_f = inf;
    v_star = max(v_hat,0);
    for k = 0:n
        ub = inf;
        lb = 0;
        if k >= 1, ub = rd(k); end
        if k < n, lb = rd(k+1); end
        v_cand = (csum(k+1) + v_hat) / (k + 1);
        v_cand = min(max(v_cand, lb), ub);
        f = sum((min(r, v_cand) - r).^2) + (v_cand - v_hat).^2;
        if f < best_f
            best_f = f;
            v_star = v_cand;
        end
    end
    v_star = max(real(v_star), 0);
    z_proj = min(r, v_star) .* exp(1j*angle(z_hat_vec));
end

function [u_bar, v_bar] = update_uvbar_fast(u, v, theta, zeta, rho)
    cu = real(u + theta/rho);
    cv = real(v + zeta/rho);

    % 快速版本：优先使用三次方程实根候选（闭式思路），仅在异常时回退到1D搜索
    alpha = max(cu, eps);
    eta = max(real(u), eps);
    coeff = [rho, (-rho*cv + 1/alpha), 0, -eta];
    roots_v = roots(coeff);
    cand_v = real(roots_v(abs(imag(roots_v)) < 1e-8));
    cand_v = cand_v(cand_v >= 0);
    cand_v = unique([cand_v; max(cv,0); 0]);

    best_cost = inf;
    v_bar = max(cv, 0);
    u_bar = max(cu, eps);
    for i = 1:numel(cand_v)
        v_try = cand_v(i);
        u_try = max(cu + (v_try - cv), eps);
        cost = v_try/max(u_try,eps) + (rho/2)*(u_try-cu)^2 + (rho/2)*(v_try-cv)^2;
        if isfinite(cost) && cost < best_cost
            best_cost = cost;
            v_bar = v_try;
            u_bar = u_try;
        end
    end

    % 数值回退：仅在候选根不可用时才进行1D搜索（避免每轮 fminbnd）
    if ~isfinite(best_cost)
        delta = cu - cv;
        lb = max(0, -delta + eps);
        ub = max([lb + 1, cv + 5*max(1, abs(cv)), abs(v) + 5]);
        obj_v = @(vb) vb./max(vb + delta, eps) + ...
            (rho/2)*(max(vb + delta, eps)-cu).^2 + (rho/2)*(vb-cv).^2;
        if exist('fminbnd','file') == 2
            [v_bar, ~] = fminbnd(obj_v, lb, ub);
        else
            grid_v = linspace(lb, ub, 200);
            [~, idx_best] = min(obj_v(grid_v));
            v_bar = grid_v(idx_best);
        end
        v_bar = max(real(v_bar), 0);
        u_bar = max(v_bar + delta, eps);
    end
end

function [w_new, solver_name] = update_w_qcqp(a0, am_mat, y, z, lambda, kappa, x, tau, rho, w_init, solver_preference, qcqp_cache)
    N = length(x);

    if nargin < 11 || isempty(solver_preference)
        solver_preference = 'proj-grad';
    end
    if nargin < 12 || isempty(qcqp_cache)
        qcqp_cache = prepare_w_qcqp_cache_alg2(a0, am_mat, rho);
        qcqp_cache = prepare_w_qcqp_cache_alg2(a0, am_mat, rho);

    end

    Rm = qcqp_cache.Rm;
    L = qcqp_cache.L;
    rvec = -rho * (a0*(y + lambda/rho) + am_mat*(z + kappa/rho));
    use_cvx = strcmpi(solver_preference, 'cvx') || strcmpi(solver_preference, 'auto');
    use_fmincon = strcmpi(solver_preference, 'fmincon') || strcmpi(solver_preference, 'auto');

    if use_cvx && exist('cvx_begin', 'file') == 2
        solver_name = 'cvx';
        cvx_begin quiet
            variable w_cvx(N) complex
            minimize( quad_form(w_cvx, Rm) + 2*real(rvec' * w_cvx) )
            subject to
                norm( x .* (w_cvx - ones(N,1)), 2 ) <= tau
        cvx_end
        w_new = w_cvx;
        return;
    end

    if use_fmincon && exist('fmincon','file') == 2
        solver_name = 'fmincon';
        w0 = [real(w_init); imag(w_init)];
        obj = @(wr) wr_obj_qcqp(wr, Rm, rvec, N);
        nonl = @(wr) wr_nonl_qcqp(wr, x, tau, N);
        opts = optimoptions('fmincon','Display','off','Algorithm','interior-point',...
            'MaxFunctionEvaluations',4000,'OptimalityTolerance',1e-6,'StepTolerance',1e-8);
        wr = fmincon(obj, w0, [], [], [], [], [], [], nonl, opts);
        w_new = wr(1:N) + 1j*wr(N+1:end);
        return;
    end

    solver_name = 'proj-grad';
    w_new = w_init;
    step = 1/max(L, eps);
    for it = 1:250
        g = 2*(Rm*w_new + rvec);
        if norm(g,2) < 1e-6
            break;
        end
        w_prev = w_new;
        w_new = w_new - step*g;
        d = x .* (w_new - 1);
        nd = norm(d,2);
        if nd > tau
            d = d * (tau/nd);
            w_new = 1 + d ./ (x + (abs(x)<eps).*eps);
        end
        if norm(w_new - w_prev,2) < 1e-8 * max(1,norm(w_prev,2))
            break;
        end
    end
end


function cache = prepare_w_qcqp_cache_alg2(a0, am_mat, rho)
    N = length(a0);
    Rm = rho * (a0*a0' + am_mat*am_mat') + 1e-8*eye(N);

    % 用幂迭代估计 Lipschitz 常数，避免每次 w 更新都调用 eig(O(N^3))
    v = ones(N,1) / sqrt(N);
    for k = 1:20
        v = Rm * v;
        nv = norm(v,2);
        if nv <= eps
            break;
        end
        v = v / nv;
    end
    L = real(v' * Rm * v);
    if ~isfinite(L) || L <= 0
        L = 1;
    end

    cache.Rm = Rm;
    cache.L = L;
end


function cache = prepare_w_qcqp_cache_alg2(a0, am_mat, rho)
    N = length(a0);
    Rm = rho * (a0*a0' + am_mat*am_mat') + 1e-8*eye(N);

    % 用幂迭代估计 Lipschitz 常数，避免每次 w 更新都调用 eig(O(N^3))
    v = ones(N,1) / sqrt(N);
    for k = 1:20
        v = Rm * v;
        nv = norm(v,2);
        if nv <= eps
            break;
        end
        v = v / nv;
    end
    L = real(v' * Rm * v);
    if ~isfinite(L) || L <= 0
        L = 1;

    end

    cache.Rm = Rm;
    cache.L = L;
end


function cache = prepare_w_qcqp_cache(a0, am_mat, rho)
    N = length(a0);
    Rm = rho * (a0*a0' + am_mat*am_mat') + 1e-8*eye(N);

    % 用幂迭代估计 Lipschitz 常数，避免每次 w 更新都调用 eig(O(N^3))
    v = ones(N,1) / sqrt(N);
    for k = 1:20
        v = Rm * v;
        nv = norm(v,2);
        if nv <= eps
            break;
        end
        v = v / nv;
    end
    L = real(v' * Rm * v);
    if ~isfinite(L) || L <= 0
        L = 1;
    end

    cache.Rm = Rm;
    cache.L = L;
end


function f = wr_obj_qcqp(wr, Rm, rvec, N)
    w = wr(1:N) + 1j*wr(N+1:end);
    f = real(w' * Rm * w + 2*real(rvec' * w));
end

function [c, ceq] = wr_nonl_qcqp(wr, x, tau, N)
    w = wr(1:N) + 1j*wr(N+1:end);
    c = norm(x .* (w - ones(N,1)), 2) - tau;
    ceq = [];
end
