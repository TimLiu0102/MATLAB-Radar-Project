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
%  Yamaoka & Oshima 2025 仿真窗对比（不改变原有口径）
% =========================================================================

idx_band = abs(f) <= B/2;
f_norm_band = 2 * f(idx_band) / B;
t_norm = f_norm_band / 2;   % 与论文 t in [-0.5,0.5] 对齐

% 三种基窗（论文系数写死）
w_nuttall = build_yamaoka_base_window('nuttall', t_norm);
w_bh = build_yamaoka_base_window('blackman-harris', t_norm);
w_bn = build_yamaoka_base_window('blackman-nuttall', t_norm);

% PM1/PM2/PM3 因子（论文式(2)(3)(4)）
g_pm1 = cos(pi * t_norm);
g_pm2 = 0.5 + 0.5 * cos(2*pi*t_norm);
g_pm3 = 0.75 * cos(pi*t_norm) + 0.25 * cos(3*pi*t_norm);

g_pm1 = max(g_pm1, 0);
g_pm2 = max(g_pm2, 0);
g_pm3 = max(g_pm3, 0);

% 频域窗构造（带内赋值 + ifftshift）
W_N_base = build_freq_window_from_segment(w_nuttall, idx_band, N);
W_N_PM1  = build_freq_window_from_segment(w_nuttall .* g_pm1, idx_band, N);
W_N_PM2  = build_freq_window_from_segment(w_nuttall .* g_pm2, idx_band, N);
W_N_PM3  = build_freq_window_from_segment(w_nuttall .* g_pm3, idx_band, N);

W_BH_base = build_freq_window_from_segment(w_bh, idx_band, N);
W_BH_PM1  = build_freq_window_from_segment(w_bh .* g_pm1, idx_band, N);
W_BH_PM2  = build_freq_window_from_segment(w_bh .* g_pm2, idx_band, N);
W_BH_PM3  = build_freq_window_from_segment(w_bh .* g_pm3, idx_band, N);

W_BN_base = build_freq_window_from_segment(w_bn, idx_band, N);
W_BN_PM1  = build_freq_window_from_segment(w_bn .* g_pm1, idx_band, N);
W_BN_PM2  = build_freq_window_from_segment(w_bn .* g_pm2, idx_band, N);
W_BN_PM3  = build_freq_window_from_segment(w_bn .* g_pm3, idx_band, N);

method_names = {
    'Legendre_opt', ...
    'Nuttall_base', 'Nuttall_PM1', 'Nuttall_PM2', 'Nuttall_PM3', ...
    'BH_base', 'BH_PM1', 'BH_PM2', 'BH_PM3', ...
    'BN_base', 'BN_PM1', 'BN_PM2', 'BN_PM3'};

windows = {W_opt, ...
    W_N_base, W_N_PM1, W_N_PM2, W_N_PM3, ...
    W_BH_base, W_BH_PM1, W_BH_PM2, W_BH_PM3, ...
    W_BN_base, W_BN_PM1, W_BN_PM2, W_BN_PM3};

metrics = zeros(numel(method_names), 3);
metrics(1,:) = [PSLR_opt, MW_opt, PAPR_opt];

for i = 2:numel(method_names)
    s_w = ifft(fft(s_LFM) .* windows{i});
    [R_w, lag_w] = xcorr(s_w);
    R_w = safe_normalize(abs(R_w));
    [pslr_i, mw_i, papr_i] = compute_metrics_single(R_w, lag_w, s_w);
    metrics(i,:) = [pslr_i, mw_i, papr_i];
end

fprintf('\n=================== Yamaoka & Oshima 2025 Window Comparison ===================\n');
fprintf('Method\t\t\tPSLR(dB)\tMW\t\tPAPR\n');
for i = 1:numel(method_names)
    fprintf('%-14s\t%.2f\t\t%.2e\t%.2f\n', method_names{i}, metrics(i,1), metrics(i,2), metrics(i,3));
end
fprintf('================================================================================\n');

%% ========================================================================
%  算法对比图（参照 test.m：窗函数频谱 / 自相关 / 主瓣范围）
%% ========================================================================
% 选择 Yamaoka 系列中 PSLR 最优的方法用于可视化对比
[~, rel_best_y] = min(metrics(2:end,1));
idx_best_y = rel_best_y + 1;
W_best_y = windows{idx_best_y};
name_best_y = method_names{idx_best_y};

s_best_y = ifft(fft(s_LFM) .* W_best_y);
[R_best_y, lag_best_y] = xcorr(s_best_y);
R_best_y = safe_normalize(abs(R_best_y));

[R_lfm_plot, lag_plot] = xcorr(s_LFM);
R_lfm_plot = safe_normalize(abs(R_lfm_plot));

% 图1：自相关函数对比图（dB）
figure(1);
plot(lag_plot/fs*1e6, 20*log10(R_lfm_plot+eps), 'k-', 'LineWidth', 1.2); hold on;
plot(lag_opt/fs*1e6, 20*log10(R_opt+eps), 'g-', 'LineWidth', 1.6);
plot(lag_hamming_ref/fs*1e6, 20*log10(R_hamming_ref+eps), 'r--', 'LineWidth', 1.2);
plot(lag_best_y/fs*1e6, 20*log10(R_best_y+eps), 'b-.', 'LineWidth', 1.3);
xlabel('Time (us)'); ylabel('Normalized Magnitude (dB)');
legend('Original LFM','Legendre opt','Hamming',name_best_y,'Location','best');
grid on; xlim([-0.5 0.5]); ylim([-80 5]);

% 图2：主瓣范围内自相关函数图（线性幅度）
[~, idx_peak] = max(R_lfm_plot);
half_width = 60;
range_idx = max(1, idx_peak-half_width) : min(length(R_lfm_plot), idx_peak+half_width);
t_corr = lag_plot / fs * 1e6;

figure(2);
plot(t_corr(range_idx), R_lfm_plot(range_idx), 'k-', 'LineWidth', 1.2); hold on;
plot(t_corr(range_idx), R_opt(range_idx), 'g-', 'LineWidth', 1.6);
plot(t_corr(range_idx), R_hamming_ref(range_idx), 'r--', 'LineWidth', 1.2);
plot(t_corr(range_idx), R_best_y(range_idx), 'b-.', 'LineWidth', 1.3);
xlabel('Delay (us)'); ylabel('Normalized Magnitude');
legend('Original LFM','Legendre opt','Hamming',name_best_y,'Location','best');
grid on;

% 图3：窗函数频谱图（dB）
figure(3);
f_MHz = f / 1e6;
plot(f_MHz, 20*log10(abs(fftshift(W_opt))+eps), 'g-', 'LineWidth', 1.6); hold on;
plot(f_MHz, 20*log10(abs(fftshift(W_hamming_ref))+eps), 'r--', 'LineWidth', 1.2);
plot(f_MHz, 20*log10(abs(fftshift(W_best_y))+eps), 'b-.', 'LineWidth', 1.3);
xlabel('Frequency (MHz)'); ylabel('Magnitude (dB)');
legend('Legendre opt','Hamming',name_best_y,'Location','best');
grid on;

% 与 test.m 对齐：显示原始 LFM 频谱主瓣（-3 dB）范围
S_lfm = abs(fftshift(fft(s_LFM)));
[~, idx_pk_spec] = max(S_lfm);
th_spec = max(S_lfm) * 10^(-3/20);
left_spec = find(S_lfm(1:idx_pk_spec) < th_spec, 1, 'last');
if isempty(left_spec)
    left_spec = 1;
end
right_rel_spec = find(S_lfm(idx_pk_spec:end) < th_spec, 1, 'first');
if isempty(right_rel_spec)
    right_spec = length(S_lfm);
else
    right_spec = idx_pk_spec + right_rel_spec - 1;
end
xlim([f(left_spec), f(right_spec)] / 1e6);
ylim([-80, 5]);

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




function W = build_freq_window_from_segment(w_seg, idx_band, N)
    W_centered = zeros(N,1);
    W_centered(idx_band) = w_seg(:);
    W_centered = W_centered / (max(abs(W_centered)) + eps);
    W = ifftshift(W_centered);
end

function w = build_yamaoka_base_window(type_name, t_norm)
    t = t_norm(:);
    switch lower(type_name)
        case 'nuttall'
            w = 0.355768 + 0.487396*cos(2*pi*t) + 0.012604*cos(4*pi*t) + 0.012604*cos(6*pi*t);
        case 'blackman-harris'
            w = 0.35875 + 0.48829*cos(2*pi*t) + 0.014128*cos(4*pi*t) + 0.01168*cos(6*pi*t);
        case 'blackman-nuttall'
            w = 0.3635819 + 0.4891775*cos(2*pi*t) + 0.01365995*cos(4*pi*t) + 0.0106411*cos(6*pi*t);
        otherwise
            error('Unknown Yamaoka base window type: %s', type_name);
    end
end
