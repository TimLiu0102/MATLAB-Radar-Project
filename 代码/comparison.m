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
%  Yamaoka & Oshima 2025: Proposed Method 1/2/3 对比
%  保持现有“频域乘窗->IFFT->xcorr->compute_metrics_single”框架
% =========================================================================

% A) 以 Legendre 最终窗 W_opt 为 base
out_leg = apply_yamaoka_oshima_methods_to_freq_window(W_opt, fs, B);

s_leg_pm1 = ifft(fft(s_LFM) .* out_leg.W_pm1);
s_leg_pm2 = ifft(fft(s_LFM) .* out_leg.W_pm2);
s_leg_pm3 = ifft(fft(s_LFM) .* out_leg.W_pm3);

[R_leg_pm1, lag_cmp] = xcorr(s_leg_pm1); R_leg_pm1 = safe_normalize(abs(R_leg_pm1));
[R_leg_pm2, ~] = xcorr(s_leg_pm2);       R_leg_pm2 = safe_normalize(abs(R_leg_pm2));
[R_leg_pm3, ~] = xcorr(s_leg_pm3);       R_leg_pm3 = safe_normalize(abs(R_leg_pm3));

[PSLR_leg_pm1, MW_leg_pm1, PAPR_leg_pm1] = compute_metrics_single(R_leg_pm1, lag_cmp, s_leg_pm1);
[PSLR_leg_pm2, MW_leg_pm2, PAPR_leg_pm2] = compute_metrics_single(R_leg_pm2, lag_cmp, s_leg_pm2);
[PSLR_leg_pm3, MW_leg_pm3, PAPR_leg_pm3] = compute_metrics_single(R_leg_pm3, lag_cmp, s_leg_pm3);

% B) 以 Hamming 参考窗 W_hamming_ref 为 base
out_ham = apply_yamaoka_oshima_methods_to_freq_window(W_hamming_ref, fs, B);

s_ham_base = ifft(fft(s_LFM) .* W_hamming_ref);
s_ham_pm1 = ifft(fft(s_LFM) .* out_ham.W_pm1);
s_ham_pm2 = ifft(fft(s_LFM) .* out_ham.W_pm2);
s_ham_pm3 = ifft(fft(s_LFM) .* out_ham.W_pm3);

[R_ham_base, ~] = xcorr(s_ham_base); R_ham_base = safe_normalize(abs(R_ham_base));
[R_ham_pm1, ~] = xcorr(s_ham_pm1);   R_ham_pm1 = safe_normalize(abs(R_ham_pm1));
[R_ham_pm2, ~] = xcorr(s_ham_pm2);   R_ham_pm2 = safe_normalize(abs(R_ham_pm2));
[R_ham_pm3, ~] = xcorr(s_ham_pm3);   R_ham_pm3 = safe_normalize(abs(R_ham_pm3));

[PSLR_ham_base, MW_ham_base, PAPR_ham_base] = compute_metrics_single(R_ham_base, lag_cmp, s_ham_base);
[PSLR_ham_pm1, MW_ham_pm1, PAPR_ham_pm1] = compute_metrics_single(R_ham_pm1, lag_cmp, s_ham_pm1);
[PSLR_ham_pm2, MW_ham_pm2, PAPR_ham_pm2] = compute_metrics_single(R_ham_pm2, lag_cmp, s_ham_pm2);
[PSLR_ham_pm3, MW_ham_pm3, PAPR_ham_pm3] = compute_metrics_single(R_ham_pm3, lag_cmp, s_ham_pm3);

fprintf('\n=================== Yamaoka & Oshima 2025 Comparison ===================\n');
fprintf('Method\t\t\tPSLR(dB)\tMW\t\tPAPR\n');
fprintf('Legendre_opt\t\t%.2f\t\t%.2e\t%.2f\n', PSLR_opt, MW_opt, PAPR_opt);
fprintf('Legendre+PM1\t\t%.2f\t\t%.2e\t%.2f\n', PSLR_leg_pm1, MW_leg_pm1, PAPR_leg_pm1);
fprintf('Legendre+PM2\t\t%.2f\t\t%.2e\t%.2f\n', PSLR_leg_pm2, MW_leg_pm2, PAPR_leg_pm2);
fprintf('Legendre+PM3\t\t%.2f\t\t%.2e\t%.2f\n', PSLR_leg_pm3, MW_leg_pm3, PAPR_leg_pm3);
fprintf('Hamming_ref\t\t%.2f\t\t%.2e\t%.2f\n', PSLR_ham_base, MW_ham_base, PAPR_ham_base);
fprintf('Hamming+PM1\t\t%.2f\t\t%.2e\t%.2f\n', PSLR_ham_pm1, MW_ham_pm1, PAPR_ham_pm1);
fprintf('Hamming+PM2\t\t%.2f\t\t%.2e\t%.2f\n', PSLR_ham_pm2, MW_ham_pm2, PAPR_ham_pm2);
fprintf('Hamming+PM3\t\t%.2f\t\t%.2e\t%.2f\n', PSLR_ham_pm3, MW_ham_pm3, PAPR_ham_pm3);
fprintf('=======================================================================\n');
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


function out = apply_yamaoka_oshima_methods_to_freq_window(W_base, fs, B)
    N = length(W_base);
    Wc = fftshift(W_base(:));
    f = (-N/2:N/2-1)' * (fs/N);

    idx_band = abs(f) <= B/2;
    f_norm = 2 * f(idx_band) / B;
    t_norm = f_norm / 2;

    W0_seg = Wc(idx_band);
    g1 = cos(pi * t_norm);
    g2 = 0.5 + 0.5 * cos(2*pi*t_norm);
    g3 = 0.75 * cos(pi*t_norm) + 0.25 * cos(3*pi*t_norm);

    g1 = max(g1, 0);
    g2 = max(g2, 0);
    g3 = max(g3, 0);

    W1c = zeros(N,1); W1c(idx_band) = W0_seg .* g1;
    W2c = zeros(N,1); W2c(idx_band) = W0_seg .* g2;
    W3c = zeros(N,1); W3c(idx_band) = W0_seg .* g3;

    W1c = W1c / (max(abs(W1c)) + eps);
    W2c = W2c / (max(abs(W2c)) + eps);
    W3c = W3c / (max(abs(W3c)) + eps);

    out.W_pm1 = ifftshift(W1c);
    out.W_pm2 = ifftshift(W2c);
    out.W_pm3 = ifftshift(W3c);
end
