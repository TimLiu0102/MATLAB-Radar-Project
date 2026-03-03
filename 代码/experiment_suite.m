%% ========================================================================
%  LFM 窗函数与优化实验补充脚本
%  覆盖：
%  1) 窗函数对比：Hamming / Kaiser / Taylor / Chebyshev
%  2) 多次随机运行统计：PSLR、MW、PAPR 均值±方差
%  3) 收敛曲线：FA 与 FA+精修；运行时间
%  4) 参数敏感性：FA 参数、惩罚权重、约束阈值
%  5) 场景变化：TBP、采样点数、SNR
%  6) 复现实验细节输出
% =========================================================================

clear; clc; close all;

%% ------------------------- 可复现实验配置 -------------------------------
base_cfg = default_config();
fprintf('=== Reproducibility Details ===\n');
fprintf('LFM: B=%.1f MHz, T=%.2f us, fs=%.1f MHz, TBP=%.1f\n', ...
    base_cfg.B/1e6, base_cfg.T*1e6, base_cfg.fs/1e6, base_cfg.B*base_cfg.T);
fprintf('Window support: |f|<=B/2, FFT-order frequency-domain tapering.\n');
fprintf('Pulse compression: autocorrelation via xcorr(s).\n');
fprintf('Metrics: PSLR(first-null), MW(-3dB), PAPR(power ratio).\n');
fprintf('Monte Carlo runs = %d, seeds = base_seed + run_id.\n\n', base_cfg.n_runs);

%% --------------------------- 1) 窗函数对比 -------------------------------
fprintf('=== 1) Window Comparison ===\n');
compare_windows(base_cfg);

%% ---------------------- 2-3) 统计+收敛+时间复杂度 -----------------------
fprintf('\n=== 2-3) Monte Carlo and Convergence ===\n');
mc_report = run_monte_carlo(base_cfg);

%% --------------------------- 4) 参数敏感性 -------------------------------
fprintf('\n=== 4) Parameter Sensitivity ===\n');
run_sensitivity(base_cfg);

%% ---------------------------- 5) 场景变化 --------------------------------
fprintf('\n=== 5) Scenario Sweep (TBP/N/SNR) ===\n');
run_scenarios(base_cfg);

%% --------------------------- 结果汇总打印 --------------------------------
fprintf('\n=== Summary (Monte Carlo) ===\n');
print_stats('FA only', mc_report.fa_stats);
print_stats('FA + refine', mc_report.refined_stats);
fprintf('Time (s): FA mean=%.3f, FA+refine mean=%.3f\n', ...
    mc_report.time_fa_mean, mc_report.time_refine_mean);
fprintf('Complexity note: FA ~ O(n_runs * n_restart * n_iter * n_fireflies^2 * eval_cost).\n');
fprintf('Refinement adds local constrained optimization cost via fmincon.\n');

%% ========================================================================
%                               FUNCTIONS
%% ========================================================================

function cfg = default_config()
    cfg.B = 600e6;
    cfg.T = 1.7e-6;
    cfg.fs = 720e6;

    cfg.dim = 5;
    cfg.lb = -5 * ones(1, cfg.dim);
    cfg.ub = 5 * ones(1, cfg.dim);

    cfg.n_fireflies = 18;
    cfg.max_iter = 40;
    cfg.n_restarts = 2;
    cfg.gamma = 1;
    cfg.beta0 = 1;
    cfg.alpha = 0.2;

    cfg.lambda_mw = 7;
    cfg.lambda_papr = 8;
    cfg.lambda_pslr = 80;
    cfg.mw_target = 1.00;
    cfg.papr_target = 1.00;
    cfg.pslr_margin = 1.2;
    cfg.pslr_floor = -30;

    cfg.refine_delta_mw = 0.2;
    cfg.refine_delta_papr = 0.2;
    cfg.max_attempts = 5;
    cfg.margin_step = 0.2;

    cfg.n_runs = 8;
    cfg.base_seed = 80;
end

function compare_windows(cfg)
    [s_lfm, f, fs, B] = build_lfm(cfg.B, cfg.T, cfg.fs);
    N = length(s_lfm);

    win_names = {'Hamming', 'Kaiser', 'Taylor', 'Chebyshev'};
    metrics = zeros(numel(win_names), 3);

    for i = 1:numel(win_names)
        W = build_reference_window(win_names{i}, f, B, N);
        s_w = ifft(fft(s_lfm) .* W);
        [R, lag] = xcorr(s_w);
        R = safe_normalize(abs(R));
        [pslr, mw, papr] = compute_metrics_single(R, lag, s_w);
        metrics(i, :) = [pslr, mw, papr];
    end

    fprintf('Window\t\tPSLR(dB)\tMW\t\tPAPR\n');
    for i = 1:numel(win_names)
        fprintf('%-10s\t%.2f\t\t%.2e\t%.3f\n', win_names{i}, metrics(i,1), metrics(i,2), metrics(i,3));
    end
end

function report = run_monte_carlo(cfg)
    pslr_fa = zeros(cfg.n_runs, 1);
    mw_fa = zeros(cfg.n_runs, 1);
    papr_fa = zeros(cfg.n_runs, 1);
    pslr_refined = zeros(cfg.n_runs, 1);
    mw_refined = zeros(cfg.n_runs, 1);
    papr_refined = zeros(cfg.n_runs, 1);
    time_fa = zeros(cfg.n_runs, 1);
    time_refine = zeros(cfg.n_runs, 1);

    hist_fa = [];
    hist_refine = [];

    for run_id = 1:cfg.n_runs
        rng(cfg.base_seed + run_id);
        [s_lfm, f, fs, B] = build_lfm(cfg.B, cfg.T, cfg.fs);

        t0 = tic;
        [b_fa, fa_hist, ref_stats] = run_fa_only(cfg, s_lfm, f, fs, B);
        time_fa(run_id) = toc(t0);

        [pslr_fa(run_id), mw_fa(run_id), papr_fa(run_id)] = evaluate_metrics(b_fa, s_lfm, fs, B);

        t1 = tic;
        [b_refine, refine_hist] = run_refine(cfg, b_fa, s_lfm, fs, B, ref_stats);
        time_refine(run_id) = time_fa(run_id) + toc(t1);

        [pslr_refined(run_id), mw_refined(run_id), papr_refined(run_id)] = evaluate_metrics(b_refine, s_lfm, fs, B);

        if isempty(hist_fa)
            hist_fa = fa_hist(:);
        else
            hist_fa = hist_fa + pad_to(fa_hist(:), length(hist_fa));
        end

        if isempty(hist_refine)
            hist_refine = refine_hist(:);
        else
            hist_refine = hist_refine + pad_to(refine_hist(:), length(hist_refine));
        end
    end

    hist_fa = hist_fa / cfg.n_runs;
    hist_refine = hist_refine / cfg.n_runs;

    figure('Name', 'Convergence Curves');
    plot(hist_fa, 'LineWidth', 1.5); hold on;
    plot(hist_refine, 'LineWidth', 1.5);
    xlabel('Iteration'); ylabel('Objective (PSLR-like)');
    title('Average Convergence: FA vs FA+Refine');
    legend('FA only', 'FA + refine', 'Location', 'best');
    grid on;

    report.fa_stats = stats_struct(pslr_fa, mw_fa, papr_fa);
    report.refined_stats = stats_struct(pslr_refined, mw_refined, papr_refined);
    report.time_fa_mean = mean(time_fa);
    report.time_refine_mean = mean(time_refine);
end

function run_sensitivity(cfg)
    [s_lfm, f, fs, B] = build_lfm(cfg.B, cfg.T, cfg.fs);

    alpha_list = [0.05, 0.1, 0.2, 0.35];
    gamma_list = [0.4, 1.0, 1.8];
    lambda_pslr_list = [40, 80, 120];
    margin_list = [0.4, 0.8, 1.2, 1.6];

    fprintf('alpha sweep: ');
    for i = 1:numel(alpha_list)
        cfg_i = cfg; cfg_i.alpha = alpha_list(i);
        [b, ~, ~] = run_fa_only(cfg_i, s_lfm, f, fs, B);
        [pslr, mw, papr] = evaluate_metrics(b, s_lfm, fs, B);
        fprintf('[a=%.2f: PSLR=%.2f, MW=%.2e, PAPR=%.2f] ', alpha_list(i), pslr, mw, papr);
    end
    fprintf('\n');

    fprintf('gamma sweep: ');
    for i = 1:numel(gamma_list)
        cfg_i = cfg; cfg_i.gamma = gamma_list(i);
        [b, ~, ~] = run_fa_only(cfg_i, s_lfm, f, fs, B);
        [pslr, mw, papr] = evaluate_metrics(b, s_lfm, fs, B);
        fprintf('[g=%.2f: PSLR=%.2f, MW=%.2e, PAPR=%.2f] ', gamma_list(i), pslr, mw, papr);
    end
    fprintf('\n');

    fprintf('lambda_pslr sweep: ');
    for i = 1:numel(lambda_pslr_list)
        cfg_i = cfg; cfg_i.lambda_pslr = lambda_pslr_list(i);
        [b, ~, ~] = run_fa_only(cfg_i, s_lfm, f, fs, B);
        [pslr, mw, papr] = evaluate_metrics(b, s_lfm, fs, B);
        fprintf('[l=%.0f: PSLR=%.2f, MW=%.2e, PAPR=%.2f] ', lambda_pslr_list(i), pslr, mw, papr);
    end
    fprintf('\n');

    fprintf('constraint margin sweep: ');
    for i = 1:numel(margin_list)
        cfg_i = cfg; cfg_i.pslr_margin = margin_list(i);
        [b, ~, ~] = run_fa_only(cfg_i, s_lfm, f, fs, B);
        [pslr, mw, papr] = evaluate_metrics(b, s_lfm, fs, B);
        fprintf('[m=%.1f dB: PSLR=%.2f, MW=%.2e, PAPR=%.2f] ', margin_list(i), pslr, mw, papr);
    end
    fprintf('\n');
end

function run_scenarios(cfg)
    scenario_grid = [
        400e6, 1.5e-6, 600e6, 20;
        600e6, 1.7e-6, 720e6, 10;
        900e6, 1.0e-6, 1200e6, 0
    ];

    fprintf('B(MHz)\tT(us)\tfs(MHz)\tSNR(dB)\tN\tTBP\tPSLR\tMW\tPAPR\n');

    for i = 1:size(scenario_grid, 1)
        B = scenario_grid(i,1);
        T = scenario_grid(i,2);
        fs = scenario_grid(i,3);
        snr_db = scenario_grid(i,4);

        [s_lfm, f] = build_lfm(B, T, fs);
        s_noisy = add_awgn(s_lfm, snr_db);

        cfg_i = cfg;
        [b, ~, ~] = run_fa_only(cfg_i, s_noisy, f, fs, B);
        [pslr, mw, papr] = evaluate_metrics(b, s_noisy, fs, B);

        N = length(s_lfm);
        tbp = B * T;
        fprintf('%.0f\t%.2f\t%.0f\t%.0f\t\t%d\t%.1f\t%.2f\t%.2e\t%.2f\n', ...
            B/1e6, T*1e6, fs/1e6, snr_db, N, tbp, pslr, mw, papr);
    end
end

function [b_best, hist_best, ref_stats] = run_fa_only(cfg, s_lfm, f, fs, B)
    N = length(s_lfm);

    idx_band_ref = abs(f) <= B/2;
    N_band_ref = sum(idx_band_ref);
    W_hamming_centered_ref = zeros(N, 1);
    W_hamming_centered_ref(idx_band_ref) = hamming(N_band_ref);
    W_hamming_centered_ref = W_hamming_centered_ref / (max(W_hamming_centered_ref) + eps);
    W_hamming_ref = ifftshift(W_hamming_centered_ref);
    s_hamming_ref = ifft(fft(s_lfm) .* W_hamming_ref);
    [R_hamming_ref, lag_hamming_ref] = xcorr(s_hamming_ref);
    R_hamming_ref = safe_normalize(abs(R_hamming_ref));
    [pslr_hamming_ref, mw_hamming_ref, papr_hamming_ref] = ...
        compute_metrics_single(R_hamming_ref, lag_hamming_ref, s_hamming_ref);

    pslr_target = pslr_hamming_ref - cfg.pslr_margin;

    global_best = inf;
    b_best = zeros(1, cfg.dim);
    hist_best = [];

    for restart_id = 1:cfg.n_restarts
        fireflies = cfg.lb + (cfg.ub - cfg.lb) .* rand(cfg.n_fireflies, cfg.dim);
        fitness = zeros(cfg.n_fireflies, 1);

        for i = 1:cfg.n_fireflies
            fitness(i) = fitness_func(fireflies(i,:), s_lfm, fs, B, cfg, pslr_target);
        end

        iter_hist = zeros(cfg.max_iter, 1);

        for iter = 1:cfg.max_iter
            [fitness, idx] = sort(fitness);
            fireflies = fireflies(idx, :);

            for i = 1:cfg.n_fireflies
                for j = 1:cfg.n_fireflies
                    if fitness(j) < fitness(i)
                        r = norm(fireflies(i,:) - fireflies(j,:));
                        beta = cfg.beta0 * exp(-cfg.gamma * r^2);
                        fireflies(i,:) = fireflies(i,:) + ...
                            beta * (fireflies(j,:) - fireflies(i,:)) + ...
                            cfg.alpha * (rand(1, cfg.dim) - 0.5) .* (cfg.ub - cfg.lb);
                        fireflies(i,:) = max(min(fireflies(i,:), cfg.ub), cfg.lb);
                        fitness(i) = fitness_func(fireflies(i,:), s_lfm, fs, B, cfg, pslr_target);
                    end
                end
            end

            iter_hist(iter) = min(fitness);
        end

        [fitness, idx] = sort(fitness);
        fireflies = fireflies(idx, :);
        if fitness(1) < global_best
            global_best = fitness(1);
            b_best = fireflies(1, :);
            hist_best = iter_hist;
        end
    end

    [R_lfm_ref, lag_lfm_ref] = xcorr(s_lfm);
    R_lfm_ref = safe_normalize(abs(R_lfm_ref));
    [~, mw_lfm, papr_lfm] = compute_metrics_single(R_lfm_ref, lag_lfm_ref, s_lfm);

    ref_stats.pslr_hamming_ref = pslr_hamming_ref;
    ref_stats.mw_hamming_factor = mw_hamming_ref / max(mw_lfm, eps);
    ref_stats.papr_hamming_factor = papr_hamming_ref / max(papr_lfm, eps);
    ref_stats.mw_lfm = mw_lfm;
    ref_stats.papr_lfm = papr_lfm;
end

function [b_refined, hist_refine] = run_refine(cfg, b_init, s_lfm, fs, B, ref_stats)
    b_refined = b_init;
    hist_refine = [];

    if exist('fmincon', 'file') ~= 2
        hist_refine = compute_PSLR(b_init, s_lfm, fs, B) * ones(20, 1);
        return;
    end

    pslr_before = compute_PSLR(b_init, s_lfm, fs, B);
    mw_target_refined = ref_stats.mw_hamming_factor + cfg.refine_delta_mw;
    papr_target_refined = ref_stats.papr_hamming_factor + cfg.refine_delta_papr;

    obj_fun = @(b) compute_PSLR(b, s_lfm, fs, B);
    options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'interior-point', ...
        'OptimalityTolerance', 1e-4, 'ConstraintTolerance', 1e-4, ...
        'StepTolerance', 1e-8, 'MaxFunctionEvaluations', 1500);

    hist_refine = pslr_before * ones(20, 1);

    accepted = false;
    best_try = b_init;
    best_pslr = pslr_before;

    for attempt = 1:cfg.max_attempts
        used_margin = max(0, cfg.pslr_margin - (attempt-1) * cfg.margin_step);
        pslr_target_refined = ref_stats.pslr_hamming_ref - used_margin;

        nonlcon = @(b) compute_constraints_v2(b, s_lfm, fs, B, mw_target_refined, ...
            papr_target_refined, pslr_target_refined, ref_stats.mw_lfm, ref_stats.papr_lfm);

        [b_try, fval_try, exitflag] = fmincon(obj_fun, best_try, [], [], [], [], cfg.lb, cfg.ub, nonlcon, options);
        [c_try, ~] = nonlcon(b_try);

        if exitflag > 0 && all(c_try <= 1e-6) && fval_try <= pslr_before
            accepted = true;
            best_try = b_try;
            best_pslr = fval_try;
            break;
        end
    end

    if accepted
        b_refined = best_try;
        hist_refine(:) = linspace(pslr_before, best_pslr, length(hist_refine));
    end
end

function y = pad_to(x, target_len)
    if length(x) < target_len
        y = [x; x(end) * ones(target_len - length(x), 1)];
    elseif length(x) > target_len
        y = x(1:target_len);
    else
        y = x;
    end
end

function st = stats_struct(pslr, mw, papr)
    st.pslr_mean = mean(pslr);
    st.pslr_var = var(pslr);
    st.mw_mean = mean(mw);
    st.mw_var = var(mw);
    st.papr_mean = mean(papr);
    st.papr_var = var(papr);
end

function print_stats(name, st)
    fprintf('%s: PSLR=%.3f ± %.3f(var), MW=%.3e ± %.3e(var), PAPR=%.3f ± %.3f(var)\n', ...
        name, st.pslr_mean, st.pslr_var, st.mw_mean, st.mw_var, st.papr_mean, st.papr_var);
end

function [s_lfm, f, fs, B] = build_lfm(B, T, fs)
    N = round(T * fs);
    t = (-N/2:N/2-1)' / fs;
    f = (-N/2:N/2-1)' * (fs/N);
    k = B / T;
    s_lfm = exp(1j * pi * k * t.^2);
end

function s_out = add_awgn(s_in, snr_db)
    sig_power = mean(abs(s_in).^2);
    snr_linear = 10^(snr_db/10);
    noise_power = sig_power / max(snr_linear, eps);
    n = sqrt(noise_power/2) * (randn(size(s_in)) + 1j * randn(size(s_in)));
    s_out = s_in + n;
end

function W = build_reference_window(name, f, B, N)
    idx_band = abs(f) <= B/2;
    N_band = sum(idx_band);

    switch lower(name)
        case 'hamming'
            w = hamming(N_band);
        case 'kaiser'
            w = kaiser(N_band, 6);
        case 'taylor'
            if exist('taylorwin', 'file') == 2
                w = taylorwin(N_band, 4, -35);
            else
                w = kaiser(N_band, 5);
            end
        case 'chebyshev'
            if exist('chebwin', 'file') == 2
                w = chebwin(N_band, 60);
            else
                w = kaiser(N_band, 7);
            end
        otherwise
            w = hamming(N_band);
    end

    W_centered = zeros(N, 1);
    W_centered(idx_band) = w;
    W_centered = W_centered / (max(W_centered) + eps);
    W = ifftshift(W_centered);
end

function [pslr, mw, papr] = evaluate_metrics(b, s_lfm, fs, B)
    N = length(s_lfm);
    W = legendre_window(b, fs, B, N);
    s_w = ifft(fft(s_lfm) .* W);
    [R, lag] = xcorr(s_w);
    R = safe_normalize(abs(R));
    [pslr, mw, papr] = compute_metrics_single(R, lag, s_w);
end

function fitness = fitness_func(b, s_lfm, fs, B, cfg, pslr_target)
    N = length(s_lfm);
    W = legendre_window(b, fs, B, N);
    s_w = ifft(fft(s_lfm) .* W);

    [R, lag] = xcorr(s_w);
    R = safe_normalize(abs(R));
    [pslr, mw, papr] = compute_metrics_single(R, lag, s_w);

    [R_lfm, lag_lfm] = xcorr(s_lfm);
    R_lfm = safe_normalize(abs(R_lfm));
    [~, mw_lfm, papr_lfm] = compute_metrics_single(R_lfm, lag_lfm, s_lfm);

    mw_factor = mw / max(mw_lfm, eps);
    papr_factor = papr / max(papr_lfm, eps);

    penalty_mw = cfg.lambda_mw * max(0, mw_factor - cfg.mw_target);
    penalty_papr = cfg.lambda_papr * max(0, papr_factor - cfg.papr_target);
    penalty_pslr = cfg.lambda_pslr * max(0, pslr - pslr_target);
    penalty_floor = 1000 * max(0, pslr - cfg.pslr_floor);

    fitness = pslr + 1.2 * penalty_mw + penalty_papr + 1.2 * penalty_pslr + penalty_floor;
end

function pslr = compute_PSLR(b, s_lfm, fs, B)
    N = length(s_lfm);
    W = legendre_window(b, fs, B, N);
    s_w = ifft(fft(s_lfm) .* W);
    [R, lag] = xcorr(s_w);
    R = safe_normalize(abs(R));
    [pslr, ~, ~] = compute_metrics_single(R, lag, s_w);
end

function [c, ceq] = compute_constraints_v2(b, s_lfm, fs, B, mw_target, papr_target, pslr_target, mw_lfm, papr_lfm)
    N = length(s_lfm);
    W = legendre_window(b, fs, B, N);
    s_w = ifft(fft(s_lfm) .* W);
    [R, lag] = xcorr(s_w);
    R = safe_normalize(abs(R));
    [pslr, mw, papr] = compute_metrics_single(R, lag, s_w);

    mw_factor = mw / max(mw_lfm, eps);
    papr_factor = papr / max(papr_lfm, eps);

    c = [mw_factor - mw_target; papr_factor - papr_target; pslr - pslr_target];
    if any(~isfinite(c))
        c = [1e3; 1e3; 1e3];
    end
    ceq = [];
end

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
            P = legendre_poly(n, f_seg);
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

function P = legendre_poly(n, x)
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
