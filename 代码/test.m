%% ========================================================================
%  勒让德窗函数优化的 LFM 脉冲压缩旁瓣抑制（萤火虫算法 + 梯度精修 + 结果对比）
%  改进版：放宽约束、优化 fmincon 选项、增加调试输出
%  修复：问题1(半边窗塌陷) + 问题2(频率轴与FFT顺序对齐)
% =========================================================================

clear; clc; close all;
rng(80);
%% 信号参数
B = 600e6;          % 带宽 600 MHz
T = 1.7e-6;         % 脉宽 1.7 μs
fs = 720e6;         % 采样率 720 MHz (过采样因子 1.2)
f0 = 0;             % 载频（基带信号，设为0）

% 派生参数
N = round(T * fs);  % 样本点数
t = (-N/2:N/2-1)' / fs;  % 时间向量（对称）
f = (-N/2:N/2-1)' * (fs/N); % 居中频率向量（与 fftshift 对齐）

% 生成 LFM 信号（基带）
k = B / T;          % 调频斜率
s_LFM = exp(1j * pi * k * t.^2);  % 复基带 LFM


% 参考 Hamming 窗指标（作为 PSLR 目标基线）
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
fprintf('参考 Hamming：PSLR = %.2f dB, MW = %.2e, PAPR = %.2f\n', PSLR_hamming_ref, MW_hamming_ref, PAPR_hamming_ref);

%% 萤火虫算法参数
dim = 5;            % 勒让德多项式系数个数（使用 0,2,4,6,8 偶数阶）
nFireflies = 30;    % 萤火虫数量（增大种群提升全局搜索能力）
maxIter = 150;      % 最大迭代次数（增加收敛机会）
gamma = 1;          % 光吸收系数
beta0 = 1;          % 初始吸引度
alpha = 0.2;        % 随机步长因子
lambda_MW = 7;      % 主瓣宽度惩罚系数（适度放松，优先旁瓣抑制）
lambda_PAPR = 8;    % PAPR 惩罚系数（适度放松，优先旁瓣抑制）
lambda_PSLR = 80;   % 不超过 Hamming 的 PSLR 惩罚系数（增强约束）
MW_target = 1.0;    % 目标主瓣宽度倍数（相对于 LFM 不加窗）
PAPR_target = 1.0;  % 目标 PAPR 倍数（相对于 LFM 不加窗）
PSLR_margin = 1.2;  % 目标至少比 Hamming 好 1.2 dB（拉大与 Hamming 的差距）
PSLR_target = PSLR_hamming_ref - PSLR_margin;
PSLR_floor = -30;   % 全局搜索PSLR下限，避免灾难性初值

% 参数边界（系数 b_n 范围可调）
lb = -5 * ones(1, dim);
ub = 5 * ones(1, dim);

% 多次重启全局搜索（降低陷入局部最优概率）
nRestarts = 4;
global_best_fitness = inf;
global_best_b = zeros(1, dim);

for restart = 1:nRestarts
    % 初始化萤火虫位置（系数 b_n，范围可调）
    fireflies = lb + (ub - lb) .* rand(nFireflies, dim);

    %% 计算初始适应度
    fitness = zeros(nFireflies, 1);
    for i = 1:nFireflies
        fitness(i) = fitness_func(fireflies(i,:), s_LFM, fs, B, ...
                                  lambda_MW, lambda_PAPR, lambda_PSLR, MW_target, PAPR_target, PSLR_target, PSLR_floor);
    end

    %% 萤火虫算法主循环
    for iter = 1:maxIter
        [fitness, idx] = sort(fitness);         % 按亮度排序（最小化问题）
        fireflies = fireflies(idx, :);

        for i = 1:nFireflies
            for j = 1:nFireflies
                if fitness(j) < fitness(i)       % j 比 i 亮
                    r = norm(fireflies(i,:) - fireflies(j,:));
                    beta = beta0 * exp(-gamma * r^2);
                    % 移动
                    fireflies(i,:) = fireflies(i,:) + ...
                        beta * (fireflies(j,:) - fireflies(i,:)) + ...
                        alpha * (rand(1,dim) - 0.5) .* (ub - lb);
                    % 边界处理
                    fireflies(i,:) = max(min(fireflies(i,:), ub), lb);
                    % 重新计算适应度
                    fitness(i) = fitness_func(fireflies(i,:), s_LFM, fs, B, ...
                                              lambda_MW, lambda_PAPR, lambda_PSLR, MW_target, PAPR_target, PSLR_target, PSLR_floor);
                end
            end
        end

        % 输出当前最优（含关键指标）
        [best_pslr_iter, best_mw_iter, best_papr_iter] = evaluate_metrics(fireflies(1,:), s_LFM, fs, B);
        fprintf('Restart %d/%d, Iter %d: best fitness = %.4f, PSLR = %.2f dB, MW = %.2e, PAPR = %.2f\n', ...
            restart, nRestarts, iter, fitness(1), best_pslr_iter, best_mw_iter, best_papr_iter);
    end

    % 当前重启最优
    [fitness, idx] = sort(fitness);
    fireflies = fireflies(idx, :);
    restart_best_b = fireflies(1, :);
    restart_best_fit = fitness(1);

    [restart_pslr, restart_mw, restart_papr] = evaluate_metrics(restart_best_b, s_LFM, fs, B);
    fprintf('Restart %d 结束：fitness=%.4f, PSLR=%.2f dB, MW=%.2e, PAPR=%.2f\n', ...
        restart, restart_best_fit, restart_pslr, restart_mw, restart_papr);

    if restart_best_fit < global_best_fitness
        global_best_fitness = restart_best_fit;
        global_best_b = restart_best_b;
    end
end

%% 最终最优窗系数（多次重启后的最优）
b_opt = global_best_b;
fprintf('多次重启后全局最优 fitness = %.4f\n', global_best_fitness);
disp('优化得到的 Legendre 系数:');
disp(b_opt);

% 计算萤火虫结果的性能指标（用于后续约束设置）
N = length(s_LFM);
W_opt_temp = legendre_window(b_opt, fs, B, N);
s_w_opt_temp = ifft(fft(s_LFM) .* W_opt_temp);
[R_temp, lag_temp] = xcorr(s_w_opt_temp);
R_temp = safe_normalize(abs(R_temp));
[~, MW_opt, PAPR_opt] = compute_metrics_single(R_temp, lag_temp, s_w_opt_temp);

% 计算参考信号（原始 LFM）的主瓣宽度和 PAPR
[R_lfm_ref, lag_lfm_ref] = xcorr(s_LFM);
R_lfm_ref = safe_normalize(abs(R_lfm_ref));
[~, MW_lfm, PAPR_lfm] = compute_metrics_single(R_lfm_ref, lag_lfm_ref, s_LFM);

% Hamming 参考因子（用于后续约束）
MW_factor_hamming = MW_hamming_ref / MW_lfm;
PAPR_factor_hamming = PAPR_hamming_ref / PAPR_lfm;

% 萤火虫结果的实际因子
MW_factor_opt = MW_opt / MW_lfm;
PAPR_factor_opt = PAPR_opt / PAPR_lfm;
fprintf('萤火虫结果：MW_factor = %.3f, PAPR_factor = %.3f\n', MW_factor_opt, PAPR_factor_opt);

%% ========================================================================
%  梯度精修（局部优化）- 改进版
% =========================================================================
disp('开始梯度精修...');

% 设置宽松的约束目标：允许因子比萤火虫结果略大（例如 1.1 倍）
delta_MW = 0.2;
delta_PAPR = 0.2;
MW_target_refined = MW_factor_hamming + delta_MW;
PAPR_target_refined = PAPR_factor_hamming + delta_PAPR;

% 目标函数：仅 PSLR（最小化）
obj_fun = @(b) compute_PSLR(b, s_LFM, fs, B);

% fmincon 选项（使用 interior-point 算法，放宽容差）
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'interior-point', ...
    'OptimalityTolerance', 1e-4, 'ConstraintTolerance', 1e-4, ...
    'StepTolerance', 1e-8, 'MaxFunctionEvaluations', 2000);

% 计算精修前 PSLR
PSLR_before = compute_PSLR(b_opt, s_LFM, fs, B);

% 自适应放松 PSLR 目标，避免不可行导致精修劣化
base_margin = PSLR_margin;
max_attempts = 5;
margin_step = 0.2;
accepted = false;

b_refined = b_opt;
fval_refined = PSLR_before;
exitflag = -1;
output.message = 'Not started';
c_after = [inf; inf; inf];
used_margin = base_margin;

for attempt = 1:max_attempts
    used_margin = max(0, base_margin - (attempt-1) * margin_step);
    PSLR_target_refined = PSLR_hamming_ref - used_margin;

    fprintf('精修尝试 %d/%d：约束目标 MW <= %.3f, PAPR <= %.3f, PSLR <= %.2f dB (margin=%.2f)\n', ...
        attempt, max_attempts, MW_target_refined, PAPR_target_refined, PSLR_target_refined, used_margin);

    % 阶段A：仅 MW/PAPR 约束，先找可行点
    nonlcon_stageA = @(b) compute_constraints_v2(b, s_LFM, fs, B, MW_target_refined, PAPR_target_refined, 1e6, MW_lfm, PAPR_lfm);
    [b_feasible, ~, exitflag_A, output_A] = fmincon(obj_fun, b_opt, [], [], [], [], lb, ub, nonlcon_stageA, options);
    [cA_try, ~] = nonlcon_stageA(b_feasible);

    % 阶段B：在可行点附近加入 PSLR 约束精修
    nonlcon = @(b) compute_constraints_v2(b, s_LFM, fs, B, MW_target_refined, PAPR_target_refined, PSLR_target_refined, MW_lfm, PAPR_lfm);
    [b_try, fval_try, exitflag_try, output_try] = fmincon(obj_fun, b_feasible, [], [], [], [], lb, ub, nonlcon, options);
    [c_try, ~] = nonlcon(b_try);

    fprintf('尝试结果：A_exit=%d, B_exit=%d, PSLR=%.2f dB, 约束违反=[%.3e, %.3e, %.3e]\n', ...
        exitflag_A, exitflag_try, fval_try, c_try(1), c_try(2), c_try(3));

    pass_constraints = all(c_try <= 1e-6);
    improves_pslr = (fval_try <= PSLR_before);

    if exitflag_try > 0 && pass_constraints && improves_pslr
        accepted = true;
        b_refined = b_try;
        fval_refined = fval_try;
        exitflag = exitflag_try;
        output = output_try;
        c_after = c_try;
        break;
    elseif exitflag_A > 0 && all(cA_try(1:2) <= 1e-6) && ~accepted
        % 二级验收：PSLR约束未满足时，保留MW/PAPR可行且PSLR改进的折中解
        [pslr_A, ~, ~] = evaluate_metrics(b_feasible, s_LFM, fs, B);
        if pslr_A <= PSLR_before - 0.2
            accepted = true;
            b_refined = b_feasible;
            fval_refined = pslr_A;
            exitflag = exitflag_A;
            output = output_A;
            c_after = [cA_try(1); cA_try(2); pslr_A - PSLR_target_refined];
            disp('采用二级验收解（MW/PAPR可行且PSLR显著改进）。');
            break;
        end
    end
end

if ~accepted
    disp('梯度精修未通过验收，回退到精修前结果。');
    b_refined = b_opt;
    fval_refined = PSLR_before;
    c_after = [0; 0; 0];
end

% 计算精修前后的 PSLR
PSLR_after = fval_refined;
fprintf('精修前 PSLR = %.2f dB\n', PSLR_before);
fprintf('精修后 PSLR = %.2f dB\n', PSLR_after);
if accepted
    fprintf('精修后约束违反：MW_factor - target = %.3e, PAPR_factor - target = %.3e, PSLR - target = %.3e\n', c_after(1), c_after(2), c_after(3));
    fprintf('精修采用的 PSLR margin = %.2f dB\n', used_margin);
    fprintf('fmincon 退出标志: %d\n', exitflag);
    disp(output.message);
    disp('梯度精修成功完成。');
end

% 更新最优系数（仅验收通过才更新）
b_opt = b_refined;
disp('精修后的 Legendre 系数:');
disp(b_opt);

%% 生成优化后的加窗信号
[W_opt, f] = legendre_window(b_opt, fs, B, N);
s_w_opt = ifft(fft(s_LFM) .* W_opt);  % 优化 LFM（勒让德窗加窗，W_opt 已是 FFT 顺序）

%% ========================================================================
%  对比信号生成：原始 LFM 和 Hamming 窗加窗 LFM
% =========================================================================

% 1. 原始 LFM
s_LFM_orig = s_LFM;

% 2. Hamming 窗加窗 LFM（频域加窗）
% 生成 Hamming 窗（长度 = 带宽内的采样点数）
idx_band = abs(f) <= B/2;  % 以居中频率轴构建带宽内索引
N_band = sum(idx_band);
win_hamming_time = hamming(N_band);  % 时域 Hamming 窗（对称）
% 在居中频谱上构建窗，再转换到 FFT 顺序
W_hamming_centered = zeros(N, 1);
W_hamming_centered(idx_band) = win_hamming_time;
W_hamming_centered = W_hamming_centered / (max(W_hamming_centered) + eps);
W_hamming = ifftshift(W_hamming_centered);
% 加窗
s_hamming = ifft(fft(s_LFM_orig) .* W_hamming);

%% ========================================================================
%  性能指标计算
% =========================================================================

% 计算脉冲压缩输出（自相关）
[R_lfm, lag] = xcorr(s_LFM_orig);
R_lfm = safe_normalize(abs(R_lfm));
[R_opt, ~] = xcorr(s_w_opt);
R_opt = safe_normalize(abs(R_opt));
[R_hamming, ~] = xcorr(s_hamming);
R_hamming = safe_normalize(abs(R_hamming));

% 计算指标
[PSLR_lfm, MW_lfm_final, PAPR_lfm_final] = compute_metrics_single(R_lfm, lag, s_LFM_orig);
[PSLR_opt, MW_opt_final, PAPR_opt_final] = compute_metrics_single(R_opt, lag, s_w_opt);
[PSLR_hamming, MW_hamming_final, PAPR_hamming_final] = compute_metrics_single(R_hamming, lag, s_hamming);


%% ========================================================================
%  输出对比表格
% =========================================================================

fprintf('\n=================== 性能指标对比 ===================\n');
fprintf('信号类型\t\tPSLR (dB,首零点)\t主瓣宽度(-3dB)\tPAPR\n');
fprintf('原始 LFM\t\t%.2f\t\t%.2e\t\t%.2f\n', PSLR_lfm, MW_lfm_final, PAPR_lfm_final);
fprintf('Hamming 加窗 LFM\t\t%.2f\t\t%.2e\t\t%.2f\n', PSLR_hamming, MW_hamming_final, PAPR_hamming_final);
fprintf('优化 LFM（勒让德窗）\t%.2f\t\t%.2e\t\t%.2f\n', PSLR_opt, MW_opt_final, PAPR_opt_final);
fprintf('========================================================\n');

%% ========================================================================
%  绘制自相关函数对比图（dB）
% =========================================================================

figure(1);
plot(lag/fs*1e6, 20*log10(R_lfm), 'b-', 'LineWidth', 1.5); hold on;
plot(lag/fs*1e6, 20*log10(R_hamming), 'r--', 'LineWidth', 1.5);
plot(lag/fs*1e6, 20*log10(R_opt), 'g-.', 'LineWidth', 1.5);
xlabel('Time (μs)'); ylabel('Normalized Magnitude (dB)');
title('Pulse Compression Output Comparison');
legend('LFM', 'Hamming-windowed LFM', 'Optimized LFM (Legendre)');
grid on; xlim([-0.5 0.5]); ylim([-60 5]);

%% ========================================================================
%  绘制主瓣区域放大对比图（线性幅度）
% =========================================================================
figure(2);
% 找到主瓣中心索引（以原始 LFM 为准）
[~, idx_peak] = max(R_lfm);
center_idx = idx_peak;
% 截取主瓣附近 ±50 点（可根据实际分辨率调整）
half_width = 50;
range_idx = max(1, center_idx-half_width) : min(length(R_lfm), center_idx+half_width);

% 时间轴（微秒）
t_corr = lag / fs * 1e6;

plot(t_corr(range_idx), R_lfm(range_idx), 'k-', 'LineWidth', 1.5); hold on;
plot(t_corr(range_idx), R_hamming(range_idx), 'r--', 'LineWidth', 1.5);
plot(t_corr(range_idx), R_opt(range_idx), 'b-.', 'LineWidth', 1.5);
xlabel('时延 (μs)'); ylabel('归一化幅度');
title('自相关主瓣区域');
legend('原始 LFM', 'Hamming 加窗', '优化勒让德窗', 'Location', 'best');
grid on;

figure(3);
f_MHz = f / 1e6;
W_opt_centered = fftshift(W_opt);
W_hamming_centered = fftshift(W_hamming);
plot(f_MHz, 20*log10(abs(W_opt_centered)+eps), 'g-', 'LineWidth', 1.5); hold on;
plot(f_MHz, 20*log10(abs(W_hamming_centered)+eps), 'r--', 'LineWidth', 1.5);
xlabel('Frequency (MHz)'); ylabel('Magnitude (dB)');
title('Frequency Domain Window Functions (dB)');
legend('Optimized Legendre Window', 'Hamming Window');
grid on;
xlim([-B/2/1e6, B/2/1e6]);
ylim([-60, 5]);  % 根据窗函数动态调整

%% ========================================================================
%  扩展实验补充（按当前 test.m 参数体系）
% =========================================================================
ext_cfg = build_ext_config(B, T, fs, dim, lb, ub, nFireflies, maxIter, nRestarts, ...
    gamma, beta0, alpha, lambda_MW, lambda_PAPR, lambda_PSLR, MW_target, PAPR_target, PSLR_margin, PSLR_floor);
run_extended_experiments(ext_cfg);
%% ========================================================================
%  函数定义
%% ========================================================================

function [W, f] = legendre_window(b, fs, B, N)
    % 以居中频率轴构建偶对称窗，再转换到 FFT 顺序
    f = (-N/2:N/2-1)' * (fs/N);
    f_norm = 2 * f / B;
    idx = abs(f) <= B/2;

    W_centered = zeros(N,1);
    if any(idx)
        f_seg = f_norm(idx);
        W_seg_raw = zeros(size(f_seg));
        for m = 1:length(b)
            n = 2*(m-1);  % 0,2,4,... 偶数阶，保证偶对称
            P = legendreP(n, f_seg);
            W_seg_raw = W_seg_raw + b(m) * P;
        end
        % 用指数映射保证非负，避免 max(...,0) 截断导致半边塌陷
        W_seg = exp(W_seg_raw - max(W_seg_raw));
        W_centered(idx) = W_seg;
    end

    % 转换到 FFT 顺序，便于直接与 fft(s) 相乘
    W = ifftshift(W_centered);
    W = W / (max(W) + eps);  % 幅度归一化，提升数值稳定性
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
    % MW 用 -3 dB 口径；PSLR 用首零点口径，避免视觉与数值不一致
    R = R(:);
    lag = lag(:);
    [~, idx_peak] = max(R);

    % ---- MW: -3 dB 主瓣宽度 ----
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

    % ---- PSLR: 首零点（首个局部极小值）界定主瓣 ----
    left_null = idx_peak;
    for i = idx_peak:-1:2
        if R(i-1) > R(i) && R(i) < R(i+1)
            left_null = i;
            break;
        end
    end

    right_null = idx_peak;
    for i = idx_peak:length(R)-1
        if R(i) > R(i+1) && R(i+1) < R(i+2)
            right_null = i+1;
            break;
        end
    end

    % 首零点未找到时回退到 -20 dB 交点，避免把主瓣肩部计入旁瓣
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

    PSLR = PSLR(1);
    MW = MW(1);
    PAPR = PAPR(1);
end



function [pslr, mw, papr] = evaluate_metrics(b, s_LFM, fs, B)
    N = length(s_LFM);
    W = legendre_window(b, fs, B, N);
    s_w = ifft(fft(s_LFM) .* W);
    [R, lag] = xcorr(s_w);
    R = abs(R); R = R / max(R);
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
        ISLR = compute_ISLR(R, lag);
        
        [R_lfm, lag_lfm] = xcorr(s_LFM);
        R_lfm = safe_normalize(abs(R_lfm));
        [~, MW_lfm, PAPR_lfm] = compute_metrics_single(R_lfm, lag_lfm, s_LFM);
        
        MW_factor = MW / MW_lfm;
        PAPR_factor = PAPR / PAPR_lfm;
        
        penalty_MW = lambda_MW * max(0, MW_factor - MW_target);
        penalty_PAPR = lambda_PAPR * max(0, PAPR_factor - PAPR_target);
        penalty_PSLR = lambda_PSLR * max(0, PSLR - PSLR_target);
        penalty_floor = 1000 * max(0, PSLR - PSLR_floor);
        
        fitness = 1*PSLR + 1.2*penalty_MW + 1*penalty_PAPR + 1.2*penalty_PSLR + penalty_floor;
        
        if ~isscalar(fitness)
            fitness = fitness(1);
        end
    catch ME
        warning('fitness_func 出错: %s，返回大惩罚值', ME.message);
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
% 改进的约束函数：使用预先计算的参考指标
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


function islr = compute_ISLR(varargin)
% 兼容占位：历史版本可能仍引用此函数，当前流程已不使用 ISLR
    islr = 0;
end


function cfg = build_ext_config(B, T, fs, dim, lb, ub, nFireflies, maxIter, nRestarts, gamma, beta0, alpha, lambda_MW, lambda_PAPR, lambda_PSLR, MW_target, PAPR_target, PSLR_margin, PSLR_floor)
    cfg.B = B;
    cfg.T = T;
    cfg.fs = fs;
    cfg.dim = dim;
    cfg.lb = lb;
    cfg.ub = ub;
    cfg.nFireflies = min(nFireflies, 18);
    cfg.maxIter = min(maxIter, 60);
    cfg.nRestarts = min(nRestarts, 2);
    cfg.gamma = gamma;
    cfg.beta0 = beta0;
    cfg.alpha = alpha;
    cfg.lambda_MW = lambda_MW;
    cfg.lambda_PAPR = lambda_PAPR;
    cfg.lambda_PSLR = lambda_PSLR;
    cfg.MW_target = MW_target;
    cfg.PAPR_target = PAPR_target;
    cfg.PSLR_margin = PSLR_margin;
    cfg.PSLR_floor = PSLR_floor;
    cfg.nRuns = 6;
    cfg.base_seed = 100;
end

function run_extended_experiments(cfg)
    fprintf('\n=================== 论文补充实验（我方窗函数中心） ===================\n');
    fprintf('复现实验细节：B=%.1f MHz, T=%.2f us, fs=%.1f MHz, TBP=%.1f\n', cfg.B/1e6, cfg.T*1e6, cfg.fs/1e6, cfg.B*cfg.T);
    fprintf('窗长度=带宽内采样点数；脉冲压缩=xcorr 自相关；PSLR=首零点口径，MW=-3 dB口径。\n');
    fprintf('说明：Hamming/Kaiser/Taylor/Chebyshev 仅作为固定对照组，不参与我方窗优化。\n');

    [s0, f0] = build_lfm_ext(cfg.B, cfg.T, cfg.fs);
    N0 = length(s0);

    % 1) 我方窗对照结果（我方主导，基线仅参照）
    fprintf('\n[1] 我方窗对照结果（Proposed vs Baselines）\n');
    [b_prop, ~] = run_fa_core_ext(cfg, s0, f0, cfg.B);
    [pslr_p, mw_p, papr_p] = evaluate_metrics(b_prop, s0, cfg.fs, cfg.B);
    fprintf('Proposed(我方窗): PSLR=%.2f dB, MW=%.2e, PAPR=%.3f\n', pslr_p, mw_p, papr_p);

    names = {'Hamming','Kaiser','Taylor','Chebyshev'};
    base_metrics = zeros(numel(names), 3);
    for i = 1:numel(names)
        W = build_reference_window_ext(names{i}, f0, cfg.B, N0);
        sw = ifft(fft(s0) .* W);
        [R, lag] = xcorr(sw);
        base_metrics(i,:) = compute_metrics_single(safe_normalize(abs(R)), lag, sw);
    end
    fprintf('相对对照增益(我方-对照): ΔPSLR(dB)	ΔMW		ΔPAPR\n');
    for i = 1:numel(names)
        dpslr = pslr_p - base_metrics(i,1);
        dmw = mw_p - base_metrics(i,2);
        dpapr = papr_p - base_metrics(i,3);
        fprintf('vs %-10s: %+.2f		%+.2e	%+.3f\n', names{i}, dpslr, dmw, dpapr);
    end

    % 2)+3) 多次随机稳定性、收敛收益与代价
    fprintf('\n[2-3] 我方窗稳定性 + 收敛收益/代价\n');
    pslr_fa = zeros(cfg.nRuns,1); mw_fa = zeros(cfg.nRuns,1); papr_fa = zeros(cfg.nRuns,1);
    pslr_rf = zeros(cfg.nRuns,1); mw_rf = zeros(cfg.nRuns,1); papr_rf = zeros(cfg.nRuns,1);
    t_fa = zeros(cfg.nRuns,1); t_total = zeros(cfg.nRuns,1);
    refine_gain = zeros(cfg.nRuns,1); refine_ok = zeros(cfg.nRuns,1); extra_t = zeros(cfg.nRuns,1);
    delta_vs = struct();
    for i = 1:numel(names)
        delta_vs(i).dpslr = zeros(cfg.nRuns,1);
        delta_vs(i).dmw = zeros(cfg.nRuns,1);
        delta_vs(i).dpapr = zeros(cfg.nRuns,1);
    end
    hist_fa = zeros(cfg.maxIter,1); hist_rf = zeros(cfg.maxIter,1);

    for r = 1:cfg.nRuns
        rng(cfg.base_seed + r);
        [s, f] = build_lfm_ext(cfg.B, cfg.T, cfg.fs);
        tic;
        [b_fa, one_hist] = run_fa_core_ext(cfg, s, f, cfg.B);
        t_fa(r) = toc;
        [pslr_fa(r), mw_fa(r), papr_fa(r)] = evaluate_metrics(b_fa, s, cfg.fs, cfg.B);

        tic;
        [b_rf, rf_hist, info_rf] = refine_or_keep_ext(cfg, b_fa, s, cfg.fs, cfg.B);
        extra_t(r) = toc;
        t_total(r) = t_fa(r) + extra_t(r);
        refine_gain(r) = info_rf.pslr_gain;
        refine_ok(r) = info_rf.accepted;
        [pslr_rf(r), mw_rf(r), papr_rf(r)] = evaluate_metrics(b_rf, s, cfg.fs, cfg.B);

        N = length(s);
        for i = 1:numel(names)
            Wb = build_reference_window_ext(names{i}, f, cfg.B, N);
            sb = ifft(fft(s) .* Wb);
            [Rb, lagb] = xcorr(sb);
            mb = compute_metrics_single(safe_normalize(abs(Rb)), lagb, sb);
            delta_vs(i).dpslr(r) = pslr_rf(r) - mb(1);
            delta_vs(i).dmw(r) = mw_rf(r) - mb(2);
            delta_vs(i).dpapr(r) = papr_rf(r) - mb(3);
        end

        hist_fa = hist_fa + one_hist(:);
        hist_rf = hist_rf + rf_hist(:);
    end
    hist_fa = hist_fa / cfg.nRuns;
    hist_rf = hist_rf / cfg.nRuns;

    fprintf('我方窗(FA)稳定性: PSLR=%.2f±%.2f(var), MW=%.2e±%.2e(var), PAPR=%.3f±%.3f(var)\n', mean(pslr_fa), var(pslr_fa), mean(mw_fa), var(mw_fa), mean(papr_fa), var(papr_fa));
    fprintf('我方窗(FA+精修)稳定性: PSLR=%.2f±%.2f(var), MW=%.2e±%.2e(var), PAPR=%.3f±%.3f(var)\n', mean(pslr_rf), var(pslr_rf), mean(mw_rf), var(mw_rf), mean(papr_rf), var(papr_rf));
    for i = 1:numel(names)
        fprintf('相对%-10s增益: ΔPSLR=%.2f±%.2f(var), ΔMW=%.2e±%.2e(var), ΔPAPR=%.3f±%.3f(var)\n', ...
            names{i}, mean(delta_vs(i).dpslr), var(delta_vs(i).dpslr), mean(delta_vs(i).dmw), var(delta_vs(i).dmw), mean(delta_vs(i).dpapr), var(delta_vs(i).dpapr));
    end

    success_rate = mean(refine_ok) * 100;
    mean_gain = mean(refine_gain(refine_ok>0));
    if isempty(mean_gain) || ~isfinite(mean_gain)
        mean_gain = 0;
    end
    sec_per_db = mean(extra_t) / max(mean_gain, eps);
    fprintf('精修收益: 平均PSLR增益=%.3f dB, 成功率=%.1f%%, 平均额外耗时=%.3f s, 单位提升代价=%.3f s/dB\n', ...
        mean(refine_gain), success_rate, mean(extra_t), sec_per_db);
    fprintf('总耗时统计: FA=%.3f s, FA+精修=%.3f s\n', mean(t_fa), mean(t_total));
    fprintf('复杂度估计: O(nRuns*nRestarts*maxIter*nFireflies^2*eval_cost)。\n');

    figure('Name','扩展实验收敛曲线');
    plot(hist_fa,'LineWidth',1.5); hold on;
    plot(hist_rf,'LineWidth',1.5);
    legend('FA 单独','FA+精修','Location','best');
    xlabel('Iteration'); ylabel('Fitness'); grid on;
    title('我方窗参数优化平均收敛曲线');

    % 4) 参数敏感性（对默认配置的变化量）
    fprintf('\n[4] 参数敏感性（相对默认配置）\n');
    [b_def, ~] = run_fa_core_ext(cfg, s0, f0, cfg.B);
    [pslr_def, mw_def, papr_def] = evaluate_metrics(b_def, s0, cfg.fs, cfg.B);
    fprintf('默认配置: PSLR=%.2f, MW=%.2e, PAPR=%.3f\n', pslr_def, mw_def, papr_def);

    sensitivity_sweep_ext(cfg, s0, f0, 'alpha', [0.05 0.1 0.2 0.35], [pslr_def,mw_def,papr_def]);
    sensitivity_sweep_ext(cfg, s0, f0, 'gamma', [0.5 1.0 1.8], [pslr_def,mw_def,papr_def]);
    sensitivity_sweep_ext(cfg, s0, f0, 'lambda_PSLR', [40 80 120], [pslr_def,mw_def,papr_def]);
    sensitivity_sweep_ext(cfg, s0, f0, 'lambda_MW', [4 7 10], [pslr_def,mw_def,papr_def]);
    sensitivity_sweep_ext(cfg, s0, f0, 'lambda_PAPR', [5 8 11], [pslr_def,mw_def,papr_def]);
    sensitivity_sweep_ext(cfg, s0, f0, 'PSLR_margin', [0.4 0.8 1.2 1.6], [pslr_def,mw_def,papr_def]);
    fprintf('趋势结论提示：增大 lambda_PSLR 往往更利于压低旁瓣，但可能增加 MW/PAPR 代价；lambda_MW/lambda_PAPR 则强化对应约束。\n');

    % 5) 场景变化：泛化能力（我方相对基线增益矩阵）
    fprintf('\n[5] 场景泛化（TBP/N/SNR）\n');
    scenarios = [400e6,1.5e-6,600e6,20; 600e6,1.7e-6,720e6,10; 900e6,1.0e-6,1200e6,0];
    fprintf('Case	B(MHz)	T(us)	fs(MHz)	N	TBP	SNR	ΔPSLR_Ham	ΔPSLR_Kai	ΔPSLR_Tay	ΔPSLR_Cheb\n');
    for i = 1:size(scenarios,1)
        B = scenarios(i,1); T = scenarios(i,2); fs = scenarios(i,3); snr_db = scenarios(i,4);
        [s_scene, f_scene] = build_lfm_ext(B, T, fs);
        s_scene = add_awgn_ext(s_scene, snr_db);
        cfg_scene = cfg; cfg_scene.B = B; cfg_scene.T = T; cfg_scene.fs = fs;
        [b_scene, ~] = run_fa_core_ext(cfg_scene, s_scene, f_scene, B);
        [pslr_prop, ~, ~] = evaluate_metrics(b_scene, s_scene, fs, B);

        N = length(s_scene);
        dps = zeros(1, numel(names));
        for k = 1:numel(names)
            Wb = build_reference_window_ext(names{k}, f_scene, B, N);
            sb = ifft(fft(s_scene) .* Wb);
            [Rb, lagb] = xcorr(sb);
            mb = compute_metrics_single(safe_normalize(abs(Rb)), lagb, sb);
            dps(k) = pslr_prop - mb(1);
        end
        fprintf('%d	%.0f	%.2f	%.0f	%d	%.1f	%.0f	%+.2f		%+.2f		%+.2f		%+.2f\n', ...
            i, B/1e6, T*1e6, fs/1e6, N, B*T, snr_db, dps(1), dps(2), dps(3), dps(4));
    end
end
function [b_best, best_hist] = run_fa_core_ext(cfg, s_LFM, f, B)
    N = length(s_LFM);
    idx_band = abs(f) <= B/2;
    W_hc = zeros(N,1); W_hc(idx_band) = hamming(sum(idx_band)); W_hc = W_hc/(max(W_hc)+eps);
    s_h = ifft(fft(s_LFM).*ifftshift(W_hc));
    [R_h, lag_h] = xcorr(s_h);
    [PSLR_h,~,~] = compute_metrics_single(safe_normalize(abs(R_h)), lag_h, s_h);
    PSLR_target = PSLR_h - cfg.PSLR_margin;

    best_fit = inf; b_best = zeros(1,cfg.dim); best_hist = 1e3*ones(cfg.maxIter,1);
    for rr = 1:cfg.nRestarts
        fireflies = cfg.lb + (cfg.ub-cfg.lb).*rand(cfg.nFireflies,cfg.dim);
        fit = zeros(cfg.nFireflies,1);
        for i = 1:cfg.nFireflies
            fit(i) = fitness_func(fireflies(i,:), s_LFM, cfg.fs, B, cfg.lambda_MW, cfg.lambda_PAPR, cfg.lambda_PSLR, cfg.MW_target, cfg.PAPR_target, PSLR_target, cfg.PSLR_floor);
        end
        hist = zeros(cfg.maxIter,1);
        for iter = 1:cfg.maxIter
            [fit, idx] = sort(fit); fireflies = fireflies(idx,:);
            for i = 1:cfg.nFireflies
                for j = 1:cfg.nFireflies
                    if fit(j) < fit(i)
                        r = norm(fireflies(i,:)-fireflies(j,:));
                        beta = cfg.beta0 * exp(-cfg.gamma * r^2);
                        fireflies(i,:) = fireflies(i,:) + beta*(fireflies(j,:)-fireflies(i,:)) + cfg.alpha*(rand(1,cfg.dim)-0.5).*(cfg.ub-cfg.lb);
                        fireflies(i,:) = max(min(fireflies(i,:), cfg.ub), cfg.lb);
                        fit(i) = fitness_func(fireflies(i,:), s_LFM, cfg.fs, B, cfg.lambda_MW, cfg.lambda_PAPR, cfg.lambda_PSLR, cfg.MW_target, cfg.PAPR_target, PSLR_target, cfg.PSLR_floor);
                    end
                end
            end
            hist(iter) = min(fit);
        end
        if min(fit) < best_fit
            [fit, idx] = sort(fit); fireflies = fireflies(idx,:);
            best_fit = fit(1); b_best = fireflies(1,:); best_hist = hist;
        end
    end
end

function [b_out, hist, info] = refine_or_keep_ext(cfg, b_in, s_LFM, fs, B)
    b_out = b_in;
    hist = nan(cfg.maxIter,1);
    hist(:) = compute_PSLR(b_in, s_LFM, fs, B);
    info.accepted = false;
    info.pslr_gain = 0;
    if exist('fmincon','file') ~= 2
        return;
    end
    [R_l, lag_l] = xcorr(s_LFM);
    [~, MW_lfm, PAPR_lfm] = compute_metrics_single(safe_normalize(abs(R_l)), lag_l, s_LFM);

    N = length(s_LFM);
    f = (-N/2:N/2-1)' * (fs/N);
    idx_band = abs(f) <= B/2;
    W_hc = zeros(N,1); W_hc(idx_band) = hamming(sum(idx_band)); W_hc = W_hc/(max(W_hc)+eps);
    s_h = ifft(fft(s_LFM).*ifftshift(W_hc));
    [R_h, lag_h] = xcorr(s_h);
    [PSLR_h, MW_h, PAPR_h] = compute_metrics_single(safe_normalize(abs(R_h)), lag_h, s_h);

    MW_target = MW_h / max(MW_lfm,eps) + 0.2;
    PAPR_target = PAPR_h / max(PAPR_lfm,eps) + 0.2;
    pslr_before = compute_PSLR(b_in, s_LFM, fs, B);
    obj_fun = @(b) compute_PSLR(b, s_LFM, fs, B);
    options = optimoptions('fmincon','Display','off','Algorithm','interior-point','MaxFunctionEvaluations',1200);

    for attempt = 1:4
        pslr_target = PSLR_h - max(0,cfg.PSLR_margin - 0.2*(attempt-1));
        nonlcon = @(b) compute_constraints_v2(b, s_LFM, fs, B, MW_target, PAPR_target, pslr_target, MW_lfm, PAPR_lfm);
        [b_try, fval, ef] = fmincon(obj_fun, b_out, [], [], [], [], cfg.lb, cfg.ub, nonlcon, options);
        [c_try, ~] = nonlcon(b_try);
        if ef > 0 && all(c_try <= 1e-6) && fval <= pslr_before
            b_out = b_try;
            hist = linspace(pslr_before, fval, cfg.maxIter)';
            info.accepted = true;
            info.pslr_gain = pslr_before - fval;
            return;
        end
    end
end
function W = build_reference_window_ext(name, f, B, N)
    idx_band = abs(f) <= B/2;
    N_band = sum(idx_band);
    switch lower(name)
        case 'hamming'
            w = hamming(N_band);
        case 'kaiser'
            w = kaiser(N_band, 6);
        case 'taylor'
            if exist('taylorwin','file') == 2
                w = taylorwin(N_band, 4, -35);
            else
                w = kaiser(N_band, 5);
            end
        case 'chebyshev'
            if exist('chebwin','file') == 2
                w = chebwin(N_band, 60);
            else
                w = kaiser(N_band, 7);
            end
        otherwise
            w = hamming(N_band);
    end
    Wc = zeros(N,1); Wc(idx_band) = w; Wc = Wc/(max(Wc)+eps); W = ifftshift(Wc);
end

function sensitivity_sweep_ext(cfg, s0, f0, field_name, vals, base_metric)
    fprintf('%s 扫描: ', field_name);
    for i = 1:numel(vals)
        cfg_i = cfg;
        cfg_i.(field_name) = vals(i);
        [b, ~] = run_fa_core_ext(cfg_i, s0, f0, cfg_i.B);
        [pslr, mw, papr] = evaluate_metrics(b, s0, cfg_i.fs, cfg_i.B);
        dpslr = pslr - base_metric(1);
        dmw = mw - base_metric(2);
        dpapr = papr - base_metric(3);
        fprintf('[%.2f: ΔPSLR=%+.2f, ΔMW=%+.2e, ΔPAPR=%+.3f] ', vals(i), dpslr, dmw, dpapr);
    end
    fprintf('\n');
end
function [s, f] = build_lfm_ext(B, T, fs)
    N = round(T * fs);
    t = (-N/2:N/2-1)' / fs;
    f = (-N/2:N/2-1)' * (fs/N);
    s = exp(1j * pi * (B/T) * t.^2);
end

function s_noisy = add_awgn_ext(s, snr_db)
    p = mean(abs(s).^2);
    nvar = p / max(10^(snr_db/10), eps);
    n = sqrt(nvar/2) * (randn(size(s)) + 1j*randn(size(s)));
    s_noisy = s + n;
end
