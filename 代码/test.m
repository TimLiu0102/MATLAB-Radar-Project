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

%% 萤火虫算法参数
dim = 5;            % 勒让德多项式系数个数（使用 0,2,4,6,8 偶数阶，保证偶对称）
nFireflies = 30;    % 萤火虫数量
maxIter = 100;      % 最大迭代次数
gamma = 1;          % 光吸收系数
beta0 = 1;          % 初始吸引度
alpha = 0.2;        % 随机步长因子
lambda_MW = 10;     % 主瓣宽度惩罚系数
lambda_PAPR = 10;   % PAPR 惩罚系数
MW_target = 1.0;    % 目标主瓣宽度倍数（相对于 LFM 不加窗）
PAPR_target = 1.0;  % 目标 PAPR 倍数（相对于 LFM 不加窗）

% 初始化萤火虫位置（系数 b_n，范围可调）
lb = -5 * ones(1, dim);   % 下界
ub = 5 * ones(1, dim);    % 上界
fireflies = lb + (ub - lb) .* rand(nFireflies, dim);

%% 计算初始适应度
fitness = zeros(nFireflies, 1);
for i = 1:nFireflies
    fitness(i) = fitness_func(fireflies(i,:), s_LFM, fs, B, ...
                              lambda_MW, lambda_PAPR, MW_target, PAPR_target);
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
                                          lambda_MW, lambda_PAPR, MW_target, PAPR_target);
            end
        end
    end

    % 输出当前最优
    fprintf('Iter %d: best fitness = %.4f\n', iter, fitness(1));
end

%% 最终最优窗系数（萤火虫结果）
b_opt = fireflies(1, :);
disp('萤火虫优化得到的 Legendre 系数:');
disp(b_opt);

% 计算萤火虫结果的性能指标（用于后续约束设置）
N = length(s_LFM);
W_opt_temp = legendre_window(b_opt, fs, B, N);
s_w_opt_temp = ifft(fft(s_LFM) .* W_opt_temp);
[R_temp, lag_temp] = xcorr(s_w_opt_temp);
R_temp = abs(R_temp); R_temp = R_temp / max(R_temp);
[~, MW_opt, PAPR_opt] = compute_metrics_single(R_temp, lag_temp, s_w_opt_temp);

% 计算参考信号（原始 LFM）的主瓣宽度和 PAPR
[R_lfm_ref, lag_lfm_ref] = xcorr(s_LFM);
R_lfm_ref = abs(R_lfm_ref); R_lfm_ref = R_lfm_ref / max(R_lfm_ref);
[~, MW_lfm, PAPR_lfm] = compute_metrics_single(R_lfm_ref, lag_lfm_ref, s_LFM);

% 萤火虫结果的实际因子
MW_factor_opt = MW_opt / MW_lfm;
PAPR_factor_opt = PAPR_opt / PAPR_lfm;
fprintf('萤火虫结果：MW_factor = %.3f, PAPR_factor = %.3f\n', MW_factor_opt, PAPR_factor_opt);

%% ========================================================================
%  梯度精修（局部优化）- 改进版
% =========================================================================
disp('开始梯度精修...');

% 设置宽松的约束目标：允许因子比萤火虫结果略大（例如 1.1 倍）
relax_factor = 1.1;
MW_target_refined = max(1.0, relax_factor * MW_factor_opt);   % 至少为 1.0
PAPR_target_refined = max(1.0, relax_factor * PAPR_factor_opt);
fprintf('精修约束目标：MW <= %.3f, PAPR <= %.3f\n', MW_target_refined, PAPR_target_refined);

% 目标函数：仅 PSLR（最小化）
obj_fun = @(b) compute_PSLR(b, s_LFM, fs, B);

% 非线性约束函数（传入参考指标）
nonlcon = @(b) compute_constraints_v2(b, s_LFM, fs, B, MW_target_refined, PAPR_target_refined, MW_lfm, PAPR_lfm);

% fmincon 选项（使用 interior-point 算法，放宽容差）
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'interior-point', ...
    'OptimalityTolerance', 1e-4, 'ConstraintTolerance', 1e-4, ...
    'StepTolerance', 1e-8, 'MaxFunctionEvaluations', 1000);

% 调用 fmincon
[b_refined, fval_refined, exitflag, output] = fmincon(obj_fun, b_opt, [], [], [], [], lb, ub, nonlcon, options);

% 显示退出信息
fprintf('fmincon 退出标志: %d\n', exitflag);
disp(output.message);

% 计算精修前后的 PSLR
PSLR_before = compute_PSLR(b_opt, s_LFM, fs, B);
PSLR_after = fval_refined;
fprintf('精修前 PSLR = %.2f dB\n', PSLR_before);
fprintf('精修后 PSLR = %.2f dB\n', PSLR_after);

% 检查约束满足情况（精修后）
[c_after, ~] = nonlcon(b_refined);
fprintf('精修后约束违反：MW_factor - target = %.3e, PAPR_factor - target = %.3e\n', c_after(1), c_after(2));

if exitflag > 0
    disp('梯度精修成功完成。');
else
    disp('梯度精修可能未完全收敛，但将继续使用当前结果。');
end

% 更新最优系数
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
R_lfm = abs(R_lfm); R_lfm = R_lfm / max(R_lfm);
[R_opt, ~] = xcorr(s_w_opt);
R_opt = abs(R_opt); R_opt = R_opt / max(R_opt);
[R_hamming, ~] = xcorr(s_hamming);
R_hamming = abs(R_hamming); R_hamming = R_hamming / max(R_hamming);

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

function fitness = fitness_func(b, s_LFM, fs, B, lambda_MW, lambda_PAPR, MW_target, PAPR_target)
    try
        N = length(s_LFM);
        W = legendre_window(b, fs, B, N);
        s_w = ifft(fft(s_LFM) .* W);
        [R, lag] = xcorr(s_w);
        R = abs(R); R = R / max(R);
        [PSLR, MW, PAPR] = compute_metrics_single(R, lag, s_w);
        
        [R_lfm, lag_lfm] = xcorr(s_LFM);
        R_lfm = abs(R_lfm); R_lfm = R_lfm / max(R_lfm);
        [~, MW_lfm, PAPR_lfm] = compute_metrics_single(R_lfm, lag_lfm, s_LFM);
        
        MW_factor = MW / MW_lfm;
        PAPR_factor = PAPR / PAPR_lfm;
        
        penalty_MW = lambda_MW * max(0, MW_factor - MW_target);
        penalty_PAPR = lambda_PAPR * max(0, PAPR_factor - PAPR_target);
        
        fitness = 1*PSLR + 1.5*penalty_MW + 1*penalty_PAPR;
        
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
    R = abs(R); R = R / max(R);
    [pslr, ~, ~] = compute_metrics_single(R, lag, s_w);
end

function [c, ceq] = compute_constraints_v2(b, s_LFM, fs, B, MW_target, PAPR_target, MW_lfm, PAPR_lfm)
% 改进的约束函数：使用预先计算的参考指标
    N = length(s_LFM);
    W = legendre_window(b, fs, B, N);
    s_w = ifft(fft(s_LFM) .* W);
    [R, lag] = xcorr(s_w);
    R = abs(R); R = R / max(R);
    [~, MW, PAPR] = compute_metrics_single(R, lag, s_w);
    
    MW_factor = MW / MW_lfm;
    PAPR_factor = PAPR / PAPR_lfm;
    
    c = [MW_factor - MW_target; PAPR_factor - PAPR_target];
    ceq = [];
end
