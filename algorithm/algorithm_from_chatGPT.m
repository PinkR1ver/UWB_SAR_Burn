clear all; clc; close all;

load '../data/data_8080_2_1_25.mat'

raw_data = data_8080_2_1_25;
index = 5; % one row have 5 points to scan
time_samples = size(raw_data, 2);

%% Reshape raw data

reshape_data = zeros(index, index, time_samples);

for i = 1:size(raw_data, 1)
    reshape_data(ceil(i/index), i - ((ceil(i/index) - 1) * 5), :) = raw_data(i, :);
end


raw_data = reshape_data;

% 参数设置
c = 3e8; % 光速，用于计算距离和时间
fc = 1 / (0.00051978 * 10 ^ -9); % 雷达的中心频率
bw = 1e9; % 雷达的带宽
lambda = c/fc; % 波长
range_resolution = c/(2*bw); % 距离分辨率

% 获取数据尺寸
[num_points, num_samples] = size(raw_data);

% 预分配三维图像数组
range_bins = ceil(range_resolution*num_samples);
image = zeros(range_bins, num_points, num_points);

% SAR图像重建
parfor i = 1:num_points
    for j = 1:num_points
        % 提取当前扫描点的时间序列数据
        data = raw_data(i, j, :);
        data = squeeze(data);
        
        % 脉冲压缩
        compressed_data = fft(data, range_bins);
        
        % SAR图像重建
        for k = 1:num_samples
            % 计算当前距离
            distance = (k-1)*range_resolution;
            
            % 复制压缩数据到对应位置
            image(:, i, j) = image(:, i, j) + compressed_data.*exp(1i*4*pi*distance/lambda)';
        end
    end
end


% 显示三维图像
figure;
for i = 1:num_points
    for j = 1:num_points
        subplot(5, 5, (i-1)*5 + j);
        imagesc(abs(squeeze(image(i, j, :))));
        title(sprintf('Point %d', i));
        xlabel('Range');
        ylabel('Cross Range');
    end
end

