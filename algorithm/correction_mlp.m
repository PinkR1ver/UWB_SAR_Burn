clear all;clc; close all;

% Load the data from the .csv file
matrix = readmatrix('data_to_calculateCorrectionParam.csv');

%% find unique to help me to determine z value
unique_z = unique(matrix(:, 3));

correction_factor = [];

tic
for i = 1:length(unique_z)

    if ((unique_z(i) > 0.198) && (unique_z(i) < 0.207)) % use z as 0.2000, 0.203, 0.2061
        indices = find(matrix(:, 3) == unique_z(i));
        data = matrix(indices, :);
        % 进行校正并将结果存储在矩阵中

        max_value = max(data(:, end));

        correction_factor_tmp = data;
        correction_factor_tmp(:, end) = max_value ./ data(:, end);
        correction_factor_tmp(correction_factor_tmp(:, 1) < 0.05 | correction_factor_tmp(:, 1) > 0.1 | correction_factor_tmp(:, 2) < 0.05 | correction_factor_tmp(:, 2) > 0.1, :) = []; % delete non in surface data

        correction_factor = cat(1, correction_factor, correction_factor_tmp);
    end

end

toc

correction_factor = array2table(correction_factor, 'VariableNames', {'x', 'y', 'z', 'azimuth', 'elevation', 'dis', 'correction_param'});

writetable(correction_factor, 'correction_Param.csv')

data = readmatrix('correction_Param.csv');

% Split the data into input and output
inputs = data(:,4:6);
targets = data(:,end);

% Define the architecture of the MLP
hiddenLayerSize = [8,32,64,64,32,8];
net = feedforwardnet(hiddenLayerSize);

% Train the MLP
net.trainFcn = 'trainscg';  % Specify the training algorithm
net = train(net, inputs', targets');  % Transpose inputs and targets for compatibility

% Test the MLP on a validation dataset
outputs = net(inputs');
mse = mean((targets' - outputs).^2);

% Save the trained MLP to a .mat file
save('trained_mlp.mat', 'net');



