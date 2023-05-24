clear all;clc; close all;

% Load the data from the .csv file
matrix = readmatrix('data_to_calculateCorrectionParam.csv');
matrix(matrix(:, end) == 0, :) = [];

%% find unique to help me to determine z value
unique_z = unique(matrix(:, 3));

correction_factor = [];

tic
for i = 1:length(unique_z)
    indices = find(matrix(:, 3) == unique_z(i));
    data = matrix(indices, :);
    % 进行校正并将结果存储在矩阵中


    if (unique_z(i) > 0.198)
        non_center_data = data;            
        non_center_data(non_center_data(:, 1) > 0.05 & non_center_data(:, 1) < 0.1 | non_center_data(:, 2) > 0.05 | non_center_data(:, 2) < 0.1, :) = [];

        center_data = data;
        center_data(center_data(:, 1) <= 0.05 | center_data(:, 1) >= 0.1 & center_data(:, 2) <= 0.05 & center_data(:, 2) >= 0.1, :) = [];

        reference_value_non_center = mean(non_center_data(:, end));
        correction_factor_tmp = non_center_data;
        correction_factor_tmp(:, end) = reference_value_non_center ./ non_center_data(:, end);
        correction_factor = cat(1, correction_factor, correction_factor_tmp);

        reference_value_center = max(center_data(:, end));
        correction_factor_tmp = center_data;
        correction_factor_tmp(:, end) = reference_value_center ./ center_data(:, end);
        correction_factor = cat(1, correction_factor, correction_factor_tmp);
    else
        reference_value = mean(data(:, end));
        
        correction_factor_tmp = data;
        correction_factor_tmp(:, end) = reference_value ./ data(:, end);
        
        correction_factor = cat(1, correction_factor, correction_factor_tmp);
    end
    
end

toc

    %{
    if (correction_factor_tmp(:, 1) > 0.05 && correction_factor_tmp(:, 1) < 0.1 && correction_factor_tmp(:, 2) > 0.05 && correction_factor_tmp(:, 2) < 0.1 && correction_factor_tmp(:, 3) > 0.2 && correction_factor_tmp(:, 3) < 0.3)
        
        reference_value = max(data(:, end));
        
        correction_factor_tmp = data;
        correction_factor_tmp(:, end) = reference_value ./ data(:, end);
        
        correction_factor_tmp(correction_factor_tmp(:, 1) < 0.05 | correction_factor_tmp(:, 1) > 0.1 | correction_factor_tmp(:, 2) < 0.05 | correction_factor_tmp(:, 2) > 0.1, :) = []; % delete non in surface data
        correction_factor = cat(1, correction_factor, correction_factor_tmp);

    elseif (correction_factor_tmp(:, 3) > 0.2 && correction_factor_tmp(:, 3) < 0.3)
        
        dismiss_data = data;
        
        dismiss_data(dismiss_data(:, 1) > 0.05 & dismiss_data(:, 1) < 0.1 | dismiss_data(:, 2) > 0.05 | dismiss_data(:, 2) < 0.1, :) = [];
        reference_value = max(dismiss_data(:, end));
    else
        reference_value = min(data(:, end));
    end
    %}

correction_factor = array2table(correction_factor, 'VariableNames', {'x', 'y', 'z', 'azimuth', 'elevation', 'dis', 'correction_param'});

writetable(correction_factor, 'correction_Param_all.csv')

data = readmatrix('correction_Param_all.csv');

% Split the data into input and output
inputs = data(:,4:6);
targets = data(:,end);

% Convert data to GPU arrays
% inputsGPU = gpuArray(inputs);
% targetsGPU = gpuArray(targets);

% Define the architecture of the MLP
hiddenLayerSize = [8,32,32,8];
net = feedforwardnet(hiddenLayerSize);

% Set up MLP for GPU training
% net = configure(net, inputsGPU, targetsGPU);

% Specify training options with GPU acceleration
net.trainFcn = 'trainscg';  % Specify the training algorithm

% Train the MLP
net = train(net, inputs', targets');

% Transfer the trained network back to the CPU
% net = gather(net);

% Test the MLP on a validation dataset
outputs = net(inputs');
mse = mean((targets' - outputs).^2);

% Save the trained MLP to a .mat file
save('trained_mlp_all.mat', 'net');

%{

% Define the neural network architecture
layers = [
    featureInputLayer(3)
    fullyConnectedLayer(8)
    reluLayer
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 1000, ...
    'MiniBatchSize', 256, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu');

% Set up the neural network for GPU training
net = trainNetwork(inputsGPU, targetsGPU, layers, options);

%}



