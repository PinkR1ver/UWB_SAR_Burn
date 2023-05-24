% Import data from CSV file
data = readmatrix('correction_Param.csv');

% Extract columns
X = data(:,4:6);
Y = data(:,end);

% Plot data
scatter3(X(:,1), X(:,2), Y);
xlabel('azimuth');
ylabel('elevation');
zlabel('dis');

%{
% Generate some sample data
x = gpuArray(linspace(0, 1, 100)');
y = gpuArray(sin(2*pi*x) + 0.1*randn(size(x)));

% Create a feedforward neural network with one hidden layer
net = feedforwardnet(10);

% Set the training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'UseGPU', true); % enable GPU acceleration

% Train the network
net = train(net, x, y, options);    
%}