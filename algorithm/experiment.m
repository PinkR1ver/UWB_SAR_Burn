clear all;clc; close all;

%% parameters

c = 3e8; % light speed
range_compan = 0; % range compression factor
fs = 1 / (0.00051978 * 10 ^ -9); % burn_8080_2.1  interpolation=> time_8080_2_1_12

%% load data

load '../data/data_8080_2_1_25.mat'

s = data_8080_2_1_25;

%% GBP 3D, Generalized Back Projection, A algorithm for SAR image formation
%% XYZ pixel

Nx = 100;
Ny = 100;
Nz = 100;

% Set real world coordinates

Xbeg = 0; Xend = 0.16;
Ybeg = 0; Yend = 0.16;
Zbeg = 0; Zend = 0.3; % This is because the objects surface is at 0.3m

%% Pixel distance

lx = (Xend - Xbeg) / (Nx - 1);
ly = (Yend - Ybeg) / (Ny - 1);
lz = (Zend - Zbeg) / (Nz - 1);

%% Set aperture parameters, including position and numbers

index_row_max = 5;
index_col_max = 5;

N_aper = index_row_max * index_col_max;

%% X_aperature store the position of each aperture
X_aper = zeros(N_aper, 1); % X aperture contain both emit and reveive
Y_aper = zeros(N_aper, 1);
Z_aper = zeros(N_aper, 1);

%% Calulate aperture positions

dx = 0.04; % pixel distance calulated from the data
dy = 0.04;

%% Every aperture position

for index_row = 1:index_row_max
    for index_col = 1:index_col_max
        index = (index_row - 1) * index_col_max + index_col;
        X_aper(index) = (index_col - 1) * dx;
        Y_aper(index) = (index_row - 1) * dy;
        Z_aper(index) = 0;
    end
end

s = aperture_interpolation(s, X_aper, Y_aper, 0.01, Xbeg, Xend, Ybeg, Yend);


for i = 1:size(new_sample, 1)
    plot(new_sample(i, :))
    hold on
end

figure 
for i = 1:size(s, 1)
    plot(s(i, :))
    hold on
end
