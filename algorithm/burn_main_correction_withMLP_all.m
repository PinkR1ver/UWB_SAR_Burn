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

% position = [X_aper, Y_aper];

%{
%% Show Aperature stop position
figure
scatter(X_aper,Y_aper)
xlim([-0.05, 0.21])
ylim([-0.05, 0.21])
%}

load("trained_mlp_all.mat");

range_z = [0, 0.3];
range_x = [0, 0.16];
range_y = [0, 0.16];

tic
Image3D = GBP_3D_simu_correction(Nx, Ny, Nz, Xbeg, Xend, Ybeg, Yend, Zbeg, Zend, s, N_aper, X_aper, Y_aper, Z_aper, range_compan, c, fs, net, range_x, range_y, range_z);
toc

%% convert cube to vector to scatter 3D cube
x = repmat (1:100,1,100*100);
y = repmat (reshape (repmat (1:100,100,1),1,[]),1,100);
z = reshape (repmat (1:100,100*100,1),1,[]);
v = Image3D (:);

figure (1)
scatter3 (x,y,z,[],v,'filled')
xlabel ('X')
ylabel ('Y')
zlabel ('Z')
title ('3D view')

colormap (jet)
colorbar


%% plot slice
for i = 67:69
    figure
    set (gcf,'Position',[100 100 800 600])
    slice (Image3D,[],[],i) % 提取沿x,y,z方向的中间切片
    colormap (jet) % 选择颜色映射
    colorbar % 显示颜色条
    xlabel ('X')
    ylabel ('Y')
    zlabel ('Z')
    title ('slice')
    view(90, 90)
end