clear all;clc; close all;

%% parameters

c = 3e8; % light speed
range_compan = 0; % range compression factor
fs = 1 / (0.00051978 * 10 ^ -9); % burn_8080_2.1  interpolation=> time_8080_2_1_12

%% load data

load '../data/data_8080_2_1_33.mat'

s = data_8080_2_1_33;

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


%{

tic
Image3D = GBP_3D_simu(Nx, Ny, Nz, Xbeg, Xend, Ybeg, Yend, Zbeg, Zend, s, N_aper, X_aper, Y_aper, Z_aper, range_compan, c, fs);
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
figure
set (gcf,'Position',[100 100 800 600])
slice (Image3D,[],[],68) % 提取沿x,y,z方向的中间切片
colormap (jet) % 选择颜色映射
colorbar % 显示颜色条
xlabel ('X')
ylabel ('Y')
zlabel ('Z')
title ('slice')
view(90, 90)

%}




%% collect data to calculate correction parameter

x_grid = linspace(Xbeg, Xend, Nx);
y_grid = linspace(Ybeg,Yend, Ny);
z_grid = linspace(Zbeg, Zend, Nz);

length_s = size(s, 2);

imgI = zeros(Nx, Ny, Nz);

%% Calculate the image

tic

for n_aper = 1:N_aper

    imgTmp = zeros(Nx, Ny, Nz); % Everytime radar aperture moves, the image is reset to zero to generate a new image to add them together
    R = zeros(Nx, Ny, Nz);

    for nz = 1:Nz
        Z2t = (Z_aper(n_aper) - z_grid(nz))^2;
        for ny = 1:Ny 
            Y2t = (Y_aper(n_aper) - y_grid(ny))^2;
            for nx = 1:Nx
                X2t = (X_aper(n_aper, 1) - x_grid(nx))^2;
                
                dis = sqrt(X2t + Y2t + Z2t);
                
                R(nx, ny, nz) = 2 * dis;

                t = (R(nx, ny, nz) - range_compan) / c; % Calculate the time of flight
                index = floor(t * fs) + 1;

                if(index<length_s)
                    imgTmp(nx, ny, nz) = s(n_aper, index);
                end

            end
        end
    end

    imgI = imgI + imgTmp;

end

toc

correction_data_collect_dict = zeros(Nx * Ny * Nz, 7);

flag = 1;

n_aper = (N_aper + 1) / 2;

for nz = 1:Nz
    Z2t = (Z_aper(n_aper) - z_grid(nz))^2;
    for ny = 1:Ny 
        Y2t = (Y_aper(n_aper) - y_grid(ny))^2;
        for nx = 1:Nx
            X2t = (X_aper(n_aper, 1) - x_grid(nx))^2;
            
            dis = sqrt(X2t + Y2t + Z2t);

            azimuth = atan2d(Y2t, X2t); 
            elevation = asind(Z2t / dis);

            correction_data_collect_dict(flag,:) = [x_grid(nx), y_grid(ny), z_grid(nz), azimuth, elevation, dis, imgI(nx, ny, nz)];

            flag = flag + 1;

        end
    end
end

correction_data_collect_dict = array2table(correction_data_collect_dict, 'VariableNames', {'x', 'y', 'z', 'azimuth', 'elevation', 'dis', 'value'});

writetable(correction_data_collect_dict,'data_to_calculateCorrectionParam.csv')