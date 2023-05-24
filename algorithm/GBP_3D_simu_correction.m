function imgI = GBP_3D_simu(Nx, Ny, Nz, Xbeg, Xend, Ybeg, Yend, Zbeg, Zend, s, N_aper, X_aper, Y_aper, Z_aper, range_compan, c, fs, net, range_x, range_y, range_z)


    x_grid = linspace(Xbeg, Xend, Nx);
    y_grid = linspace(Ybeg,Yend, Ny);
    z_grid = linspace(Zbeg, Zend, Nz);

    length_s = size(s, 2);

    imgI = zeros(Nx, Ny, Nz);

    %% Calculate the image

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

    %% Normalization
    minI = min(min(min(imgI)));
    maxI = max(max(max(imgI)));
    imgI = (imgI - 0) / (maxI - 0);


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

                if (z_grid(nz) > range_z(1) && z_grid(nz) < range_z(2) && y_grid(ny) > range_y(1) && y_grid(ny) < range_y(2) && x_grid(nx) > range_x(1) && x_grid(nx) < range_x(2))
                    input_data = [azimuth;elevation;dis];
                    imgI(nx, ny, nz) = imgI(nx, ny, nz) * net(input_data);
                end

            end
        end
    end
    
end
