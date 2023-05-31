for i = 1:gif_number
    Image3D = GBP_3D_simu_window(Nx, Ny, Nz, Xbeg, Xend, Ybeg, Yend, Zbeg, Zend, s, N_aper, X_aper, Y_aper, Z_aper, range_compan, c, fs, 0.004 * i);

    %% convert cube to vector to scatter 3D cube
    x = repmat (1:100,1,100*100);
    y = repmat (reshape (repmat (1:100,100,1),1,[]),1,100);
    z = reshape (repmat (1:100,100*100,1),1,[]);
    v = Image3D (:);

    data = Image3D(:,:,68);
    cmin = min(min(data));
    cmax = max(max(data));

    clims = [cmin cmax];

    imagesc(data, [cmin cmax]);
    axis equal;

    F=getframe(gcf);
    I=frame2im(F);
    [I,map]=rgb2ind(I,256);
    if pic_num == 1
        imwrite(I,map,'test.gif','gif','Loopcount',inf,'DelayTime',0.2);
    else
        imwrite(I,map,'test.gif','gif','WriteMode','append','DelayTime',0.2);
    end

    pic_num = pic_num + 1;
end

close all