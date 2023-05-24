% Define index_to_position function
function [x, y] = index_to_position(index, Xbeg, Xend, Ybeg, Yend, scan_points)
    Xstep = (Xend - Xbeg) / (scan_points - 1);
    Ystep = (Yend - Ybeg) / (scan_points - 1);

    x = floor(index / 5) * Xstep + Xbeg;
    y = mod(index, 5) * Ystep + Ybeg;
end
