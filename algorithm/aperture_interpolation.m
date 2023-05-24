function new_sample = aperture_interpolation(sample, X_aper, Y_aper, interpolation_dis, Xbeg, Xend, Ybeg, Yend)

    new_sample = zeros(((Xend - Xbeg) / interpolation_dis + 1) * ((Yend - Ybeg) / interpolation_dis + 1), size(sample, 2));

    position = [X_aper, Y_aper];
    postisions_size = size(position, 1);

    flag = 1;
    for j = Ybeg:interpolation_dis:Yend

        for i = Xbeg:interpolation_dis:Xend

            %% find the nearest point

            record_dis = 0;
            min = 100;
            position_index = 0;
            record_x = 0;
            record_y = 0;

            for index = 1:postisions_size
                    dis_x = abs(i - position(index, 1));
                    dis_y = abs(j - position(index, 2));
                    dis = sqrt(dis_x ^ 2 + dis_y ^ 2);

                    if (dis < min)

                        min = dis;
                        position_index = index;
                        record_dis = dis;

                        record_x = i - position(index, 1);
                        record_y = j - position(index, 2);

                end


            end

            %% interpolation
            if (record_dis == 0)
                new_sample(flag, :) = sample(position_index, :);
                flag = flag + 1;
            else

                pair_index = position_index;
                neighbor_index_x = position_index;
                neighbor_index_y = position_index;

                if (record_x < 0 )
                    pair_index = pair_index - 1;
                    neighbor_index_x  = neighbor_index_x - 1;
                elseif (record_x > 0)
                    pair_index = pair_index + 1;
                    neighbor_index_x  = neighbor_index_x + 1;
                end
                
                if (record_y < 0 )
                    pair_index = pair_index - 5;
                    neighbor_index_y  = neighbor_index_y - 5;
                elseif (record_y > 0)
                    pair_index = pair_index + 5;
                    neighbor_index_y  = neighbor_index_y + 5;
                end
                
                %{
                dis1 = sqrt(abs(i - position(index, 1))^2 + abs(j - position(index, 2))^2);
                dis2 = sqrt(abs(i - position(pair_index, 1))^2 + abs(j - position(pair_index, 2))^2);
                dis3 = sqrt(abs(i - position(neighbor_index_x, 1))^2 + abs(j - position(neighbor_index_x, 2))^2);
                dis4 = sqrt(abs(i - position(neighbor_index_y, 1))^2 + abs(j - position(neighbor_index_y, 2))^2);
                %} 
                % liner interpolation

                dis1 = abs(i - position(index, 1))^2 + abs(j - position(index, 2))^2;
                dis2 = abs(i - position(pair_index, 1))^2 + abs(j - position(pair_index, 2))^2;
                dis3 = abs(i - position(neighbor_index_x, 1))^2 + abs(j - position(neighbor_index_x, 2))^2;
                dis4 = abs(i - position(neighbor_index_y, 1))^2 + abs(j - position(neighbor_index_y, 2))^2;

                part1 = dis1 / (dis1 + dis2);
                part2 = dis2 / (dis1 + dis2);
                part3 = dis3 / (dis3 + dis4);
                part4 = dis4 / (dis3 + dis4);

                signal1 = sample(index, :) * part1;
                signal2 = sample(pair_index, :) * part2;
                signal3 = sample(neighbor_index_x, :) * part3;
                signal4 = sample(neighbor_index_y, :) * part4;

                echo = signal1 + signal2 + signal3 + signal4;

                new_sample(flag, :) = echo;
                flag = flag + 1;

            end

        end

    end

end
