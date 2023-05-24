% Load .mat file
SAR_sample = load('../data/data_8080_2_1_25.mat');
SAR_sample = SAR_sample.data_8080_2_1_25;

% Create combinations of 4 numbers from 0 to 24
numbers = 0:24;
combinations = combnk(numbers, 4);

% Create the combined list of combinations and remaining numbers
comb_list = [];
for i = 1:size(combinations, 1)
    comb = combinations(i, :);
    for num = 0:24
        if ~ismember(num, comb)
            comb_list = [comb_list; [comb, num]];
        end
    end
end

data_size = size(comb_list, 1);

% Create arrays to store the data
ts1 = cell(data_size, 1);
ts2 = cell(data_size, 1);
ts3 = cell(data_size, 1);
ts4 = cell(data_size, 1);
ts_ans = cell(data_size, 1);
dis1 = cell(data_size, 1);
dis2 = cell(data_size, 1);
dis3 = cell(data_size, 1);
dis4 = cell(data_size, 1);

for index = 1:data_size
    ts1{index} = SAR_sample(comb_list(index, 1) + 1);
    ts2{index} = SAR_sample(comb_list(index, 2) + 1);
    ts3{index} = SAR_sample(comb_list(index, 3) + 1);
    ts4{index} = SAR_sample(comb_list(index, 4) + 1);
    ts_ans{index} = SAR_sample(comb_list(index, 5) + 1);

    [x1, y1] = index_to_position(comb_list(index, 1), 0, 0.16, 0, 0.16, 5);
    [x2, y2] = index_to_position(comb_list(index, 2), 0, 0.16, 0, 0.16, 5);
    [x3, y3] = index_to_position(comb_list(index, 3), 0, 0.16, 0, 0.16, 5);
    [x4, y4] = index_to_position(comb_list(index, 4), 0, 0.16, 0, 0.16, 5);
    [x_ans, y_ans] = index_to_position(comb_list(index, 5), 0, 0.16, 0, 0.16, 5);

    dis1{index} = [x1 - x_ans, y1 - y_ans];
    dis2{index} = [x2 - x_ans, y2 - y_ans];
    dis3{index} = [x3 - x_ans, y3 - y_ans];
    dis4{index} = [x4 - x_ans, y4 - y_ans];
end

% Create a table to store the data
df = table(ts1, ts2, ts3, ts4, ts_ans, dis1, dis2, dis3, dis4);

% Save the table to a CSV file
writetable(df, '../data/interploation_data.csv');