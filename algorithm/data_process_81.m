clear all;
clc ;close all;
%% read the data from file
% 8080_2_1
[time_8080_2_1_1, Ampli_8080_2_1_1] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-1.txt', '%s%s','headerlines',2);
[time_8080_2_1_2, Ampli_8080_2_1_2] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-2.txt', '%s%s','headerlines',2);
[time_8080_2_1_3, Ampli_8080_2_1_3] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-3.txt', '%s%s','headerlines',2);
[time_8080_2_1_4, Ampli_8080_2_1_4] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-2.txt', '%s%s','headerlines',2);
[time_8080_2_1_5, Ampli_8080_2_1_5] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-1.txt', '%s%s','headerlines',2);
[time_8080_2_1_6, Ampli_8080_2_1_6] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-6.txt', '%s%s','headerlines',2);
[time_8080_2_1_7, Ampli_8080_2_1_7] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-7.txt', '%s%s','headerlines',2);
[time_8080_2_1_8, Ampli_8080_2_1_8] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-8.txt', '%s%s','headerlines',2);
[time_8080_2_1_9, Ampli_8080_2_1_9] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-7.txt', '%s%s','headerlines',2);
[time_8080_2_1_10, Ampli_8080_2_1_10] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-6.txt', '%s%s','headerlines',2);
[time_8080_2_1_11, Ampli_8080_2_1_11] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-11.txt', '%s%s','headerlines',2);
[time_8080_2_1_12, Ampli_8080_2_1_12] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-12.txt', '%s%s','headerlines',2);
[time_8080_2_1_13, Ampli_8080_2_1_13] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-13.txt', '%s%s','headerlines',2);
[time_8080_2_1_14, Ampli_8080_2_1_14] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-12.txt', '%s%s','headerlines',2);
[time_8080_2_1_15, Ampli_8080_2_1_15] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-11.txt', '%s%s','headerlines',2);
[time_8080_2_1_16, Ampli_8080_2_1_16] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-6.txt', '%s%s','headerlines',2);
[time_8080_2_1_17, Ampli_8080_2_1_17] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-7.txt', '%s%s','headerlines',2);
[time_8080_2_1_18, Ampli_8080_2_1_18] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-8.txt', '%s%s','headerlines',2);
[time_8080_2_1_19, Ampli_8080_2_1_19] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-7.txt', '%s%s','headerlines',2);
[time_8080_2_1_20, Ampli_8080_2_1_20] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-6.txt', '%s%s','headerlines',2);
[time_8080_2_1_21, Ampli_8080_2_1_21] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-1.txt', '%s%s','headerlines',2);
[time_8080_2_1_22, Ampli_8080_2_1_22] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-2.txt', '%s%s','headerlines',2);
[time_8080_2_1_23, Ampli_8080_2_1_23] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-3.txt', '%s%s','headerlines',2);
[time_8080_2_1_24, Ampli_8080_2_1_24] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-2.txt', '%s%s','headerlines',2);
[time_8080_2_1_25, Ampli_8080_2_1_25] = textread('D:\CSTDE\data\burn-8080\burn-8080-2.1\burn-8080-2.1-1.txt', '%s%s','headerlines',2);

% 8080_normal
[time_noise, Ampli_noise] = textread('D:\CSTDE\data\vivaldi-data-bladder\vivaldi-antenna-noise(1-11GHz).txt','%s%s','headerlines',2);
[time_8080_normal_1, Ampli_8080_normal_1] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-1.txt', '%s%s','headerlines',2);
[time_8080_normal_2, Ampli_8080_normal_2] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-2.txt', '%s%s','headerlines',2);
[time_8080_normal_3, Ampli_8080_normal_3] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-3.txt', '%s%s','headerlines',2);
[time_8080_normal_4, Ampli_8080_normal_4] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-4.txt', '%s%s','headerlines',2);
[time_8080_normal_5, Ampli_8080_normal_5] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-5.txt', '%s%s','headerlines',2);
[time_8080_normal_6, Ampli_8080_normal_6] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-6.txt', '%s%s','headerlines',2);
[time_8080_normal_7, Ampli_8080_normal_7] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-7.txt', '%s%s','headerlines',2);
[time_8080_normal_8, Ampli_8080_normal_8] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-8.txt', '%s%s','headerlines',2);
[time_8080_normal_9, Ampli_8080_normal_9] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-9.txt', '%s%s','headerlines',2);
[time_8080_normal_10, Ampli_8080_normal_10] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-10.txt', '%s%s','headerlines',2);
[time_8080_normal_11, Ampli_8080_normal_11] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-11.txt', '%s%s','headerlines',2);
[time_8080_normal_12, Ampli_8080_normal_12] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-12.txt', '%s%s','headerlines',2);
[time_8080_normal_13, Ampli_8080_normal_13] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-13.txt', '%s%s','headerlines',2);
[time_8080_normal_14, Ampli_8080_normal_14] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-14.txt', '%s%s','headerlines',2);
[time_8080_normal_15, Ampli_8080_normal_15] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-15.txt', '%s%s','headerlines',2);
[time_8080_normal_16, Ampli_8080_normal_16] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-16.txt', '%s%s','headerlines',2);
[time_8080_normal_17, Ampli_8080_normal_17] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-17.txt', '%s%s','headerlines',2);
[time_8080_normal_18, Ampli_8080_normal_18] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-18.txt', '%s%s','headerlines',2);
[time_8080_normal_19, Ampli_8080_normal_19] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-19.txt', '%s%s','headerlines',2);
[time_8080_normal_20, Ampli_8080_normal_20] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-20.txt', '%s%s','headerlines',2);
[time_8080_normal_21, Ampli_8080_normal_21] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-21.txt', '%s%s','headerlines',2);
[time_8080_normal_22, Ampli_8080_normal_22] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-22.txt', '%s%s','headerlines',2);
[time_8080_normal_23, Ampli_8080_normal_23] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-23.txt', '%s%s','headerlines',2);
[time_8080_normal_24, Ampli_8080_normal_24] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-24.txt', '%s%s','headerlines',2);
[time_8080_normal_25, Ampli_8080_normal_25] = textread('D:\CSTDE\data\burn-8080\burn-8080-normal\burn-8080-normal-25.txt', '%s%s','headerlines',2);


% 8080_1_6
[time_8080_1_6_1, Ampli_8080_1_6_1] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-1.txt', '%s%s','headerlines',2);
[time_8080_1_6_2, Ampli_8080_1_6_2] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-14.txt', '%s%s','headerlines',2);
[time_8080_1_6_3, Ampli_8080_1_6_3] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-2.txt', '%s%s','headerlines',2);
[time_8080_1_6_4, Ampli_8080_1_6_4] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-15.txt', '%s%s','headerlines',2);
[time_8080_1_6_5, Ampli_8080_1_6_5] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-3.txt', '%s%s','headerlines',2);
[time_8080_1_6_6, Ampli_8080_1_6_6] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-3.txt', '%s%s','headerlines',2);
[time_8080_1_6_7, Ampli_8080_1_6_7] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-2.txt', '%s%s','headerlines',2);
[time_8080_1_6_8, Ampli_8080_1_6_8] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-14.txt', '%s%s','headerlines',2);
[time_8080_1_6_9, Ampli_8080_1_6_9] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-1.txt', '%s%s','headerlines',2);
[time_8080_1_6_10, Ampli_8080_1_6_10] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-20.txt', '%s%s','headerlines',2);
[time_8080_1_6_11, Ampli_8080_1_6_11] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-21.txt', '%s%s','headerlines',2);
[time_8080_1_6_12, Ampli_8080_1_6_12] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-22.txt', '%s%s','headerlines',2);
[time_8080_1_6_13, Ampli_8080_1_6_13] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-23.txt', '%s%s','headerlines',2);
[time_8080_1_6_14, Ampli_8080_1_6_14] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-24.txt', '%s%s','headerlines',2);
[time_8080_1_6_15, Ampli_8080_1_6_15] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-23.txt', '%s%s','headerlines',2);
[time_8080_1_6_16, Ampli_8080_1_6_16] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-22.txt', '%s%s','headerlines',2);
[time_8080_1_6_17, Ampli_8080_1_6_17] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-21.txt', '%s%s','headerlines',2);
[time_8080_1_6_18, Ampli_8080_1_6_18] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-20.txt', '%s%s','headerlines',2);
[time_8080_1_6_19, Ampli_8080_1_6_19] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-6.txt', '%s%s','headerlines',2);
[time_8080_1_6_20, Ampli_8080_1_6_20] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-16.txt', '%s%s','headerlines',2);
[time_8080_1_6_21, Ampli_8080_1_6_21] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-7.txt', '%s%s','headerlines',2);
[time_8080_1_6_22, Ampli_8080_1_6_22] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-17.txt', '%s%s','headerlines',2);
[time_8080_1_6_23, Ampli_8080_1_6_23] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-8.txt', '%s%s','headerlines',2);
[time_8080_1_6_24, Ampli_8080_1_6_24] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-17.txt', '%s%s','headerlines',2);
[time_8080_1_6_25, Ampli_8080_1_6_25] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-7.txt', '%s%s','headerlines',2);
[time_8080_1_6_26, Ampli_8080_1_6_26] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-16.txt', '%s%s','headerlines',2);
[time_8080_1_6_27, Ampli_8080_1_6_27] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-6.txt', '%s%s','headerlines',2);
[time_8080_1_6_28, Ampli_8080_1_6_28] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-25.txt', '%s%s','headerlines',2);
[time_8080_1_6_29, Ampli_8080_1_6_29] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-26.txt', '%s%s','headerlines',2);
[time_8080_1_6_30, Ampli_8080_1_6_30] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-27.txt', '%s%s','headerlines',2);
[time_8080_1_6_31, Ampli_8080_1_6_31] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-28.txt', '%s%s','headerlines',2);
[time_8080_1_6_32, Ampli_8080_1_6_32] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-29.txt', '%s%s','headerlines',2);
[time_8080_1_6_33, Ampli_8080_1_6_33] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-28.txt', '%s%s','headerlines',2);
[time_8080_1_6_34, Ampli_8080_1_6_34] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-27.txt', '%s%s','headerlines',2);
[time_8080_1_6_35, Ampli_8080_1_6_35] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-26.txt', '%s%s','headerlines',2);
[time_8080_1_6_36, Ampli_8080_1_6_36] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-25.txt', '%s%s','headerlines',2);
[time_8080_1_6_37, Ampli_8080_1_6_37] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-11.txt', '%s%s','headerlines',2);
[time_8080_1_6_38, Ampli_8080_1_6_38] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-18.txt', '%s%s','headerlines',2);
[time_8080_1_6_39, Ampli_8080_1_6_39] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-12.txt', '%s%s','headerlines',2);
[time_8080_1_6_40, Ampli_8080_1_6_40] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-19.txt', '%s%s','headerlines',2);
[time_8080_1_6_41, Ampli_8080_1_6_41] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-13.txt', '%s%s','headerlines',2);
[time_8080_1_6_42, Ampli_8080_1_6_42] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-19.txt', '%s%s','headerlines',2);
[time_8080_1_6_43, Ampli_8080_1_6_43] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-12.txt', '%s%s','headerlines',2);
[time_8080_1_6_44, Ampli_8080_1_6_44] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-18.txt', '%s%s','headerlines',2);
[time_8080_1_6_45, Ampli_8080_1_6_45] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-11.txt', '%s%s','headerlines',2);
[time_8080_1_6_46, Ampli_8080_1_6_46] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-25.txt', '%s%s','headerlines',2);
[time_8080_1_6_47, Ampli_8080_1_6_47] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-26.txt', '%s%s','headerlines',2);
[time_8080_1_6_48, Ampli_8080_1_6_48] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-27.txt', '%s%s','headerlines',2);
[time_8080_1_6_49, Ampli_8080_1_6_49] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-28.txt', '%s%s','headerlines',2);
[time_8080_1_6_50, Ampli_8080_1_6_50] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-29.txt', '%s%s','headerlines',2);
[time_8080_1_6_51, Ampli_8080_1_6_51] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-28.txt', '%s%s','headerlines',2);
[time_8080_1_6_52, Ampli_8080_1_6_52] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-27.txt', '%s%s','headerlines',2);
[time_8080_1_6_53, Ampli_8080_1_6_53] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-26.txt', '%s%s','headerlines',2);
[time_8080_1_6_54, Ampli_8080_1_6_54] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-25.txt', '%s%s','headerlines',2);
[time_8080_1_6_55, Ampli_8080_1_6_55] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-6.txt', '%s%s','headerlines',2);
[time_8080_1_6_56, Ampli_8080_1_6_56] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-16.txt', '%s%s','headerlines',2);
[time_8080_1_6_57, Ampli_8080_1_6_57] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-7.txt', '%s%s','headerlines',2);
[time_8080_1_6_58, Ampli_8080_1_6_58] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-17.txt', '%s%s','headerlines',2);
[time_8080_1_6_59, Ampli_8080_1_6_59] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-8.txt', '%s%s','headerlines',2);
[time_8080_1_6_60, Ampli_8080_1_6_60] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-17.txt', '%s%s','headerlines',2);
[time_8080_1_6_61, Ampli_8080_1_6_61] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-7.txt', '%s%s','headerlines',2);
[time_8080_1_6_62, Ampli_8080_1_6_62] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-16.txt', '%s%s','headerlines',2);
[time_8080_1_6_63, Ampli_8080_1_6_63] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-6.txt', '%s%s','headerlines',2);
[time_8080_1_6_64, Ampli_8080_1_6_64] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-20.txt', '%s%s','headerlines',2);
[time_8080_1_6_65, Ampli_8080_1_6_65] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-21.txt', '%s%s','headerlines',2);
[time_8080_1_6_66, Ampli_8080_1_6_66] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-22.txt', '%s%s','headerlines',2);
[time_8080_1_6_67, Ampli_8080_1_6_67] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-23.txt', '%s%s','headerlines',2);
[time_8080_1_6_68, Ampli_8080_1_6_68] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-24.txt', '%s%s','headerlines',2);
[time_8080_1_6_69, Ampli_8080_1_6_69] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-23.txt', '%s%s','headerlines',2);
[time_8080_1_6_70, Ampli_8080_1_6_70] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-22.txt', '%s%s','headerlines',2);
[time_8080_1_6_71, Ampli_8080_1_6_71] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-21.txt', '%s%s','headerlines',2);
[time_8080_1_6_72, Ampli_8080_1_6_72] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-20.txt', '%s%s','headerlines',2);
[time_8080_1_6_73, Ampli_8080_1_6_73] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-1.txt', '%s%s','headerlines',2);
[time_8080_1_6_74, Ampli_8080_1_6_74] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-14.txt', '%s%s','headerlines',2);
[time_8080_1_6_75, Ampli_8080_1_6_75] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-2.txt', '%s%s','headerlines',2);
[time_8080_1_6_76, Ampli_8080_1_6_76] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-15.txt', '%s%s','headerlines',2);
[time_8080_1_6_77, Ampli_8080_1_6_77] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-3.txt', '%s%s','headerlines',2);
[time_8080_1_6_78, Ampli_8080_1_6_78] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-15.txt', '%s%s','headerlines',2);
[time_8080_1_6_79, Ampli_8080_1_6_79] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-2.txt', '%s%s','headerlines',2);
[time_8080_1_6_80, Ampli_8080_1_6_80] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-14.txt', '%s%s','headerlines',2);
[time_8080_1_6_81, Ampli_8080_1_6_81] = textread('D:\CSTDE\data\burn-8080\burn-8080-1.6\burn-8080-1.6-1.txt', '%s%s','headerlines',2);


% 8080_0_6
[time_8080_0_6_1, Ampli_8080_0_6_1] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-1.txt', '%s%s','headerlines',2);
[time_8080_0_6_2, Ampli_8080_0_6_2] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-2.txt', '%s%s','headerlines',2);
[time_8080_0_6_3, Ampli_8080_0_6_3] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-3.txt', '%s%s','headerlines',2);
[time_8080_0_6_4, Ampli_8080_0_6_4] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-2.txt', '%s%s','headerlines',2);
[time_8080_0_6_5, Ampli_8080_0_6_5] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-1.txt', '%s%s','headerlines',2);
[time_8080_0_6_6, Ampli_8080_0_6_6] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-6.txt', '%s%s','headerlines',2);
[time_8080_0_6_7, Ampli_8080_0_6_7] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-7.txt', '%s%s','headerlines',2);
[time_8080_0_6_8, Ampli_8080_0_6_8] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-8.txt', '%s%s','headerlines',2);
[time_8080_0_6_9, Ampli_8080_0_6_9] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-7.txt', '%s%s','headerlines',2);
[time_8080_0_6_10, Ampli_8080_0_6_10] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-6.txt', '%s%s','headerlines',2);
[time_8080_0_6_11, Ampli_8080_0_6_11] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-11.txt', '%s%s','headerlines',2);
[time_8080_0_6_12, Ampli_8080_0_6_12] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-12.txt', '%s%s','headerlines',2);
[time_8080_0_6_13, Ampli_8080_0_6_13] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-13.txt', '%s%s','headerlines',2);
[time_8080_0_6_14, Ampli_8080_0_6_14] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-12.txt', '%s%s','headerlines',2);
[time_8080_0_6_15, Ampli_8080_0_6_15] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-11.txt', '%s%s','headerlines',2);
[time_8080_0_6_16, Ampli_8080_0_6_16] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-6.txt', '%s%s','headerlines',2);
[time_8080_0_6_17, Ampli_8080_0_6_17] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-7.txt', '%s%s','headerlines',2);
[time_8080_0_6_18, Ampli_8080_0_6_18] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-8.txt', '%s%s','headerlines',2);
[time_8080_0_6_19, Ampli_8080_0_6_19] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-7.txt', '%s%s','headerlines',2);
[time_8080_0_6_20, Ampli_8080_0_6_20] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-6.txt', '%s%s','headerlines',2);
[time_8080_0_6_21, Ampli_8080_0_6_21] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-1.txt', '%s%s','headerlines',2);
[time_8080_0_6_22, Ampli_8080_0_6_22] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-2.txt', '%s%s','headerlines',2);
[time_8080_0_6_23, Ampli_8080_0_6_23] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-3.txt', '%s%s','headerlines',2);
[time_8080_0_6_24, Ampli_8080_0_6_24] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-2.txt', '%s%s','headerlines',2);
[time_8080_0_6_25, Ampli_8080_0_6_25] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.6\burn-8080-0.6-1.txt', '%s%s','headerlines',2);

% 8080_0_02
[time_8080_0_02_1, Ampli_8080_0_02_1] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-1.txt', '%s%s','headerlines',2);
[time_8080_0_02_2, Ampli_8080_0_02_2] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-2.txt', '%s%s','headerlines',2);
[time_8080_0_02_3, Ampli_8080_0_02_3] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-3.txt', '%s%s','headerlines',2);
[time_8080_0_02_4, Ampli_8080_0_02_4] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-2.txt', '%s%s','headerlines',2);
[time_8080_0_02_5, Ampli_8080_0_02_5] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-1.txt', '%s%s','headerlines',2);
[time_8080_0_02_6, Ampli_8080_0_02_6] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-6.txt', '%s%s','headerlines',2);
[time_8080_0_02_7, Ampli_8080_0_02_7] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-7.txt', '%s%s','headerlines',2);
[time_8080_0_02_8, Ampli_8080_0_02_8] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-8.txt', '%s%s','headerlines',2);
[time_8080_0_02_9, Ampli_8080_0_02_9] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-7.txt', '%s%s','headerlines',2);
[time_8080_0_02_10, Ampli_8080_0_02_10] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-6.txt', '%s%s','headerlines',2);
[time_8080_0_02_11, Ampli_8080_0_02_11] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-11.txt', '%s%s','headerlines',2);
[time_8080_0_02_12, Ampli_8080_0_02_12] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-12.txt', '%s%s','headerlines',2);
[time_8080_0_02_13, Ampli_8080_0_02_13] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-13.txt', '%s%s','headerlines',2);
[time_8080_0_02_14, Ampli_8080_0_02_14] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-12.txt', '%s%s','headerlines',2);
[time_8080_0_02_15, Ampli_8080_0_02_15] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-11.txt', '%s%s','headerlines',2);
[time_8080_0_02_16, Ampli_8080_0_02_16] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-6.txt', '%s%s','headerlines',2);
[time_8080_0_02_17, Ampli_8080_0_02_17] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-7.txt', '%s%s','headerlines',2);
[time_8080_0_02_18, Ampli_8080_0_02_18] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-8.txt', '%s%s','headerlines',2);
[time_8080_0_02_19, Ampli_8080_0_02_19] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-7.txt', '%s%s','headerlines',2);
[time_8080_0_02_20, Ampli_8080_0_02_20] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-6.txt', '%s%s','headerlines',2);
[time_8080_0_02_21, Ampli_8080_0_02_21] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-1.txt', '%s%s','headerlines',2);
[time_8080_0_02_22, Ampli_8080_0_02_22] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-2.txt', '%s%s','headerlines',2);
[time_8080_0_02_23, Ampli_8080_0_02_23] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-3.txt', '%s%s','headerlines',2);
[time_8080_0_02_24, Ampli_8080_0_02_24] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-2.txt', '%s%s','headerlines',2);
[time_8080_0_02_25, Ampli_8080_0_02_25] = textread('D:\CSTDE\data\burn-8080\burn-8080-0.02\burn-8080-0.02-1.txt', '%s%s','headerlines',2);


%% convert data into double
time_noise = cell2double(time_noise);
Ampli_noise = cell2double(Ampli_noise);

% 8080_normal
for i = 1:25
    eval(['name = time_8080_normal_',num2str(i),';']);
    name = cell2double(name);
    eval(['time_8080_normal_',num2str(i),'=name;']);
end

for i = 1:25
    eval(['name = Ampli_8080_normal_',num2str(i),';']);
    name = cell2double(name);
    eval(['Ampli_8080_normal_',num2str(i),'=name;']);
end

% 8080_2_1
for i = 1:25
    eval(['name = time_8080_2_1_',num2str(i),';']);
    name = cell2double(name);
    eval(['time_8080_2_1_',num2str(i),'=name;']);
end

for i = 1:25
    eval(['name = Ampli_8080_2_1_',num2str(i),';']);
    name = cell2double(name);
    eval(['Ampli_8080_2_1_',num2str(i),'=name;']);
end

% 8080_1_6
for i = 1:81
    eval(['name = time_8080_1_6_',num2str(i),';']);
    name = cell2double(name);
    eval(['time_8080_1_6_',num2str(i),'=name;']);
end

for i = 1:81
    eval(['name = Ampli_8080_1_6_',num2str(i),';']);
    name = cell2double(name);
    eval(['Ampli_8080_1_6_',num2str(i),'=name;']);
end

% 8080_0_6
for i = 1:25
    eval(['name = time_8080_0_6_',num2str(i),';']);
    name = cell2double(name);
    eval(['time_8080_0_6_',num2str(i),'=name;']);
end

for i = 1:25
    eval(['name = Ampli_8080_0_6_',num2str(i),';']);
    name = cell2double(name);
    eval(['Ampli_8080_0_6_',num2str(i),'=name;']);
end

% 8080_0_02
for i = 1:25
    eval(['name = time_8080_0_02_',num2str(i),';']);
    name = cell2double(name);
    eval(['time_8080_0_02_',num2str(i),'=name;']);
end

for i = 1:25
    eval(['name = Ampli_8080_0_02_',num2str(i),';']);
    name = cell2double(name);
    eval(['Ampli_8080_0_02_',num2str(i),'=name;']);
end



%% interpolation
t_interval = (time_8080_2_1_11(end) - time_8080_2_1_11(1)) / length(time_8080_2_1_11); %插值后的时间间隔
t_unified = 0 : t_interval : t_interval * (length(time_8080_2_1_11)-1); %插值后的时间序列

Ampli_unified = interp1(time_noise, Ampli_noise, t_unified, 'spline'); %插值后的幅值

% 8080_normal
for i = 1:25
    eval(['rep1 = time_8080_normal_',num2str(i),';']);
    eval(['rep2 = Ampli_8080_normal_',num2str(i),';']);
    rep3 = interpolation(rep1,rep2,t_unified);
    eval(['Ampli_8080_normal_',num2str(i),'_unified','= rep3;'])
end

% 8080_2_1
for i = 1:25
    eval(['rep4 = time_8080_2_1_',num2str(i),';']);
    eval(['rep5 = Ampli_8080_2_1_',num2str(i),';']);
    rep6 = interpolation(rep4, rep5, t_unified);
    eval(['Ampli_8080_2_1_',num2str(i),'_unified','= rep6;'])
end

% 8080_1_6
for i = 1:81
    eval(['rep14 = time_8080_1_6_',num2str(i),';']);
    eval(['rep15 = Ampli_8080_1_6_',num2str(i),';']);
    rep16 = interpolation(rep14, rep15, t_unified);
    eval(['Ampli_8080_1_6_',num2str(i),'_unified','= rep16;'])
end

% 8080_0_6
for i = 1:25
    eval(['rep11 = time_8080_0_6_',num2str(i),';']);
    eval(['rep12 = Ampli_8080_0_6_',num2str(i),';']);
    rep13 = interpolation(rep11, rep12, t_unified);
    eval(['Ampli_8080_0_6_',num2str(i),'_unified','= rep13;'])
end

% 8080_0_02
for i = 1:25
    eval(['rep77 = time_8080_0_02_',num2str(i),';']);
    eval(['rep88 = Ampli_8080_0_02_',num2str(i),';']);
    rep99 = interpolation(rep77, rep88, t_unified);
    eval(['Ampli_8080_0_02_',num2str(i),'_unified','= rep99;'])
end


%% denoise
for i = 1:25
    eval(['rep7 = Ampli_8080_normal_',num2str(i),'_unified;']);
    rep8 = denoise(rep7, Ampli_unified);
    eval(['Ampli_8080_normal_',num2str(i),'_pure','= rep8;'])  
end

for i = 1:25
    eval(['rep9 = Ampli_8080_2_1_',num2str(i),'_unified;']);
    rep10 = denoise(rep9, Ampli_unified);
    eval(['Ampli_8080_2_1_',num2str(i),'_pure','= rep10;'])
end

for i = 1:25
    eval(['rep17 = Ampli_8080_0_6_',num2str(i),'_unified;']);
    rep18 = denoise(rep17, Ampli_unified);
    eval(['Ampli_8080_0_6_',num2str(i),'_pure','= rep18;'])  
end

for i = 1:81
    eval(['rep19 = Ampli_8080_1_6_',num2str(i),'_unified;']);
    rep20 = denoise(rep19, Ampli_unified);
    eval(['Ampli_8080_1_6_',num2str(i),'_pure','= rep20;'])
end

for i = 1:25
    eval(['rep29 = Ampli_8080_0_02_',num2str(i),'_unified;']);
    rep30 = denoise(rep29, Ampli_unified);
    eval(['Ampli_8080_0_02_',num2str(i),'_pure','= rep30;'])
end

s_index_start = 1928;%1371;% 788为峰值/1379为实际的峰值  1215
s_index_end = s_index_start + 3000; %


%% 调整峰值信号 pure为去噪后的信号，pure2为调整后的信号
% 以8080_normal的峰值信号为基准，故其不需要调整；其余信号调成跟这个一样
% 8080_zerosetting
Ampli_8080_normal_1_pure2 = zeros(1,4975);
Ampli_8080_normal_1_residue1 = Ampli_8080_normal_1_pure(4107:4241);
Ampli_8080_normal_1_pure2(4107:4241)= Ampli_8080_normal_1_residue1;
Ampli_8080_normal_5_pure2 = Ampli_8080_normal_1_pure2;
Ampli_8080_normal_21_pure2 = Ampli_8080_normal_1_pure2;
Ampli_8080_normal_25_pure2 = Ampli_8080_normal_1_pure2;

Ampli_8080_normal_2_pure2 = zeros(1,4975);
Ampli_8080_normal_2_residue1 = Ampli_8080_normal_2_pure(4054:4191);
Ampli_8080_normal_2_pure2(4054:4191) = Ampli_8080_normal_2_residue1;
Ampli_8080_normal_4_pure2 = Ampli_8080_normal_2_pure2;
Ampli_8080_normal_22_pure2 = Ampli_8080_normal_2_pure2;
Ampli_8080_normal_24_pure2 = Ampli_8080_normal_2_pure2;

Ampli_8080_normal_3_pure2 = zeros(1,4975);
Ampli_8080_normal_3_residue1 = Ampli_8080_normal_3_pure(4042:4175);
Ampli_8080_normal_3_pure2(4042:4175) = Ampli_8080_normal_3_residue1;
Ampli_8080_normal_23_pure2 = Ampli_8080_normal_3_pure2;

Ampli_8080_normal_6_pure2 = zeros(1,4975);
Ampli_8080_normal_6_residue1 = Ampli_8080_normal_6_pure(4055:4206);
Ampli_8080_normal_6_pure2(4055:4206) = Ampli_8080_normal_6_residue1;
Ampli_8080_normal_10_pure2 = Ampli_8080_normal_6_pure2;
Ampli_8080_normal_16_pure2 = Ampli_8080_normal_6_pure2;
Ampli_8080_normal_20_pure2 = Ampli_8080_normal_6_pure2;

Ampli_8080_normal_7_pure2 = zeros(1,4975);
Ampli_8080_normal_7_residue1 = Ampli_8080_normal_7_pure(4002:4140);
Ampli_8080_normal_7_pure2(4002:4140) = Ampli_8080_normal_7_residue1;
Ampli_8080_normal_9_pure2 = Ampli_8080_normal_7_pure2;
Ampli_8080_normal_17_pure2 = Ampli_8080_normal_7_pure2;
Ampli_8080_normal_19_pure2 = Ampli_8080_normal_7_pure2;

Ampli_8080_normal_8_pure2 = zeros(1,4975);
Ampli_8080_normal_8_residue1 = Ampli_8080_normal_8_pure(3990:4127);
Ampli_8080_normal_8_pure2(3990:4127) = Ampli_8080_normal_8_residue1;
Ampli_8080_normal_18_pure2 = Ampli_8080_normal_8_pure2;

Ampli_8080_normal_11_pure2 = zeros(1,4975);
Ampli_8080_normal_11_residue1 = Ampli_8080_normal_11_pure(4031:4165);
Ampli_8080_normal_11_pure2(4031:4165) = Ampli_8080_normal_11_residue1;
Ampli_8080_normal_15_pure2 = Ampli_8080_normal_11_pure2;

Ampli_8080_normal_12_pure2 = zeros(1,4975);
Ampli_8080_normal_12_residue1 = Ampli_8080_normal_12_pure(3982:4107);
Ampli_8080_normal_12_pure2(3982:4107) = Ampli_8080_normal_12_residue1;
Ampli_8080_normal_14_pure2 = Ampli_8080_normal_12_pure2;

Ampli_8080_normal_13_pure2 = zeros(1,4975);
Ampli_8080_normal_13_residue1 = Ampli_8080_normal_13_pure(3971:4094);
Ampli_8080_normal_13_pure2(3971:4094) = Ampli_8080_normal_13_residue1;


% 8080_0_02_zerosetting
Ampli_8080_0_02_1_pure2 = zeros(1,4975);
Ampli_8080_0_02_1_residue1 = Ampli_8080_0_02_1_pure(4108:4242);
Ampli_8080_0_02_1_pure2(4108:4242)=Ampli_8080_0_02_1_residue1;
Ampli_8080_0_02_5_pure2 = Ampli_8080_0_02_1_pure2;
Ampli_8080_0_02_21_pure2 = Ampli_8080_0_02_1_pure2;
Ampli_8080_0_02_25_pure2 = Ampli_8080_0_02_1_pure2;

Ampli_8080_0_02_2_pure2 = zeros(1,4975);
Ampli_8080_0_02_2_residue1 = Ampli_8080_0_02_2_pure(4054:4192);
Ampli_8080_0_02_2_pure2(4054:4192)=Ampli_8080_0_02_2_residue1;
Ampli_8080_0_02_4_pure2 = Ampli_8080_0_02_2_pure2;
Ampli_8080_0_02_22_pure2 = Ampli_8080_0_02_2_pure2;
Ampli_8080_0_02_24_pure2 = Ampli_8080_0_02_2_pure2;

Ampli_8080_0_02_3_pure2 = zeros(1,4975);
Ampli_8080_0_02_3_residue1 = Ampli_8080_0_02_3_pure(4042:4174);
Ampli_8080_0_02_3_pure2(4042:4174)=Ampli_8080_0_02_3_residue1;
Ampli_8080_0_02_23_pure2 = Ampli_8080_0_02_3_pure2;

Ampli_8080_0_02_6_pure2 = zeros(1,4975);
Ampli_8080_0_02_6_residue1 = Ampli_8080_0_02_6_pure(4055:4208);
Ampli_8080_0_02_6_pure2(4055:4208)=Ampli_8080_0_02_6_residue1;
Ampli_8080_0_02_10_pure2 = Ampli_8080_0_02_6_pure2;
Ampli_8080_0_02_16_pure2 = Ampli_8080_0_02_6_pure2;
Ampli_8080_0_02_20_pure2 = Ampli_8080_0_02_6_pure2;

Ampli_8080_0_02_7_pure2 = zeros(1,4975);
Ampli_8080_0_02_7_residue1 = Ampli_8080_0_02_7_pure(4002:4141);
Ampli_8080_0_02_7_pure2(4002:4141)=Ampli_8080_0_02_7_residue1;
Ampli_8080_0_02_9_pure2 = Ampli_8080_0_02_7_pure2;
Ampli_8080_0_02_17_pure2 = Ampli_8080_0_02_7_pure2;
Ampli_8080_0_02_19_pure2 = Ampli_8080_0_02_7_pure2;

Ampli_8080_0_02_8_pure2 = zeros(1,4975);
Ampli_8080_0_02_8_residue1 = Ampli_8080_0_02_8_pure(3991:4127);
Ampli_8080_0_02_8_pure2(3991:4127)=Ampli_8080_0_02_8_residue1;
Ampli_8080_0_02_18_pure2 = Ampli_8080_0_02_8_pure2;

Ampli_8080_0_02_11_pure2 = zeros(1,4975);
Ampli_8080_0_02_11_residue1 = Ampli_8080_0_02_11_pure(4032:4165);
Ampli_8080_0_02_11_pure2(4032:4165)=Ampli_8080_0_02_11_residue1;
Ampli_8080_0_02_15_pure2 = Ampli_8080_0_02_11_pure2;

Ampli_8080_0_02_12_pure2 = zeros(1,4975);
Ampli_8080_0_02_12_residue1 = Ampli_8080_0_02_12_pure(3982:4107);
Ampli_8080_0_02_12_pure2(3982:4107)=Ampli_8080_0_02_12_residue1;
Ampli_8080_0_02_14_pure2 = Ampli_8080_0_02_12_pure2;

Ampli_8080_0_02_13_pure2 = zeros(1,4975);
Ampli_8080_0_02_13_residue1 = Ampli_8080_0_02_13_pure(3971:4094);
Ampli_8080_0_02_13_pure2(3971:4094)=Ampli_8080_0_02_13_residue1;

% 8080_0_6_offset
Ampli_8080_0_6_1_pure2 = zeros(1,4975);
Ampli_8080_0_6_1_residue1 = Ampli_8080_0_6_1_pure(4110:4243);
Ampli_8080_0_6_1_pure2(4106:4239)=Ampli_8080_0_6_1_residue1;
Ampli_8080_0_6_5_pure2 = Ampli_8080_0_6_1_pure2;
Ampli_8080_0_6_21_pure2 = Ampli_8080_0_6_1_pure2;
Ampli_8080_0_6_25_pure2 = Ampli_8080_0_6_1_pure2;

Ampli_8080_0_6_2_pure2 = zeros(1,4975);
Ampli_8080_0_6_2_residue1 = Ampli_8080_0_6_2_pure(4057:4194);
Ampli_8080_0_6_2_pure2(4053:4190)=Ampli_8080_0_6_2_residue1;
Ampli_8080_0_6_4_pure2 = Ampli_8080_0_6_2_pure2;
Ampli_8080_0_6_22_pure2 = Ampli_8080_0_6_2_pure2;
Ampli_8080_0_6_24_pure2 = Ampli_8080_0_6_2_pure2;

Ampli_8080_0_6_3_pure2 = zeros(1,4975);
Ampli_8080_0_6_3_residue1 = Ampli_8080_0_6_3_pure(4044:4178);
Ampli_8080_0_6_3_pure2(4040:4174)=Ampli_8080_0_6_3_residue1;
Ampli_8080_0_6_23_pure2 = Ampli_8080_0_6_3_pure2;

Ampli_8080_0_6_6_pure2 = zeros(1,4975);
Ampli_8080_0_6_6_residue1 = Ampli_8080_0_6_6_pure(4058:4210);
Ampli_8080_0_6_6_pure2(4054:4206)=Ampli_8080_0_6_6_residue1;
Ampli_8080_0_6_10_pure2 = Ampli_8080_0_6_6_pure2;
Ampli_8080_0_6_16_pure2 = Ampli_8080_0_6_6_pure2;
Ampli_8080_0_6_20_pure2 = Ampli_8080_0_6_6_pure2;

Ampli_8080_0_6_7_pure2 = zeros(1,4975);
Ampli_8080_0_6_7_residue1 = Ampli_8080_0_6_7_pure(4006:4143);
Ampli_8080_0_6_7_pure2(4002:4139)=Ampli_8080_0_6_7_residue1;
Ampli_8080_0_6_9_pure2 = Ampli_8080_0_6_7_pure2;
Ampli_8080_0_6_17_pure2 = Ampli_8080_0_6_7_pure2;
Ampli_8080_0_6_19_pure2 = Ampli_8080_0_6_7_pure2;

Ampli_8080_0_6_8_pure2 = zeros(1,4975);
Ampli_8080_0_6_8_residue1 = Ampli_8080_0_6_8_pure(3994:4131);
Ampli_8080_0_6_8_pure2(3990:4127)=Ampli_8080_0_6_8_residue1;
Ampli_8080_0_6_18_pure2 = Ampli_8080_0_6_8_pure2;

Ampli_8080_0_6_11_pure2 = zeros(1,4975);
Ampli_8080_0_6_11_residue1 = Ampli_8080_0_6_11_pure(4035:4168);
Ampli_8080_0_6_11_pure2(4031:4164)=Ampli_8080_0_6_11_residue1;
Ampli_8080_0_6_15_pure2 = Ampli_8080_0_6_11_pure2;

Ampli_8080_0_6_12_pure2 = zeros(1,4975);
Ampli_8080_0_6_12_residue1 = Ampli_8080_0_6_12_pure(3985:4111);
Ampli_8080_0_6_12_pure2(3981:4107)=Ampli_8080_0_6_12_residue1;
Ampli_8080_0_6_14_pure2 = Ampli_8080_0_6_12_pure2;

Ampli_8080_0_6_13_pure2 = zeros(1,4975);
Ampli_8080_0_6_13_residue1 = Ampli_8080_0_6_13_pure(3974:4098);
Ampli_8080_0_6_13_pure2(3970:4094)=Ampli_8080_0_6_13_residue1;

% 8080_1_6_offset
Ampli_8080_1_6_1_pure2 = zeros(1,4975);
Ampli_8080_1_6_1_residue1 = Ampli_8080_1_6_1_pure(4113:4245);
Ampli_8080_1_6_1_pure2(4102:4234)=Ampli_8080_1_6_1_residue1;
Ampli_8080_1_6_9_pure2 = Ampli_8080_1_6_1_pure2;
Ampli_8080_1_6_73_pure2 = Ampli_8080_1_6_1_pure2;
Ampli_8080_1_6_81_pure2 = Ampli_8080_1_6_1_pure2;

Ampli_8080_1_6_2_pure2 = zeros(1,4975);
Ampli_8080_1_6_2_residue1 = Ampli_8080_1_6_2_pure(4084:4227);
Ampli_8080_1_6_2_pure2(4073:4216)=Ampli_8080_1_6_2_residue1;
Ampli_8080_1_6_8_pure2 = Ampli_8080_1_6_2_pure2;
Ampli_8080_1_6_74_pure2 = Ampli_8080_1_6_2_pure2;
Ampli_8080_1_6_80_pure2 = Ampli_8080_1_6_2_pure2;

Ampli_8080_1_6_3_pure2 = zeros(1,4975);
Ampli_8080_1_6_3_residue1 = Ampli_8080_1_6_3_pure(4061:4201);
Ampli_8080_1_6_3_pure2(4050:4190)=Ampli_8080_1_6_3_residue1;
Ampli_8080_1_6_7_pure2 = Ampli_8080_1_6_3_pure2;
Ampli_8080_1_6_75_pure2 = Ampli_8080_1_6_3_pure2;
Ampli_8080_1_6_79_pure2 = Ampli_8080_1_6_3_pure2;

Ampli_8080_1_6_4_pure2 = zeros(1,4975);
Ampli_8080_1_6_4_residue1 = Ampli_8080_1_6_4_pure(4053:4188);
Ampli_8080_1_6_4_pure2(4042:4177)=Ampli_8080_1_6_4_residue1;
Ampli_8080_1_6_6_pure2 = Ampli_8080_1_6_4_pure2;
Ampli_8080_1_6_76_pure2 = Ampli_8080_1_6_4_pure2;
Ampli_8080_1_6_78_pure2 = Ampli_8080_1_6_4_pure2;

Ampli_8080_1_6_5_pure2 = zeros(1,4975);
Ampli_8080_1_6_5_residue1 = Ampli_8080_1_6_5_pure(4049:4186);
Ampli_8080_1_6_5_pure2(4038:4175)=Ampli_8080_1_6_5_residue1;
Ampli_8080_1_6_77_pure2 = Ampli_8080_1_6_5_pure2;


Ampli_8080_1_6_10_pure2 = zeros(1,4975);
Ampli_8080_1_6_10_residue1 = Ampli_8080_1_6_10_pure(4085:4235);
Ampli_8080_1_6_10_pure2(4074:4224)=Ampli_8080_1_6_10_residue1;
Ampli_8080_1_6_18_pure2 = Ampli_8080_1_6_10_pure2;
Ampli_8080_1_6_64_pure2 = Ampli_8080_1_6_10_pure2;
Ampli_8080_1_6_72_pure2 = Ampli_8080_1_6_10_pure2;

Ampli_8080_1_6_11_pure2 = zeros(1,4975);
Ampli_8080_1_6_11_residue1 = Ampli_8080_1_6_11_pure(4051:4208);
Ampli_8080_1_6_11_pure2(4040:4197)=Ampli_8080_1_6_11_residue1;
Ampli_8080_1_6_17_pure2 = Ampli_8080_1_6_11_pure2;
Ampli_8080_1_6_65_pure2 = Ampli_8080_1_6_11_pure2;
Ampli_8080_1_6_71_pure2 = Ampli_8080_1_6_11_pure2;

Ampli_8080_1_6_12_pure2 = zeros(1,4975);
Ampli_8080_1_6_12_residue1 = Ampli_8080_1_6_12_pure(4030:4173);
Ampli_8080_1_6_12_pure2(4019:4162)=Ampli_8080_1_6_12_residue1;
Ampli_8080_1_6_16_pure2 = Ampli_8080_1_6_12_pure2;
Ampli_8080_1_6_66_pure2 = Ampli_8080_1_6_12_pure2;
Ampli_8080_1_6_70_pure2 = Ampli_8080_1_6_12_pure2;

Ampli_8080_1_6_13_pure2 = zeros(1,4975);
Ampli_8080_1_6_13_residue1 = Ampli_8080_1_6_13_pure(4021:4159);
Ampli_8080_1_6_13_pure2(4010:4148)=Ampli_8080_1_6_13_residue1;
Ampli_8080_1_6_15_pure2 = Ampli_8080_1_6_13_pure2;
Ampli_8080_1_6_67_pure2 = Ampli_8080_1_6_13_pure2;
Ampli_8080_1_6_69_pure2 = Ampli_8080_1_6_13_pure2;

Ampli_8080_1_6_14_pure2 = zeros(1,4975);
Ampli_8080_1_6_14_residue1 = Ampli_8080_1_6_14_pure(4018:4157);
Ampli_8080_1_6_14_pure2(4007:4146)=Ampli_8080_1_6_14_residue1;
Ampli_8080_1_6_68_pure2 = Ampli_8080_1_6_14_pure2;


Ampli_8080_1_6_19_pure2 = zeros(1,4975);
Ampli_8080_1_6_19_residue1 = Ampli_8080_1_6_19_pure(4063:4213);
Ampli_8080_1_6_19_pure2(4052:4202)=Ampli_8080_1_6_19_residue1;
Ampli_8080_1_6_27_pure2 = Ampli_8080_1_6_19_pure2;
Ampli_8080_1_6_55_pure2 = Ampli_8080_1_6_19_pure2;
Ampli_8080_1_6_63_pure2 = Ampli_8080_1_6_19_pure2;

Ampli_8080_1_6_20_pure2 = zeros(1,4975);
Ampli_8080_1_6_20_residue1 = Ampli_8080_1_6_20_pure(4031:4179);
Ampli_8080_1_6_20_pure2(4020:4168)=Ampli_8080_1_6_20_residue1;
Ampli_8080_1_6_26_pure2 = Ampli_8080_1_6_20_pure2;
Ampli_8080_1_6_56_pure2 = Ampli_8080_1_6_20_pure2;
Ampli_8080_1_6_62_pure2 = Ampli_8080_1_6_20_pure2;

Ampli_8080_1_6_21_pure2 = zeros(1,4975);
Ampli_8080_1_6_21_residue1 = Ampli_8080_1_6_21_pure(4011:4149);
Ampli_8080_1_6_21_pure2(4000:4138)=Ampli_8080_1_6_21_residue1;
Ampli_8080_1_6_25_pure2 = Ampli_8080_1_6_21_pure2;
Ampli_8080_1_6_57_pure2 = Ampli_8080_1_6_21_pure2;
Ampli_8080_1_6_61_pure2 = Ampli_8080_1_6_21_pure2;

Ampli_8080_1_6_22_pure2 = zeros(1,4975);
Ampli_8080_1_6_22_residue1 = Ampli_8080_1_6_22_pure(4003:4138);
Ampli_8080_1_6_22_pure2(3992:4127)=Ampli_8080_1_6_22_residue1;
Ampli_8080_1_6_24_pure2 = Ampli_8080_1_6_22_pure2;
Ampli_8080_1_6_58_pure2 = Ampli_8080_1_6_22_pure2;
Ampli_8080_1_6_60_pure2 = Ampli_8080_1_6_22_pure2;

Ampli_8080_1_6_23_pure2 = zeros(1,4975);
Ampli_8080_1_6_23_residue1 = Ampli_8080_1_6_23_pure(3999:4136);
Ampli_8080_1_6_23_pure2(3988:4125)=Ampli_8080_1_6_23_residue1;
Ampli_8080_1_6_59_pure2 = Ampli_8080_1_6_23_pure2;

Ampli_8080_1_6_28_pure2 = zeros(1,4975);
Ampli_8080_1_6_28_residue1 = Ampli_8080_1_6_28_pure(4048:4188);
Ampli_8080_1_6_28_pure2(4037:4177)=Ampli_8080_1_6_28_residue1;
Ampli_8080_1_6_36_pure2 = Ampli_8080_1_6_28_pure2;
Ampli_8080_1_6_46_pure2 = Ampli_8080_1_6_28_pure2;
Ampli_8080_1_6_54_pure2 = Ampli_8080_1_6_28_pure2;

Ampli_8080_1_6_29_pure2 = zeros(1,4975);
Ampli_8080_1_6_29_residue1 = Ampli_8080_1_6_29_pure(4017:4153);
Ampli_8080_1_6_29_pure2(4006:4142)=Ampli_8080_1_6_29_residue1;
Ampli_8080_1_6_35_pure2 = Ampli_8080_1_6_29_pure2;
Ampli_8080_1_6_47_pure2 = Ampli_8080_1_6_29_pure2;
Ampli_8080_1_6_53_pure2 = Ampli_8080_1_6_29_pure2;

Ampli_8080_1_6_30_pure2 = zeros(1,4975);
Ampli_8080_1_6_30_residue1 = Ampli_8080_1_6_30_pure(3998:4128);
Ampli_8080_1_6_30_pure2(3987:4117)=Ampli_8080_1_6_30_residue1;
Ampli_8080_1_6_34_pure2 = Ampli_8080_1_6_30_pure2;
Ampli_8080_1_6_48_pure2 = Ampli_8080_1_6_30_pure2;
Ampli_8080_1_6_52_pure2 = Ampli_8080_1_6_30_pure2;

Ampli_8080_1_6_31_pure2 = zeros(1,4975);
Ampli_8080_1_6_31_residue1 = Ampli_8080_1_6_31_pure(3990:4118);
Ampli_8080_1_6_31_pure2(3979:4107)=Ampli_8080_1_6_31_residue1;
Ampli_8080_1_6_33_pure2 = Ampli_8080_1_6_31_pure2;
Ampli_8080_1_6_49_pure2 = Ampli_8080_1_6_31_pure2;
Ampli_8080_1_6_51_pure2 = Ampli_8080_1_6_31_pure2;

Ampli_8080_1_6_32_pure2 = zeros(1,4975);
Ampli_8080_1_6_32_residue1 = Ampli_8080_1_6_32_pure(3987:4116);
Ampli_8080_1_6_32_pure2(3976:4105)=Ampli_8080_1_6_32_residue1;
Ampli_8080_1_6_50_pure2 = Ampli_8080_1_6_32_pure2;

Ampli_8080_1_6_37_pure2 = zeros(1,4975);
Ampli_8080_1_6_37_residue1 = Ampli_8080_1_6_37_pure(4040:4174);
Ampli_8080_1_6_37_pure2(4029:4163)=Ampli_8080_1_6_37_residue1;
Ampli_8080_1_6_45_pure2 = Ampli_8080_1_6_37_pure2;

Ampli_8080_1_6_38_pure2 = zeros(1,4975);
Ampli_8080_1_6_38_residue1 = Ampli_8080_1_6_38_pure(4010:4140);
Ampli_8080_1_6_38_pure2(3999:4129)=Ampli_8080_1_6_38_residue1;
Ampli_8080_1_6_44_pure2 = Ampli_8080_1_6_38_pure2;

Ampli_8080_1_6_39_pure2 = zeros(1,4975);
Ampli_8080_1_6_39_residue1 = Ampli_8080_1_6_39_pure(3990:4117);
Ampli_8080_1_6_39_pure2(3979:4106)=Ampli_8080_1_6_39_residue1;
Ampli_8080_1_6_43_pure2 = Ampli_8080_1_6_39_pure2;

Ampli_8080_1_6_40_pure2 = zeros(1,4975);
Ampli_8080_1_6_40_residue1 = Ampli_8080_1_6_40_pure(3983:4104);
Ampli_8080_1_6_40_pure2(3972:4093)=Ampli_8080_1_6_40_residue1;
Ampli_8080_1_6_42_pure2 = Ampli_8080_1_6_40_pure2;

Ampli_8080_1_6_41_pure2 = zeros(1,4975);
Ampli_8080_1_6_41_residue1 = Ampli_8080_1_6_41_pure(3979:4104);
Ampli_8080_1_6_41_pure2(3968:4093)=Ampli_8080_1_6_41_residue1;


% 8080_2_1_offset
Ampli_8080_2_1_1_pure2 = zeros(1,4975);
Ampli_8080_2_1_1_residue1 = Ampli_8080_2_1_1_pure(4117:4246);
Ampli_8080_2_1_1_pure2(4117:4246)=Ampli_8080_2_1_1_residue1;
Ampli_8080_2_1_5_pure2 = Ampli_8080_2_1_1_pure2;
Ampli_8080_2_1_21_pure2 = Ampli_8080_2_1_1_pure2;
Ampli_8080_2_1_25_pure2 = Ampli_8080_2_1_1_pure2;

Ampli_8080_2_1_2_pure2 = zeros(1,4975);
Ampli_8080_2_1_2_residue1 = Ampli_8080_2_1_2_pure(4065:4205);
Ampli_8080_2_1_2_pure2(4049:4189)=Ampli_8080_2_1_2_residue1;
Ampli_8080_2_1_4_pure2 = Ampli_8080_2_1_2_pure2;
Ampli_8080_2_1_22_pure2 = Ampli_8080_2_1_2_pure2;
Ampli_8080_2_1_24_pure2 = Ampli_8080_2_1_2_pure2;

Ampli_8080_2_1_3_pure2 = zeros(1,4975);
Ampli_8080_2_1_3_residue1 = Ampli_8080_2_1_3_pure(4053:4191);
Ampli_8080_2_1_3_pure2(4037:4175)=Ampli_8080_2_1_3_residue1;
Ampli_8080_2_1_23_pure2 = Ampli_8080_2_1_3_pure2;

Ampli_8080_2_1_6_pure2 = zeros(1,4975);
Ampli_8080_2_1_6_residue1 = Ampli_8080_2_1_6_pure(4066:4216);
Ampli_8080_2_1_6_pure2(4050:4200)=Ampli_8080_2_1_6_residue1;
Ampli_8080_2_1_10_pure2 = Ampli_8080_2_1_6_pure2;
Ampli_8080_2_1_16_pure2 = Ampli_8080_2_1_6_pure2;
Ampli_8080_2_1_20_pure2 = Ampli_8080_2_1_6_pure2;

Ampli_8080_2_1_7_pure2 = zeros(1,4975);
Ampli_8080_2_1_7_residue1 = Ampli_8080_2_1_7_pure(4014:4152);
Ampli_8080_2_1_7_pure2(3998:4136)=Ampli_8080_2_1_7_residue1;
Ampli_8080_2_1_9_pure2 = Ampli_8080_2_1_7_pure2;
Ampli_8080_2_1_17_pure2 = Ampli_8080_2_1_7_pure2;
Ampli_8080_2_1_19_pure2 = Ampli_8080_2_1_7_pure2;

Ampli_8080_2_1_8_pure2 = zeros(1,4975);
Ampli_8080_2_1_8_residue1 = Ampli_8080_2_1_8_pure(4003:4140);
Ampli_8080_2_1_8_pure2(3987:4124)=Ampli_8080_2_1_8_residue1;
Ampli_8080_2_1_18_pure2 = Ampli_8080_2_1_8_pure2;

Ampli_8080_2_1_11_pure2 = zeros(1,4975);
Ampli_8080_2_1_11_residue1 = Ampli_8080_2_1_11_pure(4043:4178);
Ampli_8080_2_1_11_pure2(4027:4162)=Ampli_8080_2_1_11_residue1;
Ampli_8080_2_1_15_pure2 = Ampli_8080_2_1_11_pure2;

Ampli_8080_2_1_12_pure2 = zeros(1,4975);
Ampli_8080_2_1_12_residue1 = Ampli_8080_2_1_12_pure(3994:4120);
Ampli_8080_2_1_12_pure2(3978:4104)=Ampli_8080_2_1_12_residue1;
Ampli_8080_2_1_14_pure2 = Ampli_8080_2_1_12_pure2;

Ampli_8080_2_1_13_pure2 = zeros(1,4975);
Ampli_8080_2_1_13_residue1 = Ampli_8080_2_1_13_pure(3983:4109);
Ampli_8080_2_1_13_pure2(3967:4093)=Ampli_8080_2_1_13_residue1;

%% save data
% for i = 1:25
%     eval(['rep = Ampli_8080_normal_',num2str(i),'_pure2(s_index_start:s_index_end);'])
%     data_8080_normal_25_zerosetting(i,:) = rep;
% end
% save('data_8080_normal_25_zerosetting.mat','data_8080_normal_25_zerosetting');


% //test
% txy = zeros(3001,5,5);
% 
% C = Ampli_8080_normal_1_pure2(s_index_start:s_index_end);
% B = [1,2,3,4,5];
% B = repmat(B,3001,1);
% A = [-1,-2,-3,-4,-5];
% A = repmat(A,3001,1);
% txy(:,1,:)=reshape(B,[3001,1,5]); % y
% txy(:,:,1)=reshape(A,[3001,5,1]); % x
% txy(:,1,1)= reshape(C,[3001,1,1]);
% //


% for i = 1:25
%     eval(['rep = Ampli_8080_0_6_',num2str(i),'_pure2(s_index_start:s_index_end);'])
%     data_8080_0_6_25_zerosetting(i,:) = rep;
% end
% save('data_8080_0_6_25_zerosetting.mat','data_8080_0_6_25_zerosetting');
% 
% for i = 1:25
%     eval(['rep = Ampli_8080_1_6_',num2str(i),'_pure2(s_index_start:s_index_end);'])
%     data_8080_1_6_25_zerosetting(i,:) = rep;
% end
% save('data_8080_1_6_25_zerosetting.mat','data_8080_1_6_25_zerosetting');


% for i = 1:81
%     eval(['rep = Ampli_8080_1_6_',num2str(i),'_pure2(s_index_start:s_index_end);'])
%     data_8080_1_6_81_zerosetting(i,:) = rep;
% end

% figure
% for i = 1 : 100
%     plot(data_8080_1_6_100_zerosetting(i,:))
%     hold on
% end
% xlabel('Time sample');ylabel('Amplitude (V)')
% save('./MatData/data_8080_1_6_81_zerosetting.mat','data_8080_1_6_81_zerosetting');



% data_8080_1_6_25_zerosetting_test = [Ampli_8080_1_6_1_pure2(s_index_start:s_index_end); Ampli_8080_1_6_3_pure2(s_index_start:s_index_end); Ampli_8080_1_6_5_pure2(s_index_start:s_index_end);  
%     Ampli_8080_1_6_5_pure2(s_index_start:s_index_end); Ampli_8080_1_6_3_pure2(s_index_start:s_index_end); Ampli_8080_1_6_1_pure2(s_index_start:s_index_end);
%     Ampli_8080_1_6_21_pure2(s_index_start:s_index_end); Ampli_8080_1_6_23_pure2(s_index_start:s_index_end); Ampli_8080_1_6_25_pure2(s_index_start:s_index_end); 
%     Ampli_8080_1_6_25_pure2(s_index_start:s_index_end); Ampli_8080_1_6_23_pure2(s_index_start:s_index_end); Ampli_8080_1_6_21_pure2(s_index_start:s_index_end); 
%     Ampli_8080_1_6_41_pure2(s_index_start:s_index_end); Ampli_8080_1_6_43_pure2(s_index_start:s_index_end); Ampli_8080_1_6_45_pure2(s_index_start:s_index_end); 
%     Ampli_8080_1_6_45_pure2(s_index_start:s_index_end); Ampli_8080_1_6_43_pure2(s_index_start:s_index_end); Ampli_8080_1_6_41_pure2(s_index_start:s_index_end);
%     Ampli_8080_1_6_41_pure2(s_index_start:s_index_end); Ampli_8080_1_6_43_pure2(s_index_start:s_index_end); Ampli_8080_1_6_45_pure2(s_index_start:s_index_end); 
%     Ampli_8080_1_6_45_pure2(s_index_start:s_index_end); Ampli_8080_1_6_43_pure2(s_index_start:s_index_end); Ampli_8080_1_6_41_pure2(s_index_start:s_index_end);
%     Ampli_8080_1_6_21_pure2(s_index_start:s_index_end); Ampli_8080_1_6_23_pure2(s_index_start:s_index_end); Ampli_8080_1_6_25_pure2(s_index_start:s_index_end); 
%     Ampli_8080_1_6_25_pure2(s_index_start:s_index_end); Ampli_8080_1_6_23_pure2(s_index_start:s_index_end); Ampli_8080_1_6_21_pure2(s_index_start:s_index_end); 
%     Ampli_8080_1_6_1_pure2(s_index_start:s_index_end); Ampli_8080_1_6_3_pure2(s_index_start:s_index_end); Ampli_8080_1_6_5_pure2(s_index_start:s_index_end);  
%     Ampli_8080_1_6_5_pure2(s_index_start:s_index_end); Ampli_8080_1_6_3_pure2(s_index_start:s_index_end); Ampli_8080_1_6_1_pure2(s_index_start:s_index_end);
%    ]; %
% 
% save('./MatData/data_8080_1_6_25_zerosetting_test.mat','data_8080_1_6_25_zerosetting_test');

% 
% 8080_2_1
% for i = 1:25
%     eval(['rep = Ampli_8080_2_1_',num2str(i),'_pure2(s_index_start:s_index_end);'])
%     data_8080_2_1_25_zerosetting(i,:) = rep;
% end
% save('data_8080_2_1_25_zerosetting.mat','data_8080_2_1_25_zerosetting');



%% plot

% figure(1)
% plot(time_8080_normal_11, Ampli_8080_normal_11);
% hold on
% plot(time_8080_2_1_11, Ampli_8080_2_1_11);
% xlabel('Time (ns)');ylabel('Amplitude (v)')
% legend('normal-11','深Ⅲ度烧伤-11')



% figure(3)
% plot(t_unified, Ampli_8080_normal_1_pure)
% hold on
% plot(t_unified, Ampli_8080_0_02_1_pure);
% hold on
% plot(t_unified, Ampli_8080_normal_2_pure);
% hold on
% plot(t_unified, Ampli_8080_0_02_2_pure);
% xlabel('Time (ns)');ylabel('Amplitude (v)')
% legend('normal-1','Ⅰ度烧伤-1','normal-2','Ⅰ度烧伤-2')
% title('未截取、未调节峰值的时间数据')

% y = zeros(1,4975);
% figure(4)
% plot(y)
% hold on
% plot( Ampli_8080_1_6_41_pure);
% hold on
% plot( Ampli_8080_1_6_42_pure);
% hold on
% plot( Ampli_8080_1_6_43_pure);
% hold on
% plot( Ampli_8080_1_6_44_pure);
% hold on
% plot( Ampli_8080_1_6_45_pure);
% xlabel('Time (ns)');ylabel('Amplitude (v)')
% legend('0','深Ⅱ度烧伤-1','深Ⅱ度烧伤-2','深Ⅱ度烧伤-3','深Ⅱ度烧伤-4','深Ⅱ度烧伤-5')
% title('未截取、未调节峰值的时间采样点数据')


figure(6)
% plot( Ampli_8080_normal_6_pure2(s_index_start:s_index_end))
% hold on
plot( Ampli_8080_1_6_13_pure2(s_index_start:s_index_end));
hold on
plot( Ampli_8080_1_6_12_pure2(s_index_start:s_index_end));
hold on
plot( Ampli_8080_1_6_11_pure2(s_index_start:s_index_end));
xlabel('Time (ns)');ylabel('Amplitude (v)')
legend('浅Ⅱ度烧伤-13','浅Ⅱ度烧伤-12','浅Ⅱ度烧伤-11')
title('截取、调节峰值的时间采样点数据')

% figure(7)
% plot(t_unified(s_index_start:s_index_end), Ampli_8080_normal_1_pure2(s_index_start:s_index_end))
% hold on
% plot(t_unified(s_index_start:s_index_end), Ampli_8080_0_02_1_pure2(s_index_start:s_index_end));
% xlabel('Time (ns)');ylabel('Amplitude (v)')
% legend('normal-1','深Ⅲ度烧伤-1')
% title('截取、调节峰值的时间数据')

% figure(8)
% for i = 1 : 25
%     plot(t_unified(s_index_start:s_index_end),data_8080_normal_25_zerosetting(i,:), 'linewidth',2)
%     hold on
% end
% xlabel('Time (ns)');ylabel('Amplitude (v)')
% title('Normal')
% ax = gca;
% ax.YAxis.Exponent = -2;
% % set(gca,'ytick',-0.1:0.04:0.1)
% set(gcf,'PaperPosition',[0 0 10 8]);
% set(gca,'LooseInset',[0 0 0 0],'FontName','Times New Roman','FontSize',14,'LineWidth',1.5)
% % print(gcf,'-r300','-dtiff',strcat('../EndDiam',num2str(5)));
% legend('1','3','5','7','9','11','13','15','17')

% figure(9)
% for i = 1 : 25
%     plot(t_unified(s_index_start:s_index_end),data_8080_2_1_25_zerosetting(i,:))
%     hold on
% end
% xlabel('Time (ns)');ylabel('Amplitude (v)')
% title('Third Degree')
% s = repmat(data(1,:),4,1);

%% define function convert cell to double type
function res=cell2double(input)
[n,m]=size(input);
res=zeros(n,1);
    for i=1:n
       temp=cell2mat(input(i));
       res(i)=str2double(temp);
    end
end

%% function
function output_ampli = interpolation(input_time, input_ampli,t_unified)

output_ampli = interp1(input_time, input_ampli, t_unified, 'spline');

end

%% function
function output_pure = denoise(input_unified, Ampli_unified)

output_pure = input_unified - Ampli_unified;

end