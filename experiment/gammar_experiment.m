clear all; clc; close all;

%{
A = [];
B = zeros(3);
C = ones(3);
C1 = cat(1,A,B)
C2 = cat(1,B,C)

C2 = array2table(C2, "VariableNames", {'A', 'B', 'C'})
%}


A = ones(4);

B = A(:,2) ./ 2;

B