
close all;
clc;
clear;
validOptions = {'building', 'window', 'mri','seismic'};
k = str2double(getenv('K_VALUE'));
addpath(genpath('../'));
addpath(genpath('./'));

M = double(imread('./data/building.jpeg'));
parameters(1).osr = 0.12;
parameters(1).iter = 15;
parameters(1).deltas = [0.11, 0.12, 0.13];
parameters(1).r = 25;
parameters(2).osr = 0.16;
parameters(2).iter = 25;
parameters(2).deltas = [0.11, 0.12, 0.13];
parameters(2).r = 25;
parameters(3).osr = 0.20;
parameters(3).iter = 25;
parameters(3).deltas = [0.11, 0.12, 0.13];
parameters(3).r = 35;


maxP = max(M(:));
M = M / maxP;
M = double(M);

psnr = zeros(length(parameters), length(parameters(1).deltas));
ssim = zeros(length(parameters), length(parameters(1).deltas));
times = zeros(length(parameters), length(parameters(1).deltas));

for i = 1:length(parameters)
    for j = 1:length(parameters(i).deltas)
        delta = parameters(i).deltas(j);
        sr = parameters(i).osr;

        [m, n, q] = size(M);
        tmp = zeros(m, n, q);
        num_r = round(m * delta);
        num_c = round(n * delta);
        I_css = randsample(m, num_r);
        J_css = randsample(n, num_c);
        tmp(:, J_css, :) = M(:, J_css, :);
        tmp(I_css, :, :) = M(I_css, :, :);
        inds = find(tmp ~= 0);
        if numel(inds) < floor(sr * m * n * q)
            selected_inds = inds;
        else
            selected_inds = randsample(inds, floor(sr * m * n * q));
        end
        X_Omega_css = zeros(m, n, q);
        X_Omega_css(selected_inds) = M(selected_inds);

        params_ICURC.eta = [1, 1, 1];
        params_ICURC.TOL = 1e-10;
        params_ICURC.max_ite = parameters(i).iter;

        Omega = find(X_Omega_css ~= 0);
        tic;
        [CC, U_pinv, R, ~, ~, ~] = ICURC(X_Omega_css, I_css, J_css, parameters(i).r, params_ICURC, M);
        times(i, j) = toc;

        CUR = CC * U_pinv * R;
        CUR = CUR.data;
        %CUR = CUR/max(CUR(:));
        CUR(Omega) = X_Omega_css(Omega);
        psnr(i, j) = PSNR(M, CUR, max(M(:)));
        ssim(i, j) = calculate_ssim(M, CUR);
        fprintf('delta: %.4f  osr: %.4f  psnr: %.4f  ssim: %.4f  time: %.4f\n', parameters(i).deltas(j),parameters(i).osr, ...
        psnr(i,j),ssim(i,j),times(i,j));
    end
end

% Create table
num_deltas = length(parameters(1).deltas);
row_labels = cell(3 * num_deltas, 1);

for i = 1:3
    for j = 1: num_deltas
        if i ==1
             row_labels{num_deltas * (i - 1) + j} = ['PSNR Delta = ' num2str(parameters(1).deltas(j)) ' : '];
        elseif i==2
             row_labels{num_deltas * (i - 1) + j} = ['SSIM Delta = ' num2str(parameters(1).deltas(j)) ' : '];
        else
             row_labels{num_deltas * (i - 1) + j} = ['TIME Delta = ' num2str(parameters(1).deltas(j)) ' : '];
        end
   end
end
col_labels = cell(length(parameters), 1);
for i = 1:length(parameters)
    col_labels{i} = [num2str(parameters(i).osr)];
end
results_table = array2table(zeros(3 * num_deltas, length(parameters)), 'VariableNames', col_labels, 'RowNames', row_labels);
for i = 1:length(parameters)
    for j = 1:length(parameters(i).deltas)
         results_table{j,i} = psnr(i,j);
         results_table{(num_deltas)+j,i} = ssim(i,j);
         results_table{2*(num_deltas)+j,i} = times(i,j);
    end
end
disp(results_table)
filename =  sprintf('/mnt/research/huanglx_group/cur_building_%d.mat', k);
save(filename,'results_table');