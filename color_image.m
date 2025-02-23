clear;
close all;
M= double(imread('./data/window.jpeg'));
%M= double(imread('./data/building.jpeg'));
 maxP = max(M(:));
M = M/maxP;
[n1, n2, n3] = size(M);
Nway = size(M);
N = ndims(M);
r = 40;  
delta = 0.12; 
sr = 0.20;
%generate t-CCS model 
[m, n,q] = size(M);
tmp = zeros(m,n,q);
num_r = round(m*delta);
num_c = round(n*delta);
I_css = randsample(m,num_r);
J_css = randsample(n,num_c);     
C = M(:,J_css,:);
R = M(I_css,:,:);
tmp(:,J_css,:) = M(:,J_css,:);
tmp(I_css,:,:) = M(I_css,:,:); 
inds = find(tmp~=0);
if numel(inds) < floor(sr*m*n*q)
    selected_inds = inds; 
else
    selected_inds = randsample(inds, floor(sr*m*n*q)); 
end
X_Omega_css = zeros(m, n, q);
X_Omega_css(selected_inds) = M(selected_inds);

%%Apply TICUC to t-CCS based sampling model
params_ICURC.eta = [1, 1, 1]; 
params_ICURC.TOL = 1e-10;
params_ICURC.max_ite = 25;
Omega = find(X_Omega_css~=0);
a1 = tic;
[CC,U_pinv,R, ICURC_time,sample_num,error_list] = ICURC(X_Omega_css, I_css, J_css, r, params_ICURC,M);
cur_time = toc(a1);
CUR = CC*U_pinv*R;
CUR =CUR.data;
CUR(Omega) = X_Omega_css(Omega);
slice = 10;
%%%%%%%%%
figure;
subplot(1,3,1);
imshow(M);
subplot(1,3,2);
imshow(X_Omega_css);
title('Observed');
subplot(1,3,3);
imshow(CUR);
title('TICUR');
sprintf('PSNR: %.4f',PSNR(M,CUR,max(M(:))))
