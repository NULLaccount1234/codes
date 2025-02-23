function [X_Omega_css, I_css, J_css] = CCS_lx(X, params_CCS)
    if(~exist('params_CCS','var'))
        params_CCS=struct();
    end
    params_CCS = SetDefaultParams_CCS(params_CCS);
    p=params_CCS.p;
    delta=params_CCS.delta;
    
    [m, n,q] = size(X);
    % Generate I_css and J_css based on delta

    num_r = round(m*delta);
    num_c = round(n*delta);
    I_css = randsample(m,num_r,false);
    J_css = randsample(n,num_c,false);     
    
    C = X(:,J_css,:);
    R = X(I_css,:,:);
    C_obs_ind = find(rand(size(C))<=p);
    R_obs_ind = find(rand(size(R))<=p);
    C_Obs = zeros(m,num_c,q);
    R_Obs = zeros(num_r,n,q);
    
    

    C_Obs(C_obs_ind) = C(C_obs_ind);
    R_Obs(R_obs_ind) = R(R_obs_ind);
    X_Omega_css = zeros(m, n,q);
    X_Omega_css(I_css, :,:) = R_Obs;
    X_Omega_css(:, J_css,:) = C_Obs;
    
end