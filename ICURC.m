
function [C,U_pinv,R, ICURC_time,sampled_num,error_list] = ICURC(X_Omega_UR, I_css, J_css, r, params_ICURC,M)
    error_list = [];
    if(~exist('params_ICURC','var'))
        params_ICURC=struct();
    end
    params_ICURC = SetDefaultParams_ICURC(params_ICURC);
    
    eta=params_ICURC.eta; 
    % fprintf('using stepsize eta_C = %f.\n', eta(1));
    % fprintf('using stepsize eta_R = %f.\n', eta(2));
    % fprintf('using stepsize eta_U = %f.\n', eta(3));
    
    TOL=params_ICURC.TOL;
    max_ite=params_ICURC.max_ite;
    steps_are1=params_ICURC.steps_are1;
    
    %This step is to extract observed C, U, and R 
    Obs_U = X_Omega_UR(I_css, J_css,:);  

    Obs_C = X_Omega_UR(:, J_css,:);
    Obs_R = X_Omega_UR(I_css, :,:);
    C_size = size(Obs_C);
    R_size = size(Obs_R);

    all_row_ind = 1:C_size(1); 
    all_col_ind = 1:R_size(2); 

    I_css_comp = setdiff(all_row_ind, I_css); 
    J_css_comp = setdiff(all_col_ind, J_css); 
    
    C = Obs_C(I_css_comp, :,:); 
    R = Obs_R(:, J_css_comp,:); 
    %C and R are C_obs\U, R_obs\U.
    Smp_C = (C ~= 0);
    Smp_R = (R ~= 0); 
    Smp_U = Obs_U ~= 0; 
    L_obs_only_row = R;
    L_obs_only_col = C;  
    L_obs_only_U = Obs_U; 
    
    Omega_row = find(Smp_R);    
    Omega_col = find(Smp_C);  
    Omega_U = find(Smp_U);  
    sampled_num = length(Omega_U)+length(Omega_col)+length(Omega_row);
    
    L_obs_row_vec = L_obs_only_row(Omega_row);
    L_obs_col_vec = L_obs_only_col(Omega_col);   
    L_obs_U_vec = L_obs_only_U(Omega_U); 
    
    normC_obs = norm(L_obs_col_vec,'fro');
    normU_obs = norm(L_obs_U_vec,'fro');
    normR_obs = norm(L_obs_row_vec,'fro'); 

    col_row_norm_sum = normC_obs + normU_obs + normR_obs;    
    
    %Initializing U
    U_i = tensor(Obs_U);
    [u,s,v] = tsvd(U_i);  
    u_shape = u.shape;
    v_shape = v.shape;
    s_shape  = s.shape;
    if u_shape(2)<r
        r = u_shape(2);
    end
    if v_shape(2)<r
        r = v_shape(2);
    end
   if s_shape(1)<r || s_shape(2)<r
        r = min(s_shape(1),s_shape(2));
    end
    u = u(:, 1:r,:);
    v = v(:, 1:r,:);
    s = s(1:r, 1:r,:); 
    U_i = u*s*v'; 
    R = tensor(R);
    L_obs_row_vec = tensor(L_obs_row_vec);
    L_obs_col_vec = tensor(L_obs_col_vec);
    L_obs_U_vec   = tensor(L_obs_U_vec);
    C = tensor(C);
    
   %Calculating error
   
    New_Error = (norm(R(Omega_row) - L_obs_row_vec) + ... 
                            norm(C(Omega_col)- L_obs_col_vec) + ... 
                            norm(U_i(Omega_U)- L_obs_U_vec))/col_row_norm_sum;
    error_list(end+1) = New_Error;
    %If all step sizes are 1
    if steps_are1 
        %fprintf('running ICURC with stepsize all equal to 1\n');
        fct_time = tic;
        for ICURC_ite = 1:max_ite
            ite_itme = tic;
            R = u*u'*R;
            C = C*v*v';        
            %Old_error = New_Error;
            cur = C*mpinv(U_i)*R;
            New_Error = norm(cur-M,"fro")/norm(M,"fro");
            % New_Error = (norm(R(Omega_row) - L_obs_row_vec) + ... 
            %                 norm(C(Omega_col)- L_obs_col_vec) + ... 
            %                 norm(U_i(Omega_U)- L_obs_U_vec))/col_row_norm_sum;  
            error_list =[error_list, New_Error];
            if New_Error < TOL || ICURC_ite == max_ite 
                ICURC_time = toc(fct_time); 
                R(Omega_row) = L_obs_row_vec;
                C(Omega_col) = L_obs_col_vec;
                U_i(Omega_U) = L_obs_U_vec; 

                Final_C = tensor(zeros(C_size));
                Final_R = tensor(zeros(R_size)); 
                Final_C(I_css_comp, :,:) = C;
                Final_R(:, J_css_comp,:) = R;
                Final_C(I_css, :,:) = U_i; 
                Final_R(:, J_css,:) = U_i; 

                C = Final_C;
                R = Final_R; 
                
                [u,s,v] = tsvd(U_i);  
                u = u(:, 1:r,:);
                v = v(:, 1:r,:);
                s = s(1:r, 1:r,:);  
                U_pinv = v*mpinv(s)*u';
                cur = C*U_pinv*R;
                cur = cur.data;
                %fprintf('ICURC finished at  %d-th iteration \n', ICURC_ite);
                %fprintf("final error is %f \n",norm(M-cur,"fro"))
                return
            end
            %Updating R, C, and U
            R(Omega_row) = L_obs_row_vec;
            C(Omega_col) = L_obs_col_vec;
            U_i(Omega_U) =  L_obs_U_vec; 

            [u,s,v] = tsvd(U_i);  
            u = u(:, 1:r,:);
            v = v(:, 1:r,:);
            s = s(1:r, 1:r,:);  
            U_i = u*s*v';
            
            %fprintf('Iteration %d: error: %e, timer: %f \n', ICURC_ite, New_Error, toc(ite_itme));
        end
        
    %If all step sizes are not 1    
    else
        %fprintf('running ICURC with given stepsizes \n');
        step_c = eta(1);
        step_r = eta(2);
        step_u = eta(3);
        
        fct_time = tic;
        for ICURC_ite = 1:max_ite
            ite_itme = tic;
      

                    R = u*u'*R;
                    C = C*v*v';

            % 
            New_Error = (norm(R(Omega_row) - L_obs_row_vec) + ... 
                            norm(C(Omega_col)- L_obs_col_vec) + ... 
                            norm(U_i(Omega_U)- L_obs_U_vec))/col_row_norm_sum;   
            error_list =[error_list, New_Error];        
            if New_Error < TOL || ICURC_ite == max_ite
                ICURC_time = toc(fct_time); 

                R(Omega_row) = L_obs_row_vec;
                C(Omega_col) = L_obs_col_vec;
                U_i(Omega_U) = L_obs_U_vec; 

                Final_C = tensor(zeros(C_size));
                Final_R = tensor(zeros(R_size)); 

                Final_C(I_css_comp, :,:) = C;
                Final_R(:, J_css_comp,:) = R;
                Final_C(I_css, :,:) = U_i; 
                Final_R(:, J_css,:) = U_i; 

                C = Final_C;
                R = Final_R; 

                [u,s,v] = tsvd(U_i);  
                u = u(:, 1:r,:);
                v = v(:, 1:r,:);
                s = s(1:r, 1:r,:);  
                U_pinv = v*mpinv(s)*u';
                %fprintf('ICURC finished at  %d-th iteration \n', ICURC_ite);
                %fprintf('Iteration %d: error: %e, timer: %f \n', ICURC_ite, New_Error, toc(ite_itme));
                return
            end  
            %Updating R, C, and U
            R(Omega_row) =  (1 - step_r).*R(Omega_row) + step_r.*(L_obs_row_vec);     
            C(Omega_col) =  (1 - step_c).*C(Omega_col) + step_c.*(L_obs_col_vec);
            U_i(Omega_U) =  (1 - step_u).*U_i(Omega_U) + step_u.*(L_obs_U_vec); 

            [u,s,v] = tsvd(U_i);  
            u = u(:, 1:r,:);
            v = v(:, 1:r,:);
            s = s(1:r, 1:r,:);  
            U_i = u*s*v';
            %error_record = [error_record, New_Error];
            %
            %fprintf('Iteration %d: error: %e, timer: %f \n', ICURC_ite, New_Error, toc(ite_itme));

         end
     end
end