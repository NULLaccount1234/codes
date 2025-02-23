function avg_ssim = calculate_ssim(tensor1, tensor2)
    % This function calculates the average SSIM between two 3D tensors.
    % tensor1 and tensor2 are the 3D tensors to be compared.
    
    [rows, cols, num_slices] = size(tensor1);
    ssim_values = zeros(num_slices, 1);

    for i = 1:num_slices
        slice1 = tensor1(:,:,i);
        slice2 = tensor2(:,:,i);

        % Calculate SSIM for each slice
        %ssim_values(i) = ssim_cnp(slice1, slice2);
        ssim_values(i) = ssim(slice1, slice2);
    end

    % Calculate average SSIM
    avg_ssim = mean(ssim_values);
end
