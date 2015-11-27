function [ERR ] = H_error(HEN,R)

samples = size(R,2);

for j=1:samples
    ERR(i).xor = mod((R(j).Corr_bits .* ~HE.Tbits + ~R(j).Corr_bits .* HE.Tbits),2);    
end
% TO VIEW EACH ERROR AND CORRECTION
% for j=1:samples
%     clc
%     fprintf('Error at %.0f is : \n \n', j); 
%     disp(R(j).Err_at);
%     fprintf('Correction is \n \n', j); 
%     disp(mod((R(j).Corr_bits .* ~HE.Tbits + ~R(j).Corr_bits .* HE.Tbits),2));
%     pause
% end
end