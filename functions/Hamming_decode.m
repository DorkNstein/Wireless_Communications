%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Hamming 7,4 Decoding

function R = Hamming_decode(HE_bits)
% HE_bits is a 8 row, many columns matrix
% Define N and K for the Hamming Encoding 7,4
% N = 7;
% K = 4;
%
% % Change the received data to 4bits in each column (for K = 4)
% H.new_data_bits = reshape(bits, K , size(bits,2) * (8/K) ) ;

% Define H
H = [ 1 0 1 0 1 0 1;
    0 1 1 0 0 1 1;
    0 0 0 1 1 1 1];

% Decoding
for i = 1 : size(HE_bits,2)
    
    R.Ebits(:,i) = H * HE_bits(:,i);
    
end
R.Ebits = mod(R.Ebits,2);


% CORRECTING
% Gives error locations as in H

% R.E_loc = (bin2dec(num2str(R.Ebits')))'; % TAKES TOO LONG

for z=1:size(R.Ebits,2);
    R.E_loc(1,z) = 4*R.Ebits(1,z) + 2*R.Ebits(2,z) + 1*R.Ebits(3,z);
end

R.H_loc = (bin2dec(num2str(H')))';

% Defining Received bits with errors
R.Rbits = HE_bits;
R.Corr_bits = R.Rbits;

for i = 1 : size(R.Rbits,2)
    R.Err_at(i) = 0;
    if(R.E_loc(i))
        % Location of the bits (in row number) of where the error occured
        % in each column
        R.Err_at(i) = (find(R.H_loc == R.E_loc(i)));
        % Corrected bits
        R.Corr_bits(R.Err_at(i),i) = ~R.Corr_bits(R.Err_at(i),i);
    end
end


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%