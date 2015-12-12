%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ADDS A BURST NOISE TO A GIVEN DATA MATRIX

function b_data = burst_add(bits_matrix, sigma)

burst_min = 3;
burst_max = 10;

Msize = size(bits_matrix);
bits = bits_matrix(:); % converting to a bit stream

% Random locations to add burst
b_prob = sigma * randn(size(bits,1)-burst_max-burst_min,1);
b_prob(b_prob<0) = 0;
b_prob(b_prob>1) = 1;
b_prob = round(b_prob);

sigma2 = 0.5;

%Choosing a random length for burst in the min and max range
length(b_prob==1) = randi(burst_max - (burst_min -1),...
    size(b_prob(b_prob==1),1),1 ) ...
    + ones(size(b_prob(b_prob==1),1),1) *burst_min - 1;


for i=1:size(b_prob,1)
    
    if(b_prob(i) == 1)
        
        % If only few errors occurs in that burst, sigma =0.5
        error = sigma2 + randn(length(i),1);
        
        error(error<0) = 0;
        error(error>1) = 1;
        error = round(error);
        
        % adding error to the bitstream
        bits(i:i+length(i)-1,1) = bits(i:i+length(i)-1,1) + error;
    end
end

% Getting the bits to binary
bits = mod(bits,2);

% Reshaping the bits back to its intial matrix form
b_data = reshape(bits,Msize);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%