%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% FUNCTION TO CONVERT THE RECEIVED BIT STREAM BACK TO ORIGINAL IMAGE
% NOTE : BITS MUST BE A ROW VECTOR OF BINARY(0 or 1) IN CHAR DATATYPE

function [R_Image] = bits2im(Bits,Image_size)

bit_length = 8; % Image values go for 0-255

% Converting to a column vector of values
for i=1:size(Bits,2)
    R_Image.bit_values(1,i) = 0;
    for j=1:8
        R_Image.bit_values(1,i) =  R_Image.bit_values(1,i)+ ...
            (Bits(j,i)*2^(8-j));
    end
end
% Reshaping to the orginal image size
R_Image.image = reshape((R_Image.bit_values),Image_size);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%