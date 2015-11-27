%% Image to bits

function T_Im = im2bits(I,bit_length)

I = double(I);
T_Im.size = size(I);
Im = I(:);

% BITS IN THE FORM OF CHARACTERS
T_Im.bits_char = dec2bin(Im,bit_length);

% CONVERTING CHARACTERS TO NUMBERS 
for j = 1:size(T_Im.bits_char,1)
    for i=1:bit_length
        T_Im.bits(i,j) = str2num(T_Im.bits_char(j,i));
    end    
end