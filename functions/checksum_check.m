%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% VERIFY THE CHECKSUM TO DETERMINE BIT ERRORS

% Returns the checksum rows
% If all of which are zero, implies no bit errors
function [summed, total_errors, BER_crc] = checksum_check(bits)

% Matrix size of the input bits
rows = size(bits,1);
cols = size(bits,2);

% Converting all bits to decimals
D = bits2im(bits,[1 cols]);
Dec = D.image;

% Summing and checking for all zeors
summed = Dec(1,1:3:cols) + Dec(1,2:3:cols) + Dec(1,3:3:cols);
summed = mod(summed,256);
summed = mod(summed,255);

% Finding the non-zero elements and classifying them as errors
total_errors = size(summed(summed~=0),2);

% BER For each 16 bit symbol
BER_crc = 100*(total_errors/size(summed,2));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%