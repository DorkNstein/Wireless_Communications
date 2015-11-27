%% ADDS A 8-BIT CHECKSUM FOR EVERY 16 BITS

function [pbits, summed] = checksum_add(bits)

% Matirx size of the input bits
rows = size(bits,1);
cols = size(bits,2);


% Converting all bits to decimals
D = bits2im(bits,[1 cols]);
Dec = D.image;

% Summing adjacent columns to create a sum matrix
sum = Dec(1:2:cols) + Dec(2:2:cols);
% Restricting values to be 8bit
sum = mod(sum,256);
% Converting sum back to decimals
sum = im2bits(sum,8);
% Flipping all the bits
summed = ~sum.bits;

% Adding that sum column next to every two columns
for i=1:1:size(summed,2)
    
    pbits(:,(3*i)-2:3*i) =  [bits(:,(2*i)-1:2*i) summed(:,i)];
    
end

end