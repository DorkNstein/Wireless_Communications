%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% RECONSTRUCT FROM HAMMING ENCODE/DECODE

% Reconstruct 8bit code from hamming output by removing parity bits and
% reshaping

function R_bits = Reconstruct_Hbits(bits)

bits_2 = [bits(3,:); bits(5:7,:)];

R_bits = bits_2(:,1:2:end);
R_bits = [R_bits; bits_2(:,2:2:end)];

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%