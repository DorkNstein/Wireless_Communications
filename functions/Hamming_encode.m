%% Hamming 7,4 Encoding


function H = Hamming_encode(bits)
% bits is a 8 row, many columns matrix
% Define N and K for the Hamming Encoding 7,4
N = 7;
K = 4;

% Change the received data to 4bits in each column (for K = 4) 
H.new_data_bits = reshape(bits, K , size(bits,2) * (8/K) ) ;

% Define G - source wiki
G = [ 1 1 0 1 ;
      1 0 1 1 ;
      1 0 0 0 ;
      0 1 1 1 ;
      0 1 0 0 ;
      0 0 1 0 ;
      0 0 0 1 ];
  

  % Encoding the 3 parity bits
  for i = 1 : size(H.new_data_bits,2)
      
   H.Tbits(:,i) = G * H.new_data_bits(:,i);  
   
  end
  
   H.Tbits = mod(H.Tbits,2);
  
end