% Add additive white gaussian noise; Also, choose the scheme to used (1|2)

function [Rpbits] = add_awgn(pbits, sigma, scheme)
%% SCHEME 1 : Defining pbits from -1 to +1 instead of 0 to 1
if(scheme == 1)
    temp_pbits = 2*pbits - ones(size(pbits));
    
    temp_Rpbits = temp_pbits + sigma*randn(size(pbits));
    
    
    Rpbits = zeros(size(temp_Rpbits));
    Rpbits(find(temp_Rpbits>0)) = 1;
end

%% SCHEME 2 : Letting pbits from 0 to 1
if(scheme == 2)
    temp_Rpbits = pbits + sigma * randn(size(pbits));
    
    temp_Rpbits(find(temp_Rpbits>1)) = 1;
    temp_Rpbits(find(temp_Rpbits<0)) = 0;
    temp_Rpbits = round(temp_Rpbits);
    
    Rpbits = temp_Rpbits;
end

end