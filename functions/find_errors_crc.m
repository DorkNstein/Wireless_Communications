%% FIND THE BIT ERRORS BETWEEN TRANSMITTED AND RECEIVED BITS

function Error = find_errors_crc(T,R)
   
    Error.diff = abs(T - R);

    Error.symbol = sum(Error.diff);
    
    Error.symbol2 = Error.symbol(1:3:size(Error.symbol,2)) + ...
                    Error.symbol(2:3:size(Error.symbol,2)) + ...
                    Error.symbol(3:3:size(Error.symbol,2));
    
    Error.total = sum(Error.symbol(:));
    
    Error.total_words = size(find(Error.symbol>0),2);
    
    Error.total_CRC = size(find(Error.symbol2>0),2);
    
    Error.BER_actual = 100*(Error.total/prod(size(T))); 
    
    Error.BER_actual_words = 100*(Error.total_words/(size(T,2)));
    
    Error.BER_CRC = 100*(Error.total_CRC/((size(T,2))/3));
    
end