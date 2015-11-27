%% FIND THE BIT ERRORS BETWEEN TRANSMITTED AND RECEIVED BITS

function Error = find_errors(T,R)
   
% 
    Error.diff = abs(T - R);

    Error.symbol = sum(Error.diff);

    Error.total = sum(Error.symbol(:));
    
    Error.total_words = size(find(Error.symbol>0),2);
    
    
    Error.BER_actual = 100*(Error.total/prod(size(T))); 
    
    Error.BER_actual_words = 100*(Error.total_words/(size(T,2)));

end