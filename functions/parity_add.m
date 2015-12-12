%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% ADDS A PARITY BIT ACCORDING TO THE SPECIFIED SCHEME ( EVEN OR ODD )

function pbits = parity_add(bits,str)

% Remainder vector for each column after dividing sum of the column by 2
R = mod(sum(bits),2);

% Deciding the remainder based on the parity scheme
if( strcmpi('even',str) )
    pbits = [bits;R];
else
    pbits = [bits;~R]; % default is odd parity scheme
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%