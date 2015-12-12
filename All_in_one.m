%% PARITY MAIN PROGRAM


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

%% To Generate New Data
% bit_length = 8;
% T = im2bits(imread('cameraman.tif'),bit_length);
% save T

% LOADING GENERATED DATA

load('T.mat','T'); % T.bits contains columns of binary bits


%% Adding the parity bit

% To add parity bit at the end of each column in T_image.bits
% Specify even or odd parity

T.pbits = parity_add(T.bits,'even');

%% Adding awgn noise to data

noise_sigma_max = 0.5;
samples = 100;

sigma = linspace(0,noise_sigma_max,samples);

for i=1:size(sigma,2)
    % Adding Noise for given sigma
    [R(i).pbits] = add_awgn(T.pbits, sigma(i), 2);
    
    % Find Bit Errors
    Err(i) = find_errors(T.pbits,R(i).pbits);
    BER_actual(i) = Err(i).BER_actual;
    BER_actual_w(i) = Err(i).BER_actual_words;
    
    undetected(i) = sum(Err(i).symbol(find(Err(i).symbol == 2 | ...
        Err(i).symbol == 4 | Err(i).symbol == 6 | Err(i).symbol == 8)));
    undetected_w(i) = size(Err(i).symbol(find(Err(i).symbol == 2 | ...
        Err(i).symbol == 4 | Err(i).symbol == 6 | Err(i).symbol == 8)),2);
    
    parity_error(i) = 100*(undetected(i)/prod(size(R(i).pbits)));
    parity_error_w(i) = 100*(undetected_w(i)/(size(R(i).pbits,2)));
    
    parity_estimated_BER(i) = BER_actual(i) - parity_error(i);
    parity_estimated_BER_w(i) = BER_actual_w(i) - parity_error_w(i);
    
end

%% PLOT AND LABELS
figure(1)
plot(sigma,BER_actual,'b','LineWidth',2);
hold on
plot(sigma,parity_estimated_BER,'g','LineWidth',2);
plot(sigma,parity_error,'--r','LineWidth',2);

title('\bf PARITY SCHEME : NOISE VARIANCE VS BER','FontSize',18);
xlabel('\bf Noise Standard deviation (  \sigma )','FontSize',16);
ylabel('\bf BER ','FontSize',16);
h = legend('Actual BER','Parity estimated BER','Parity undetected',...
    'Location','NorthWest');
set(h,'FontSize',16);
hold off

figure(2)
plot(sigma,BER_actual_w,'b','LineWidth',2);
hold on
plot(sigma,parity_estimated_BER_w,'g','LineWidth',2);
plot(sigma,parity_error_w,'--r','LineWidth',2);

title('\bf PARITY SCHEME : NOISE VARIANCE VS BER - Symbols','FontSize',18);
xlabel('\bf Noise Standard deviation (  \sigma )','FontSize',16);
ylabel('\bf BER ','FontSize',16);
h = legend('Actual BER (Symbol)','Parity estimated BER',...
    'Parity undetected','Location','NorthWest');
set(h,'FontSize',16);
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% CRC_Checksum


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

%% To Generate New Data
% bit_length = 8;
% T = im2bits(imread('cameraman.tif'),bit_length);
% save T

% LOADING GENERATED DATA

load('T.mat','T'); % T.bits contains columns of binary bits


%% Appending a 8bit Checksum for every 16 bits

% To add checksum as a new column in T_image.bits
[T.pbits, S] = checksum_add(T.bits);


%% Adding awgn noise to data

noise_sigma_max = 0.5;
samples = 100;

sigma = linspace(0,noise_sigma_max,samples);

for i=1:size(sigma,2)
    
    % Adding Noise for given sigma
    [R(i).pbits] = add_awgn(T.pbits, sigma(i), 2);
    
    % Find Bit Errors
    Err(i) = find_errors_crc(T.pbits,R(i).pbits);
    
    BER_actual(i) = Err(i).BER_CRC;
    
    [summed, t_error, BER_detected(i)] = checksum_check(R(i).pbits);
    
    disp(i);
    
end

%% PLOT AND LABELS
figure(1)
plot(sigma,BER_actual,'b','LineWidth',2);
hold on
plot(sigma,BER_detected,'g','LineWidth',2);

title('\bf CRC-CHECKSUM : NOISE VARIANCE VS BER','FontSize',18);
xlabel('\bf Noise Standard deviation (  \sigma )','FontSize',16);
ylabel('\bf BER-symbols(16bits) ','FontSize',16);
h = legend('Actual BER','CRC Detected BER',...
    'Location','NorthWest');
set(h,'FontSize',16);
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% CRC_Checksum_burst_noise


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

%% To Generate New Data
% bit_length = 8;
% T = im2bits(imread('cameraman.tif'),bit_length);
% save T

% LOADING GENERATED DATA

load('T.mat','T'); % T.bits contains columns of binary bits


%% Appending a 8bit Checksum for every 16 bits

% To add checksum as a new column in T_image.bits
[T.pbits, S] = checksum_add(T.bits);


%% Adding awgn noise to data

noise_sigma_max = 0.35;
samples = 100;

sigma = linspace(0,noise_sigma_max,samples);

for i=1:size(sigma,2)
    
    % Adding Noise for given sigma
    [R(i).pbits] = burst_add(T.pbits, sigma(i));
    
    % Find Bit Errors
    Err(i) = find_errors_crc(T.pbits,R(i).pbits);
    
    BER_actual(i) = Err(i).BER_CRC;
    
    [summed, t_error, BER_crc ] = checksum_check(R(i).pbits);
    
    BER_detected(i) = BER_crc;
    
    disp(i);
    
end

%% PLOT AND LABELS

figure(1)
plot(sigma,BER_actual,'b','LineWidth',2);
hold on
plot(sigma,BER_detected,'g','LineWidth',2);

title('\bf CRC-CHECKSUM : BURST NOISE VARIANCE VS BER','FontSize',18);
xlabel('\bf Noise Standard deviation (  \sigma )','FontSize',16);
ylabel('\bf BER-symbols(16bits) ','FontSize',16);
h = legend('Actual BER','CRC Detected BER',...
    'Location','NorthWest');
set(h,'FontSize',16);
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Hamming Main Program


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

%% To Generate New Data
% bit_length = 8;
% T = im2bits(imread('cameraman.tif'),bit_length);
% save T

% LOADING GENERATED DATA

load('T.mat','T'); % T.bits contains columns of binary bits

bits = T.bits;

% Encode
HE = Hamming_encode(bits);

%% Add awgn error
noise_sigma_max = 0.35;
samples = 100;

sigma = linspace(0,noise_sigma_max,samples);

% Looping through the noise
for i=1:size(sigma,2)
    
    % Adding Noise for given sigma
    HEN(i).Tbits = add_awgn(HE.Tbits, sigma(i), 2);
    R(i) = Hamming_decode(HEN(i).Tbits);
    sigma(i)
    
    % Find Bit Errors
    Err(i) = find_errors(HE.Tbits,R(i).Rbits);
    BER_actual(i) = Err(i).BER_actual;
    
    
    Err_after_corr(i) = find_errors(HE.Tbits,R(i).Corr_bits);
    BER_after_corr(i) = Err_after_corr(i).BER_actual;
    
    BER_corrected(i) = BER_actual(i) - BER_after_corr(i);
    
end

%% PLOT AND LABELS
figure(1)
plot(sigma,BER_actual,'b','LineWidth',2);
hold on
plot(sigma,BER_after_corr,'--r','LineWidth',2);

title('\bf HAMMING 7,4 SCHEME : NOISE VARIANCE VS BER','FontSize',18);
xlabel('\bf Noise Standard deviation (  \sigma )','FontSize',16);
ylabel('\bf BER ','FontSize',16);
h = legend('Actual BER','BER after Hamming correction',...
    'Location','NorthWest');
set(h,'FontSize',16);
hold off

%% Reconstruct

vidObj = VideoWriter('Simulation.avi');
vidObj.FrameRate = 6;
open(vidObj);


for l = 1:100
    
    % Plotting the image after the bit correction by hamming code
    R_bits = Reconstruct_Hbits(R(l).Corr_bits);
    BIT_STREAM=((R_bits));
    Im = bits2im(BIT_STREAM,[256 256]);
    
    hFig = figure(2);
    set(hFig, 'Position', [150 150 1000 600])
    subplot(1,2,2)
    imshow(Im.image,[]);
    title('\bf Image after Hamming correction','FontSize',16);
    
    % plotting the image before the correction by hamming code
    E_bits = Reconstruct_Hbits(HEN(l).Tbits);
    BIT_STREAM=((E_bits));
    Im2 = bits2im(BIT_STREAM,[256 256]);
    subplot(1,2,1)
    title_string = sprintf('Noisy image before correction, Sigma = %f',...
        sigma(l));
    imshow(Im2.image,[]);
    tt=title(title_string ,'FontSize',16);
    
    set(tt,'FontWeight','bold');
    currFrame = getframe(hFig);
    writeVideo(vidObj,currFrame);
    
end
close(hFig);
close(vidObj);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Hamming With Burst Noise Main Program


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

%% To Generate New Data
% bit_length = 8;
% T = im2bits(imread('cameraman.tif'),bit_length);
% save T

% LOADING GENERATED DATA

load('T.mat','T'); % T.bits contains columns of binary bits

bits = T.bits;

HE = Hamming_encode(bits);
%% Add burst error
noise_sigma_max = 0.3;
samples = 100;

% chance of burst
sigma = linspace(0,noise_sigma_max,samples);

% Looping through the noise
for i=1:size(sigma,2)
    
    % Adding Noise for given sigma
    HEN(i).Tbits = burst_add(HE.Tbits,sigma(i));
    R(i) = Hamming_decode(HEN(i).Tbits);
    sigma(i)
    
    % Find Bit Errors
    
    Err(i) = find_errors(HE.Tbits,R(i).Rbits);
    BER_actual(i) = Err(i).BER_actual;
    
    Err_after_corr(i) = find_errors(HE.Tbits,R(i).Corr_bits);
    BER_after_corr(i) = Err_after_corr(i).BER_actual;
    
    BER_corrected(i) = BER_actual(i) - BER_after_corr(i);
    
end

%% PLOT AND LABELS

figure(1)
plot(sigma,BER_actual,'b','LineWidth',2);
hold on
plot(sigma,BER_after_corr,'--r','LineWidth',2);

title('\bf HAMMING 7,4 SCHEME : BURST NOISE VARIANCE VS BER',...
    'FontSize',18);
xlabel('\bf Noise Standard deviation (  \sigma )','FontSize',16);
ylabel('\bf BER ','FontSize',16);
h = legend('Actual BER','BER after Hamming correction',...
    'Location','NorthWest');
set(h,'FontSize',16);
hold off

%% Reconstruct

vidObj = VideoWriter('Hamming_Burst_Simulation.avi');
vidObj.FrameRate = 6;
open(vidObj);

for l = 1:100
    
    % Plotting the image after the bit correction by hamming code
    R_bits = Reconstruct_Hbits(R(l).Corr_bits);
    BIT_STREAM=((R_bits));
    Im = bits2im(BIT_STREAM,[256 256]);
    
    hFig = figure(2);
    set(hFig, 'Position', [150 150 1000 600])
    subplot(1,2,2)
    imshow(Im.image,[]);
    title('\bf Image after Hamming correction','FontSize',16);
    
    % plotting the image before the correction by hamming code
    E_bits = Reconstruct_Hbits(HEN(l).Tbits);
    BIT_STREAM=((E_bits));
    Im2 = bits2im(BIT_STREAM,[256 256]);
    subplot(1,2,1)
    title_string = sprintf('Noisy image before correction, Sigma = %f'...
        ,sigma(l));
    imshow(Im2.image,[]);
    tt=title(title_string ,'FontSize',16);
    
    set(tt,'FontWeight','bold');
    currFrame = getframe(hFig);
    writeVideo(vidObj,currFrame);
    
end
close(hFig);
close(vidObj);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Hamming + Interleave Burst Noise Main Program


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

%% To Generate New Data
% bit_length = 8;
% T = im2bits(imread('cameraman.tif'),bit_length);
% save T

% LOADING GENERATED DATA

load('T.mat','T'); % T.bits contains columns of binary bits

bits = T.bits;

HE = Hamming_encode(bits);

%% Add burst error
noise_sigma_max = 0.35;
samples = 100;

% chance of burst
sigma = linspace(0,noise_sigma_max,samples);

% BACKUP OF BITS BEFORE INTERLEAVING
HE.before_Tbits = HE.Tbits;

% INTERLEAVING
for zz = 1 : 7
    HE.Tbits(zz,:) = circshift(HE.Tbits(zz,:)',20*zz)';
end

% Looping through the noise
for i=1:size(sigma,2)
    % Adding Noise for given sigma
    HEN(i).Tbits = burst_add(HE.Tbits,sigma(i));
    
    % DE-INTERLEAVING
    for zz = 1 : 7
        HEN(i).Rbits(zz,:) = circshift(HEN(i).Tbits(zz,:)',-20*zz)';
    end
    R(i) = Hamming_decode(HEN(i).Rbits);
    sigma(i)
    
    % Find Bit Errors
    
    Err(i) = find_errors(HE.before_Tbits,R(i).Rbits);
    BER_actual(i) = Err(i).BER_actual;
    
    Err_after_corr(i) = find_errors(HE.before_Tbits,R(i).Corr_bits);
    BER_after_corr(i) = Err_after_corr(i).BER_actual;
    
    BER_corrected(i) = BER_actual(i) - BER_after_corr(i);
    
end

%% PLOT AND LABELS
figure(1)
plot(sigma,BER_actual,'b','LineWidth',2);
hold on
plot(sigma,BER_after_corr,'--r','LineWidth',2);

title('\bf HAMMING 7,4 : BURST-INTERLEAVE NOISE VS BER','FontSize',18);
xlabel('\bf Noise Standard deviation (  \sigma )','FontSize',16);
ylabel('\bf BER ','FontSize',16);
h = legend('Actual BER','BER after Hamming correction',...
    'Location','NorthWest');
set(h,'FontSize',16);
hold off

%% Reconstruct

vidObj = VideoWriter('Hamming_Burst_Interleave_Simulation.avi');
vidObj.FrameRate = 6;
open(vidObj);

for l = 1:100
    
    % Plotting the image after the bit correction by hamming code
    R_bits = Reconstruct_Hbits(R(l).Corr_bits);
    BIT_STREAM=((R_bits));
    Im = bits2im(BIT_STREAM,[256 256]);
    
    hFig = figure(2);
    set(hFig, 'Position', [150 150 1000 600])
    subplot(1,2,2)
    imshow(Im.image,[]);
    title('\bf Image after Hamming correction','FontSize',16);
    
    % plotting the image before the correction by hamming code
    E_bits = Reconstruct_Hbits(HEN(l).Rbits);
    BIT_STREAM=((E_bits));
    Im2 = bits2im(BIT_STREAM,[256 256]);
    subplot(1,2,1)
    title_string = sprintf('Noisy image before correction, Sigma = %f'...
        ,sigma(l));
    imshow(Im2.image,[]);
    tt=title(title_string ,'FontSize',16);
    
    set(tt,'FontWeight','bold');
    currFrame = getframe(hFig);
    writeVideo(vidObj,currFrame);
    
end
close(hFig);
close(vidObj);

















%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%

%% FUNCTIONS USED

%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%% IMAGE TO BITS

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

end

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

% ADDS ADDITIVE WHITE GAUSSIAN NOISE
% CHOOSE A SCHEME (1|2)

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ADDS A BURST NOISE TO A GIVEN DATA MATRIX

function b_data = burst_add(bits_matrix, sigma)

burst_min = 3;
burst_max = 10;

Msize = size(bits_matrix);
bits = bits_matrix(:); % converting to a bit stream

% Random locations to add burst
b_prob = sigma * randn(size(bits,1)-burst_max-burst_min,1);
b_prob(b_prob<0) = 0;
b_prob(b_prob>1) = 1;
b_prob = round(b_prob);

sigma2 = 0.5;

%Choosing a random length for burst in the min and max range
length(b_prob==1) = randi(burst_max - (burst_min -1),...
    size(b_prob(b_prob==1),1),1 ) ...
    + ones(size(b_prob(b_prob==1),1),1) *burst_min - 1;


for i=1:size(b_prob,1)
    
    if(b_prob(i) == 1)
        
        % If only few errors occurs in that burst, sigma =0.5
        error = sigma2 + randn(length(i),1);
        
        error(error<0) = 0;
        error(error>1) = 1;
        error = round(error);
        
        % adding error to the bitstream
        bits(i:i+length(i)-1,1) = bits(i:i+length(i)-1,1) + error;
    end
end

% Getting the bits to binary
bits = mod(bits,2);

% Reshaping the bits back to its intial matrix form
b_data = reshape(bits,Msize);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% FIND THE BIT ERRORS BETWEEN TRANSMITTED AND RECEIVED BITS

function Error = find_errors(T,R)

Error.diff = abs(T - R);

Error.symbol = sum(Error.diff);

Error.total = sum(Error.symbol(:));

Error.total_words = size(find(Error.symbol>0),2);

Error.BER_actual = 100*(Error.total/prod(size(T)));

Error.BER_actual_words = 100*(Error.total_words/(size(T,2)));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%% FIND THE BIT ERRORS BETWEEN TRANSMITTED AND RECEIVED BITS FOR CRC

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Hamming 7,4 Encoding

function H = Hamming_encode(bits)
% bits is a 8 row, many columns matrix
% Define N and K for the Hamming Encoding 7,4
N = 7;
K = 4;

% Change the received data to 4bits in each column (for K = 4)
H.new_data_bits = reshape(bits, K , size(bits,2) * (8/K) ) ;

% Define G
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Hamming 7,4 Decoding

function R = Hamming_decode(HE_bits)
% HE_bits is a 8 row, many columns matrix
% Define N and K for the Hamming Encoding 7,4
% N = 7;
% K = 4;
%
% % Change the received data to 4bits in each column (for K = 4)
% H.new_data_bits = reshape(bits, K , size(bits,2) * (8/K) ) ;

% Define H
H = [ 1 0 1 0 1 0 1;
    0 1 1 0 0 1 1;
    0 0 0 1 1 1 1];

% Decoding
for i = 1 : size(HE_bits,2)
    
    R.Ebits(:,i) = H * HE_bits(:,i);
    
end
R.Ebits = mod(R.Ebits,2);


% CORRECTING
% Gives error locations as in H

% R.E_loc = (bin2dec(num2str(R.Ebits')))'; % TAKES TOO LONG

for z=1:size(R.Ebits,2);
    R.E_loc(1,z) = 4*R.Ebits(1,z) + 2*R.Ebits(2,z) + 1*R.Ebits(3,z);
end

R.H_loc = (bin2dec(num2str(H')))';

% Defining Received bits with errors
R.Rbits = HE_bits;
R.Corr_bits = R.Rbits;

for i = 1 : size(R.Rbits,2)
    R.Err_at(i) = 0;
    if(R.E_loc(i))
        % Location of the bits (in row number) of where the error occured
        % in each column
        R.Err_at(i) = (find(R.H_loc == R.E_loc(i)));
        % Corrected bits
        R.Corr_bits(R.Err_at(i),i) = ~R.Corr_bits(R.Err_at(i),i);
    end
end


end

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

% HAMMING CODE IMPLEMENTATION ERROR

function [ERR ] = H_error(HEN,R)

samples = size(R,2);

for j=1:samples
    ERR(i).xor = mod((R(j).Corr_bits .* ~HE.Tbits + ~R(j).Corr_bits ...
                        .* HE.Tbits),2);
end

% TO VIEW EACH ERROR AND CORRECTION
% for j=1:samples
%     clc
%     fprintf('Error at %.0f is : \n \n', j);
%     disp(R(j).Err_at);
%     fprintf('Correction is \n \n', j);
%     disp(mod((R(j).Corr_bits .* ~HE.Tbits + ~R(j).Corr_bits ...
%               .* HE.Tbits),2));
%     pause
% end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
