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