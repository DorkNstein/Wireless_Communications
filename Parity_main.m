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