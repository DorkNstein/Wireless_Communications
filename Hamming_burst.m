%% Hamming BURST Main Program
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
% HE.Tbits
% 
% HE.Tbits(6,1) = 1; 
% HE.Tbits(1,1) = 1; 
% HE.Tbits(4,3) = 0; 
% HE.Tbits(7,4) = 1; 

%% Add burst error
noise_sigma_max = 0.3;
samples = 100;

% chance of burst
sigma = linspace(0,noise_sigma_max,samples);


%% JUNK

for i=1:size(sigma,2)
  
% Adding Noise for given sigma     
%HEN(i).Tbits = add_awgn(HE.Tbits, sigma(i), 2);
HEN(i).Tbits = burst_add(HE.Tbits,sigma(i));

R(i) = Hamming_decode(HEN(i).Tbits);

sigma(i)

%% Find Bit Errors

Err(i) = find_errors(HE.Tbits,R(i).Rbits);
BER_actual(i) = Err(i).BER_actual;
% % BER_actual_w(i) = Err(i).BER_actual_words;
% 
% Err_detected_w(i) = size(find(R(i).E_loc > 0),2);

Err_after_corr(i) = find_errors(HE.Tbits,R(i).Corr_bits);
BER_after_corr(i) = Err_after_corr(i).BER_actual;
% BER_after_corr_w(i) = Err(i).BER_actual_words;

BER_corrected(i) = BER_actual(i) - BER_after_corr(i);
% BER_corrected_w(i) = BER_actual_w(i) - BER_after_corr_w(i);

end

%% PLOT AND LABELS
figure(1)
plot(sigma,BER_actual,'b','LineWidth',2);
hold on
%plot(sigma,BER_corrected,'g','LineWidth',2);
plot(sigma,BER_after_corr,'--r','LineWidth',2);

title('\bf HAMMING 7,4 SCHEME : BURST NOISE VARIANCE VS BER','FontSize',18);
xlabel('\bf Noise Standard deviation (  \sigma )','FontSize',16);
ylabel('\bf BER ','FontSize',16);
h = legend('Actual BER','BER after Hamming correction',...
    'Location','NorthWest');
set(h,'FontSize',16);
hold off

 


% R.Rbits
% R.Err_at
% R.Corr_bits
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
title_string = sprintf('Noisy image before correction, Sigma = %f',sigma(l));
imshow(Im2.image,[]);
tt=title(title_string ,'FontSize',16);

set(tt,'FontWeight','bold');
currFrame = getframe(hFig);
writeVideo(vidObj,currFrame);

end
close(hFig);
close(vidObj);