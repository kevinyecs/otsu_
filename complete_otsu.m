% Save this entire script as complete_otsu.m

% Main Otsu function
function complete_otsu()
    % Load the cat image
    img = imread('Cat03.jpg');
    
    % Run both single and multi-level thresholding
    [thresh1, quant1] = otsu(img, 1);  % Single threshold
    [thresh4, quant4] = otsu(img, 4);  % Four thresholds
    
    % Display all results
    figure('Name', 'Otsu Thresholding Results');
    
    % Original and grayscale
    subplot(2,2,1), imshow(img), title('Original');
    if size(img,3) == 3
        gray_img = rgb2gray(img);
        subplot(2,2,2), imshow(gray_img), title('Grayscale');
    end
    
    % Thresholding results
    subplot(2,2,3), imshow(quant1), title('Single-level threshold');
    subplot(2,2,4), imshow(quant4), title('4-level threshold');
    
    % Print threshold values
    fprintf('Single threshold: %.3f\n', thresh1);
    fprintf('Multi thresholds: %.3f, %.3f, %.3f, %.3f\n', thresh4);
end

% Otsu thresholding implementation
function [thresholds, quantized] = otsu(img, N)
    % Input validation and preprocessing
    if nargin < 1
        error('Image input is required');
    end
    if nargin < 2
        N = 1;  % Default to single threshold
    end
    
    % Convert RGB to grayscale if necessary
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    % Convert to double and normalize if needed
    img = double(img);
    if max(img(:)) > 1
        img = img / 255;
    end
    
    % Calculate histogram
    [counts, ~] = imhist(img);
    P = counts / sum(counts);  % Probability distribution
    
    if N == 1
        % Single-level Otsu
        threshold = singleOtsu(P);
        thresholds = threshold;
    else
        % Multi-level Otsu
        thresholds = multiOtsu(P, N);
    end
    
    % Quantize image based on thresholds
    quantized = quantizeImage(img, thresholds);
end

function threshold = singleOtsu(P)
    L = length(P);
    variance = zeros(L-1, 1);
    
    % Calculate cumulative sums
    w = cumsum(P);
    mu = cumsum((1:L)' .* P);
    mu_t = mu(end);
    
    % Calculate between-class variance for each possible threshold
    for t = 1:L-1
        w1 = w(t);
        w2 = 1 - w1;
        
        if w1 == 0 || w2 == 0
            continue;
        end
        
        mu1 = mu(t) / w1;
        mu2 = (mu_t - mu(t)) / w2;
        
        % Calculate between-class variance
        variance(t) = w1 * w2 * (mu1 - mu2)^2;
    end
    
    % Find threshold that maximizes variance
    [~, threshold] = max(variance);
    threshold = threshold / L;  % Normalize threshold
end

function thresholds = multiOtsu(P, N)
    L = length(P);
    thresholds = zeros(1, N);
    
    % Find thresholds one by one
    for k = 1:N
        max_variance = -inf;
        best_threshold = 0;
        
        % Try each possible threshold
        for t = 1:L-1
            temp_thresholds = sort([thresholds(1:k-1), t/L]);
            variance = calculateVariance(P, temp_thresholds);
            
            if variance > max_variance
                max_variance = variance;
                best_threshold = t/L;
            end
        end
        
        thresholds(k) = best_threshold;
    end
    
    thresholds = sort(thresholds);
end

function variance = calculateVariance(P, thresholds)
    L = length(P);
    thresholds = [0, thresholds, 1];
    variance = 0;
    
    % Calculate variance for each segment
    for i = 1:length(thresholds)-1
        start_idx = max(1, ceil(thresholds(i) * L));
        end_idx = min(L, ceil(thresholds(i+1) * L));
        
        segment_P = P(start_idx:end_idx);
        w = sum(segment_P);
        
        if w > 0
            mu = sum((start_idx:end_idx)' .* segment_P) / (L * w);
            variance = variance + w * mu^2;
        end
    end
end

function quantized = quantizeImage(img, thresholds)
    thresholds = sort([0, thresholds, 1]);
    quantized = zeros(size(img));
    
    % Assign intensity levels
    for i = 1:length(thresholds)-1
        mask = (img >= thresholds(i)) & (img < thresholds(i+1));
        quantized(mask) = (thresholds(i) + thresholds(i+1)) / 2;
    end
end