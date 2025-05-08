function cnnFeatures = trainDeepComponentWithPeriodicKernels(sampledData, savePath)
    % === Parameters ===
    if nargin < 2
        savePath = 'cnn_periodic_features.csv';
    end

    startCol = 3;  % Assuming column 1 = ID, column 2 = label
    X = table2array(sampledData(:, startCol:end));
    labels = double(sampledData{:, 2});
    [nSamples, nDays] = size(X);
    numWeeks = floor(nDays / 7);
    X = X(:, 1:(numWeeks * 7));  % Trim to full weeks

    % === Output initialization ===
    cnnFeatures = zeros(nSamples, 7 * numWeeks);

    fprintf('⚙️  Extracting periodic CNN features using g1/g2...\n');
    for i = 1:nSamples
        usage1D = X(i, :);
        usage2D = reshape(usage1D, [7, numWeeks]);  % 7 rows = days, columns = weeks

        % Apply custom periodic feature map
        featureMap = applyPeriodicKernels(usage2D);  % 7×W output
        cnnFeatures(i, :) = featureMap(:)';
    end

    % === Save if path specified ===
    if ~isempty(savePath)
        writematrix(cnnFeatures, savePath);
        fprintf('✅ CNN periodic features saved to %s\n', savePath);
    end
end

function C = applyPeriodicKernels(V)
    % Input V: 7×W matrix
    [d, m] = size(V);  % d=7, m=weeks
    Vpad = padarray(V, [1 1], 'replicate');  % Pad for 3×3 block operations
    C = zeros(d, m);  % Output matrix

    for i = 2:(d+1)
        for j = 2:(m+1)
            block = Vpad(i-1:i+1, j-1:j+1);  % 3x3 window
            g1val = g1(block);
            g2val = g2(block);
            val = tanh(sum(g1val(:) + g2val(:)));
            C(i-1, j-1) = val;
        end
    end
end

function out = g1(B)
    % Apply row-wise periodic kernel as per the paper
    out = [
        2*B(1,:) - B(2,:) - B(3,:);
        2*B(2,:) - B(1,:) - B(3,:);
        2*B(3,:) - B(1,:) - B(2,:)
    ];
end

function out = g2(B)
    % Column-wise version of g1
    out = g1(B')';
end
