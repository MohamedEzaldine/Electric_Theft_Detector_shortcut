function modelInfo = trainWideAndDeepWithPeriodicFeatures(sampledData, wideParams, deepParams, clfParams)
    % Default output path for cnn features if not set
    if ~isfield(deepParams, 'savePath')
        deepParams.savePath = '';
    end

    % ========== Wide Component ==========
    fprintf('🔷 Training Wide Component...\n');
    [wideFeatures, wideNet, wideInfo] = trainWideComponentAdaptive(sampledData, wideParams);

    % ========== Deep CNN Component ==========
    fprintf('🔶 Training Deep CNN Component...\n');
    [cnnFeatures1, deepNet, deepInfo] = trainDeepComponentCNN(sampledData, deepParams);

    % ========== Periodic CNN Feature Extraction ==========
    fprintf('🌀 Extracting handcrafted periodic CNN features...\n');
    cnnFeatures2 = trainDeepComponentWithPeriodicKernels(sampledData, '');

    % Combine CNN features
    cnnFeatures = [cnnFeatures1, cnnFeatures2];

    % ========== Final Logistic Regression Classifier ==========
    fprintf('🔚 Training Final Classifier...\n');
    X_combined = [wideFeatures, cnnFeatures];
    Y = double(sampledData{:,2});
    [finalWeights, clfInfo] = trainFinalClassifier(X_combined, Y, clfParams);

    % Evaluate
    logits = X_combined * finalWeights;
    probs = 1 ./ (1 + exp(-logits));
    preds = probs > clfParams.threshold;

    acc = mean(preds == Y);
    precision = sum((preds == 1) & (Y == 1)) / max(sum(preds == 1), 1);
    recall = sum((preds == 1) & (Y == 1)) / max(sum(Y == 1), 1);
    f1 = 2 * (precision * recall) / max((precision + recall), 1e-6);

    % Output all info
    modelInfo = struct();
    modelInfo.wideNet = wideNet;
    modelInfo.deepNet = deepNet;
    modelInfo.finalWeights = finalWeights;
    modelInfo.accuracy = acc;
    modelInfo.precision = precision;
    modelInfo.recall = recall;
    modelInfo.f1 = f1;
    modelInfo.wideInfo = wideInfo;
    modelInfo.deepInfo = deepInfo;
    modelInfo.clfInfo = clfInfo;

    fprintf('✅ Final Results: Accuracy = %.4f, Precision = %.4f, Recall = %.4f, F1 = %.4f\n', ...
        acc, precision, recall, f1);
end
function cnnFeatures = trainDeepComponentWithPeriodicKernels(sampledData, savePath)
    if nargin < 2
        savePath = '';
    end

    startCol = 3;  % Assuming columns: 1=ID, 2=label
    X = table2array(sampledData(:, startCol:end));
    labels = double(sampledData{:, 2});
    [nSamples, nDays] = size(X);
    numWeeks = floor(nDays / 7);
    X = X(:, 1:(numWeeks * 7));

    cnnFeatures = zeros(nSamples, 7 * numWeeks);

    for i = 1:nSamples
        usage1D = X(i, :);
        usage2D = reshape(usage1D, [7, numWeeks]);
        featureMap = applyPeriodicKernels(usage2D);
        cnnFeatures(i, :) = featureMap(:)';
    end

    if ~isempty(savePath)
        writematrix(cnnFeatures, savePath);
        fprintf('✅ Periodic CNN features saved to %s\n', savePath);
    end
end

function C = applyPeriodicKernels(V)
    [d, m] = size(V);  % d=7, m=weeks
    Vpad = padarray(V, [1 1], 'replicate');
    C = zeros(d, m);
    for i = 2:(d+1)
        for j = 2:(m+1)
            block = Vpad(i-1:i+1, j-1:j+1);
            g1val = g1(block);
            g2val = g2(block);
            val = tanh(sum(g1val(:) + g2val(:)));
            C(i-1, j-1) = val;
        end
    end
end

function out = g1(B)
    out = [
        2*B(1,:) - B(2,:) - B(3,:);
        2*B(2,:) - B(1,:) - B(3,:);
        2*B(3,:) - B(1,:) - B(2,:)
    ];
end

function out = g2(B)
    out = g1(B')';
end
