function [cnnFeatures, trainedNet, trainLossHistory, trainAccHistory, filterHistory, lrHistory] = trainDeepComponentCNN(sampledData, initialLR, initialFilters, patience)

    if nargin < 4, patience = 5; end
    if nargin < 3, initialFilters = 8; end
    if nargin < 2, initialLR = 0.005; end

    maxFilters = 32;
    filterStep = 4;
    minLR = 1e-5;
    batchSize = 128;
    rng(1); % Reproducibility

    %% Prepare Data
    X = table2array(sampledData(:, 3:end));
    labels = double(sampledData{:, 2});   % Original labels: 0 or 1
    labels = labels + 1;                  % Convert to 1 or 2
    Y = categorical(labels);

    [nSamples, nDays] = size(X);
    numWeeks = floor(nDays / 7);
    X = X(:, 1:(numWeeks*7));
    X3D = reshape(X, [nSamples, 7, numWeeks]);
    XInput = permute(X3D, [2, 3, 4, 1]); % [7, weeks, 1, samples]

    if canUseGPU
        XInput = gpuArray(single(XInput));
    else
        XInput = single(XInput);
    end

    %% Define Network Builder
    buildNet = @(filters) layerGraph([
        imageInputLayer([7 numWeeks 1], 'Normalization','none','Name','input')
        convolution2dLayer(3, filters, 'Padding','same','WeightsInitializer','he','Name','conv')
        batchNormalizationLayer('Name','bn')
        reluLayer('Name','relu')
        dropoutLayer(0.3, 'Name','drop')
        maxPooling2dLayer(2, 'Stride',2, 'Name','pool')
        fullyConnectedLayer(10, 'Name','fc')
        reluLayer('Name','relu_fc')
        fullyConnectedLayer(2, 'Name','logits')
        softmaxLayer('Name','softmax')
    ]);

    %% Initialize Network
    currentFilters = initialFilters;
    dlnet = dlnetwork(buildNet(currentFilters));
    trailingAvg = [];
    trailingAvgSq = [];

    %% Tracking
    trainLossHistory = [];
    trainAccHistory = [];
    filterHistory = [];
    lrHistory = [];
    bestLoss = inf;
    learnRate = initialLR;
    epochsNoImprovement = 0;
    epoch = 0;

    fprintf('🚀 Training Deep Component...\n');
    tic;

    %% Training Loop
    while true
        epoch = epoch + 1;
        idx = randperm(nSamples);
        totalLoss = 0;
        totalCorrect = 0;

        for i = 1:batchSize:nSamples
            batchIdx = idx(i:min(i+batchSize-1, nSamples));
            XBatch = XInput(:, :, :, batchIdx);
            YBatch = Y(batchIdx);

            dlX = dlarray(XBatch, 'SSCB');
            if canUseGPU, dlX = gpuArray(dlX); end

            % Weighting for class imbalance
            theftRatio = mean(double(YBatch) == 2);
            weights = ones(size(YBatch), 'single');
            weights(double(YBatch) == 2) = 1 + (1 - theftRatio);

            [loss, gradients, state, probs] = dlfeval(@modelGradientsWeighted, dlnet, dlX, YBatch, weights);
            dlnet.State = state;

            [~, preds] = max(extractdata(probs), [], 1);
            totalCorrect = totalCorrect + sum(preds' == double(YBatch));
            totalLoss = totalLoss + double(gather(extractdata(loss)));

            [dlnet, trailingAvg, trailingAvgSq] = adamupdate(dlnet, gradients, ...
                trailingAvg, trailingAvgSq, epoch, learnRate);
        end

        avgLoss = totalLoss / ceil(nSamples / batchSize);
        avgAcc = totalCorrect / nSamples;

        trainLossHistory(end+1) = avgLoss;
        trainAccHistory(end+1) = avgAcc;
        filterHistory(end+1) = currentFilters;
        lrHistory(end+1) = learnRate;

        fprintf('Epoch %02d - Loss: %.4f - Acc: %.2f%% - Filters: %d\n', ...
            epoch, avgLoss, 100*avgAcc, currentFilters);

        % Early stopping
        if avgLoss < bestLoss * 0.99
            bestLoss = avgLoss;
            epochsNoImprovement = 0;
        else
            epochsNoImprovement = epochsNoImprovement + 1;
        end

        if epochsNoImprovement >= patience
            fprintf('🛑 Early stopping triggered at epoch %d\n', epoch);
            break;
        end

        % Grow filters if loss worsens
        if epoch > 1 && trainLossHistory(end) > trainLossHistory(end-1) && currentFilters < maxFilters
            currentFilters = currentFilters + filterStep;
            fprintf('📈 Growing filters to %d\n', currentFilters);
            dlnet = dlnetwork(buildNet(currentFilters));
            trailingAvg = []; trailingAvgSq = [];
        end

        % Learning rate decay
        if mod(epoch, 10) == 0
            learnRate = max(learnRate * 0.9, minLR);
        end
    end

    elapsed = toc;
    fprintf('✅ Training finished in %.2f seconds.\n', elapsed);

    %% Feature Extraction from relu_fc layer
    dlXAll = dlarray(XInput, 'SSCB');
    if canUseGPU, dlXAll = gpuArray(dlXAll); end
    layerOuts = forward(dlnet, dlXAll);
    features = extractdata(layerOuts);
    cnnFeatures = gather(features)';
    trainedNet = dlnet;

    %% Save Training Progress
    plotDeepTrainingProgress(trainLossHistory, trainAccHistory, filterHistory, lrHistory);
end
%% Helper: Weighted Gradients
function [loss, gradients, state, probs] = modelGradientsWeighted(dlnet, dlX, Tcat, rawWeights)
    [scores, state] = forward(dlnet, dlX);  % scores: [2 x batch]

    % Convert categorical T to one-hot numeric for 'independent' crossentropy
    TIdx = double(Tcat);  % 1 or 2
    numClasses = size(scores,1);
    batchSize = size(scores,2);

    T = zeros(numClasses, batchSize, 'single');
    for i = 1:batchSize
        T(TIdx(i), i) = 1;
    end
    T = dlarray(T, 'CB');  % Classes x Batch

    % Format weights
    weights = single(rawWeights(:));
    weights = dlarray(weights, 'B');

    % Compute loss
    loss = crossentropy(scores, T, ...
        'TargetCategories', 'independent', ...
        'Weights', weights, ...
        'WeightsFormat', 'B');

    gradients = dlgradient(loss, dlnet.Learnables);
    probs = scores;
end
%% Helper: Plot Training Progress
function plotDeepTrainingProgress(lossHistory, accHistory, filterHistory, lrHistory)
    epochs = 1:length(lossHistory);
    fig = figure('Position', [100, 100, 1400, 900]);
    subplot(4,1,1); semilogy(epochs, lossHistory, 'b', 'LineWidth', 2); title('Loss'); grid on;
    subplot(4,1,2); plot(epochs, accHistory * 100, 'g', 'LineWidth', 2); title('Accuracy'); grid on;
    subplot(4,1,3); stairs(epochs, filterHistory, 'm', 'LineWidth', 2); title('Filter Growth'); grid on;
    subplot(4,1,4); semilogy(epochs, lrHistory, 'r', 'LineWidth', 2); title('Learning Rate Decay'); grid on;
    saveas(fig, 'deep_training_progress.png'); close(fig);
end
