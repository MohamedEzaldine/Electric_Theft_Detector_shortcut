function results = trainWideAndDeepEndToEnd(sampledData, config)
% trainWideAndDeepEndToEnd - Train and evaluate a Deep & Wide model pipeline
% Inputs:
%   sampledData   - Table containing the input dataset with features and labels
%   config        - Struct containing configuration options
% Outputs:
%   results       - Struct containing evaluation metrics (AUC, MAP@100, MAP@200)

    % === Dataset Splitting ===
    fprintf('\n🔄 Splitting Dataset into Training, Validation, and Test Sets...\n');
    n = height(sampledData);
    idx = randperm(n);
    trainIdx = idx(1:round(0.7 * n));
    valIdx = idx(round(0.7 * n) + 1:round(0.85 * n));
    testIdx = idx(round(0.85 * n) + 1:end);
    
    trainData = sampledData(trainIdx, :);
    valData = sampledData(valIdx, :);
    testData = sampledData(testIdx, :);

    % === Set Defaults for Config ===
    if nargin < 2, config = struct(); end
    if ~isfield(config, 'wideLR'), config.wideLR = 0.005; end
    if ~isfield(config, 'wideNeurons'), config.wideNeurons = 10; end
    if ~isfield(config, 'deepLR'), config.deepLR = 0.005; end
    if ~isfield(config, 'deepFilters'), config.deepFilters = 8; end
    if ~isfield(config, 'patience'), config.patience = 5; end
    if ~isfield(config, 'l2Lambda'), config.l2Lambda = 5e-5; end

    %% Step 1: Train Wide Component
    fprintf('\n🚀 Training Wide Component...\n');
    [net, ~, ~, ~, ~, ~, wideOutputTrain] = trainWideComponentDL(trainData, ...
        config.wideLR, config.wideNeurons, config.patience, config.l2Lambda);
    fprintf('✅ Wide Component Training Complete. Output Size: [%d × %d]\n', size(wideOutputTrain));

    %% Step 2: Train Deep Component
    fprintf('\n🚀 Training Deep Component...\n');
    [cnnFeaturesTrain, trainedNet, ~, ~, ~, ~] = trainDeepComponentCNN(trainData, ...
        config.deepLR, config.deepFilters, config.patience);
    fprintf('✅ Deep Component Training Complete. CNN Feature Size: [%d × %d]\n', size(cnnFeaturesTrain));

    %% Step 3: Extract Periodic Features
    fprintf('\n⚙️ Extracting Periodic Features...\n');
    cnnPeriodicFeaturesTrain = trainDeepComponentWithPeriodicKernels(trainData);
    fprintf('✅ Periodic Feature Extraction Complete. Size: [%d × %d]\n', size(cnnPeriodicFeaturesTrain));

    %% Step 4: Train Logistic Regression Classifier
    fprintf('\n🚀 Training Logistic Regression Classifier...\n');
    labelsTrain = double(trainData{:, 2});

    % Concatenate features
    X_combinedTrain = [wideOutputTrain', cnnFeaturesTrain, cnnPeriodicFeaturesTrain];
    YTrain = labelsTrain(:);

    % Normalize features
    X_combinedTrain = normalize(X_combinedTrain);

    % Add bias term
    [m, n] = size(X_combinedTrain);
    XTrain = [ones(m, 1), X_combinedTrain];
    theta = zeros(n+1, 1);

    % Hyperparameters
    alpha = 0.01;
    numIters = 500;
    lambda = 0.01;
    lossHistory = zeros(numIters, 1);

    % Logistic regression training
    for iter = 1:numIters
        z = XTrain * theta;
        h = 1 ./ (1 + exp(-z));

        % Class imbalance weights
        weight = ones(size(YTrain));
        weight(YTrain == 1) = 1.0 + (1 - mean(YTrain));

        % Regularized cost
        reg = lambda * sum(theta(2:end).^2);
        loss = -mean(weight .* (YTrain .* log(h + 1e-6) + (1 - YTrain) .* log(1 - h + 1e-6))) + reg;
        lossHistory(iter) = loss;

        % Gradient
        error = h - YTrain;
        grad = (XTrain' * (weight .* error)) / m;
        grad(2:end) = grad(2:end) + lambda * theta(2:end);  % Regularize

        theta = theta - alpha * grad;
    end

    % Plot training loss
    figure;
    plot(lossHistory, 'LineWidth', 2);
    xlabel('Iteration'); ylabel('Loss');
    title('Logistic Regression Training Loss'); grid on;
    fprintf('✅ Logistic Regression Training Complete.\n');

    %% Step 5: Evaluate Metrics on Test Data
    fprintf('\n🔍 Evaluating Model Metrics on Test Data...\n');

    % Generate wideOutput, cnnFeatures, and cnnPeriodicFeatures for test data
        wideOutputTest = predictWideComponent(net, testData);
        config.wideLR, config.wideNeurons, config.patience, config.l2Lambda);

        cnnFeaturesTest = predictDeepComponent(trainedNet, testData);
        config.deepLR, config.deepFilters, config.patience);

    cnnPeriodicFeaturesTest = trainDeepComponentWithPeriodicKernels(testData);

    % Debugging dimensions before flattening
    fprintf('Size of wideOutputTest before flattening: [%d × %d]\n', size(wideOutputTest));
    fprintf('Size of cnnFeaturesTest before flattening: [%d × %d]\n', size(cnnFeaturesTest));
    fprintf('Size of cnnPeriodicFeaturesTest before flattening: [%d × %d]\n', size(cnnPeriodicFeaturesTest));

    % Ensure wideOutputTest is replicated for all test samples if necessary
    if size(wideOutputTest, 1) == 1 && size(wideOutputTest, 2) == 1
        wideOutputTest = repmat(wideOutputTest, size(cnnFeaturesTest, 1), 1);
    end

    % Flatten cnnFeaturesTest and cnnPeriodicFeaturesTest into 1D (row-wise)
    cnnFeaturesTestFlat = reshape(cnnFeaturesTest, size(cnnFeaturesTest, 1), []);
    cnnPeriodicFeaturesTestFlat = reshape(cnnPeriodicFeaturesTest, size(cnnPeriodicFeaturesTest, 1), []);

    % Debugging dimensions after flattening
    fprintf('Size of wideOutputTest after flattening: [%d × %d]\n', size(wideOutputTest));
    fprintf('Size of cnnFeaturesTest after flattening: [%d × %d]\n', size(cnnFeaturesTestFlat));
    fprintf('Size of cnnPeriodicFeaturesTest after flattening: [%d × %d]\n', size(cnnPeriodicFeaturesTestFlat));

    % Ensure all components have the same number of rows (samples)
    if size(wideOutputTest, 1) ~= size(cnnFeaturesTestFlat, 1)
        error('Mismatch: wideOutputTest must have the same number of rows as cnnFeaturesTestFlat.');
    end
    if size(wideOutputTest, 1) ~= size(cnnPeriodicFeaturesTestFlat, 1)
        error('Mismatch: wideOutputTest must have the same number of rows as cnnPeriodicFeaturesTestFlat.');
    end

    % Concatenate all features into a single matrix
    X_combinedTest = [wideOutputTest, cnnFeaturesTestFlat, cnnPeriodicFeaturesTestFlat];
    X_combinedTest = normalize(X_combinedTest);
    XTest = [ones(size(X_combinedTest, 1), 1), X_combinedTest];

    testLabels = double(testData{:, 2});
    probs = 1 ./ (1 + exp(-XTest * theta));
    preds = probs >= 0.5;

    accuracy = mean(preds == testLabels) * 100;

    [Xroc, Yroc, ~, AUC] = perfcurve(testLabels, probs, 1);
    map100 = computeMAPK(testLabels, probs, 100);
    map200 = computeMAPK(testLabels, probs, 200);

    fprintf('📈 AUC: %.4f | MAP@100: %.4f | MAP@200: %.4f\n', AUC, map100, map200);

    %% Return Results
    results = struct('AUC', AUC, 'MAP100', map100, 'MAP200', map200, 'Accuracy', accuracy);
end