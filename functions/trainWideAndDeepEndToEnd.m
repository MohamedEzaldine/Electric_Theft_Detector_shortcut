function results = trainWideAndDeepEndToEnd(sampledData, config, forceRecompute)
% trainWideAndDeepEndToEnd - Train and evaluate a Deep & Wide model pipeline
% Inputs:
%   sampledData   - Table containing the input dataset with features and labels
%   config        - Struct containing configuration options
%   forceRecompute - Boolean flag (true = recompute even if results exist)
% Outputs:
%   results       - Struct containing evaluation metrics (AUC, MAP@100, MAP@200)

    % === Check for Saved Results ===
    resultsFile = 'saved_results.mat';
    if exist(resultsFile, 'file') && ~forceRecompute
        fprintf('📂 Loading saved results from "%s"...\n', resultsFile);
        load(resultsFile, 'results'); % Load saved results
        disp('✅ Results loaded successfully:');
        disp(results);
        return;
    end

    % === If no saved results or forceRecompute is true, run the pipeline ===
    fprintf('🔄 Running the pipeline to compute results...\n');

    %% === Set Defaults ===
    if nargin < 2, config = struct(); end
    if ~isfield(config, 'wideLR'), config.wideLR = 0.005; end
    if ~isfield(config, 'wideNeurons'), config.wideNeurons = 10; end
    if ~isfield(config, 'deepLR'), config.deepLR = 0.005; end
    if ~isfield(config, 'deepFilters'), config.deepFilters = 8; end
    if ~isfield(config, 'patience'), config.patience = 5; end
    if ~isfield(config, 'l2Lambda'), config.l2Lambda = 5e-5; end

    %% Step 1: Train Wide Component
    fprintf('\n🚀 Training Wide Component...\n');
    [net, ~, ~, ~, ~, ~, wideOutput] = trainWideComponentDL(sampledData, ...
        config.wideLR, config.wideNeurons, config.patience, config.l2Lambda);
    writematrix(wideOutput, 'wide_output_features.csv');
    fprintf('✅ Wide output saved to "wide_output_features.csv" with size [%d × %d].\n', size(wideOutput));

    %% Step 2: Train Deep Component
    fprintf('\n🚀 Training Deep Component...\n');
    [cnnFeatures, trainedNet, ~, ~, ~, ~] = trainDeepComponentCNN(sampledData, ...
        config.deepLR, config.deepFilters, config.patience);
    writematrix(cnnFeatures, 'cnn_output_features.csv');
    save('trained_deep_component_net.mat', 'trainedNet');
    fprintf('✅ CNN features and trained deep network saved successfully.\n');

    %% Step 3: Extract Periodic Features
    fprintf('\n⚙️ Extracting Periodic Features...\n');
    cnnPeriodicFeatures = trainDeepComponentWithPeriodicKernels(sampledData, 'cnn_periodic_features.csv');
    fprintf('✅ Periodic features saved to "cnn_periodic_features.csv" with size [%d × %d].\n', size(cnnPeriodicFeatures));

    %% Step 4: Train Logistic Regression Classifier
    fprintf('\n🚀 Training Logistic Regression Classifier...\n');
    wideFeatures = readmatrix('wide_output_features.csv');
    cnnDeepFeatures = readmatrix('cnn_output_features.csv');
    cnnPeriodicFeat = readmatrix('cnn_periodic_features.csv');
    labels = double(sampledData{:, 2});

    % Concatenate features
    X_combined = [wideFeatures', cnnDeepFeatures, cnnPeriodicFeat];
    Y = labels(:);

    % Normalize features
    X_combined = normalize(X_combined);

    % Add bias term
    [m, n] = size(X_combined);
    X = [ones(m, 1), X_combined];
    theta = zeros(n+1, 1);

    % Hyperparameters
    alpha = 0.01;
    numIters = 500;
    lambda = 0.01;
    lossHistory = zeros(numIters, 1);

    % Logistic regression training
    for iter = 1:numIters
        z = X * theta;
        h = 1 ./ (1 + exp(-z));

        % Class imbalance weights
        weight = ones(size(Y));
        weight(Y == 1) = 1.0 + (1 - mean(Y));

        % Regularized cost
        reg = lambda * sum(theta(2:end).^2);
        loss = -mean(weight .* (Y .* log(h + 1e-6) + (1 - Y) .* log(1 - h + 1e-6))) + reg;
        lossHistory(iter) = loss;

        % Gradient
        error = h - Y;
        grad = (X' * (weight .* error)) / m;
        grad(2:end) = grad(2:end) + lambda * theta(2:end);  % Regularize

        theta = theta - alpha * grad;
    end

    % Save training loss plot
    figure;
    plot(lossHistory, 'LineWidth', 2);
    xlabel('Iteration'); ylabel('Loss');
    title('Logistic Regression Training Loss'); grid on;
    saveas(gcf, 'logistic_regression_loss.png');
    close(gcf);

    % Predictions and evaluation
    probs = 1 ./ (1 + exp(-X * theta));
    preds = probs >= 0.5;
    accuracy = mean(preds == Y) * 100;
    fprintf('✅ Final classifier accuracy: %.2f%%\n', accuracy);

    %% Step 5: Evaluate Metrics
    fprintf('\n🔍 Evaluating Model Metrics...\n');
    trueLabels = Y;
    predictedScores = probs;

    [Xroc, Yroc, ~, AUC] = perfcurve(trueLabels, predictedScores, 1);
    map100 = computeMAPK(trueLabels, predictedScores, 100);
    map200 = computeMAPK(trueLabels, predictedScores, 200);

    fprintf('📈 AUC: %.4f | MAP@100: %.4f | MAP@200: %.4f\n', AUC, map100, map200);

    % Save ROC curve
    figure;
    plot(Xroc, Yroc, 'b-', 'LineWidth', 2); grid on;
    xlabel('False Positive Rate'); ylabel('True Positive Rate');
    title(sprintf('ROC Curve (AUC = %.4f)', AUC));
    saveas(gcf, 'roc_curve.png');
    close(gcf);

    % === Save Results ===
    results = struct('AUC', AUC, 'MAP100', map100, 'MAP200', map200, 'Accuracy', accuracy);
    save(resultsFile, 'results');
    fprintf('✅ Results saved to "%s".\n', resultsFile);

    % === Compare Performance with Baselines ===
    compareModelPerformance(AUC, map100, map200);

    fprintf('\n✅ DeepWide Model Evaluation Complete:\n');
    fprintf('   - AUC      : %.4f\n', AUC);
    fprintf('   - MAP@100  : %.4f\n', map100);
    fprintf('   - MAP@200  : %.4f\n', map200);
end