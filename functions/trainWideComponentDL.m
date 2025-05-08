function [net, trainLossHistory, valLossHistory, neuronHistory, accHistory, f1History, wideOutput] = trainWideComponentDL(sampledData, initialLR, initialNeurons, patience, l2Lambda)
    % Wide Component - DL-based dynamic neuron, LR, patience, accuracy, F1 tracking, premium plotting

    %% Parameters
    if nargin < 5, l2Lambda = 0.00005; end
    if nargin < 4, patience = 5; end
    if nargin < 3, initialNeurons = 10; end
    if nargin < 2, initialLR = 0.005; end

    rng(1); % Reproducibility

    %% Prepare Data
    X = dlarray(table2array(sampledData(:, 3:end))', 'CB');  % [features × batch]
    Y = dlarray(double(sampledData{:,2})', 'CB');            % [labels × batch]

    nSamples = size(X,2);

    % Split train/validation
    valRatio = 0.2;
    idx = randperm(nSamples);
    valIdx = idx(1:round(valRatio*nSamples));
    trainIdx = idx(round(valRatio*nSamples)+1:end);

    XTrain = X(:, trainIdx);
    YTrain = Y(:, trainIdx);
    XVal = X(:, valIdx);
    YVal = Y(:, valIdx);

    % Use GPU if available
    if canUseGPU
        XTrain = gpuArray(XTrain);
        YTrain = gpuArray(YTrain);
        XVal = gpuArray(XVal);
        YVal = gpuArray(YVal);
    end

    %% Build Network
    numFeatures = size(XTrain,1);
    neuronCount = initialNeurons;

    layers = [
        featureInputLayer(numFeatures, 'Normalization', 'none', 'Name', 'input')
        fullyConnectedLayer(neuronCount, 'Name', 'fc1')
        batchNormalizationLayer('Name','bn1')
        reluLayer('Name','relu1')
    ];
    lgraph = layerGraph(layers);
    net = dlnetwork(lgraph);

    %% Training Setup
    bestValLoss = Inf;
    bestNet = net;
    trainLossHistory = [];
    valLossHistory = [];
    neuronHistory = [];
    accHistory = [];
    f1History = [];
    learnRate = initialLR;
    epochsWithoutImprovement = 0;
    epoch = 0;

    trailingAvg = [];
    trailingAvgSq = [];

    tic;
    fprintf('🚀 Training Wide Component...\n');

    %% Training Loop
    while true
        epoch = epoch + 1;

        % Forward pass
        YPredTrain = forward(net, XTrain);
        YPredTrainMean = mean(YPredTrain, 1);
        lossTrain = mse(YPredTrainMean, YTrain) + l2Lambda * sum(cellfun(@(w) sum(w(:).^2), net.Learnables.Value));

        % Gradients
        [gradients, state] = dlfeval(@modelGradients, net, XTrain, YTrain, l2Lambda);
        net.State = state;

        % Adam update
        [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, trailingAvg, trailingAvgSq, epoch, learnRate);

        % Validation
        YPredVal = forward(net, XVal);
        YPredValMean = mean(YPredVal, 1);
        valLoss = mse(YPredValMean, YVal);

        % Metrics
        preds = gather(extractdata(YPredValMean > 0.5));
        labels = gather(extractdata(YVal));
        acc = sum(preds == labels) / numel(labels);
        precision = sum((preds == 1) & (labels == 1)) / (sum(preds == 1) + eps);
        recall = sum((preds == 1) & (labels == 1)) / (sum(labels == 1) + eps);
        f1 = 2 * (precision * recall) / (precision + recall + eps);

        % Track History
        trainLossHistory(end+1) = double(gather(extractdata(lossTrain)));
        valLossHistory(end+1) = double(gather(extractdata(valLoss)));
        neuronHistory(end+1) = neuronCount;
        accHistory(end+1) = acc;
        f1History(end+1) = f1;

        % Dynamic Patience
        if valLoss < bestValLoss * 0.99
            bestValLoss = valLoss;
            bestNet = net;
            epochsWithoutImprovement = 0;
        else
            epochsWithoutImprovement = epochsWithoutImprovement + 1;
        end

        % Display Progress
        if mod(epoch, 1) == 0
            fprintf('Epoch %03d | TrainLoss: %.4f | ValLoss: %.4f | Acc: %.4f | F1: %.4f | Neurons: %d | LR: %.5f\n', ...
                epoch, trainLossHistory(end), valLossHistory(end), acc, f1, neuronCount, learnRate);
        end

        % Early Stopping
        if epochsWithoutImprovement >= patience
            fprintf('🛑 Early stopping at epoch %d\n', epoch);
            break;
        end

        % Dynamic Neuron Growth
        if epoch > 1 && trainLossHistory(end) > trainLossHistory(end-1)
            neuronCount = neuronCount + 5;
            layers = [
                featureInputLayer(numFeatures, 'Normalization', 'none', 'Name', 'input')
                fullyConnectedLayer(neuronCount, 'Name', 'fc1')
                batchNormalizationLayer('Name','bn1')
                reluLayer('Name','relu1')
            ];
            lgraph = layerGraph(layers);
            net = dlnetwork(lgraph);
            trailingAvg = [];
            trailingAvgSq = [];
            fprintf('📈 Added neurons: %d total neurons now\n', neuronCount);
        end

        % Learning Rate Decay
        if mod(epoch, 10) == 0
            learnRate = learnRate * 0.9;
        end
    end

    elapsed = toc;
    fprintf('✅ Training completed in %.2f seconds.\n', elapsed);

    %% Finalize
    net = bestNet;
    wideOutput = gather(extractdata(forward(net, dlarray(table2array(sampledData(:,3:end))', 'CB'))));

    %% Plot
    plotTrainingProgressPremium(trainLossHistory, valLossHistory, neuronHistory, accHistory, f1History);
end

%% Helper Functions
function [gradients, state] = modelGradients(net, X, Y, l2Lambda)
    [YPred, state] = forward(net, X);
    YPredMean = mean(YPred, 1);
    loss = mse(YPredMean, Y) + l2Lambda * sum(cellfun(@(w) sum(w(:).^2), net.Learnables.Value));
    gradients = dlgradient(loss, net.Learnables);
end

function plotTrainingProgressPremium(trainLoss, valLoss, neuronHistory, accHistory, f1History)
    epochs = 1:length(trainLoss);
    fig = figure('Position', [100, 100, 1600, 900]);
    
    %% Loss Plot
    subplot(3,1,1);
    plot(epochs, trainLoss, 'b-', 'LineWidth', 2); hold on;
    plot(epochs, valLoss, 'r--', 'LineWidth', 2);
    legend('Train Loss', 'Validation Loss', 'Location', 'northeastoutside', 'FontSize', 12);
    title('Loss Progress', 'FontSize', 18, 'FontWeight', 'bold');
    xlabel('Epoch', 'FontSize', 14); ylabel('Loss', 'FontSize', 14);
    grid on;

    %% Neuron Growth
    subplot(3,1,2);
    stairs(epochs, neuronHistory, 'g-', 'LineWidth', 2);
    title('Neuron Growth Over Epochs', 'FontSize', 18, 'FontWeight', 'bold');
    xlabel('Epoch', 'FontSize', 14); ylabel('Neurons', 'FontSize', 14);
    grid on;

    %% Accuracy and F1
    subplot(3,1,3);
    plot(epochs, accHistory, 'm-', 'LineWidth', 2); hold on;
    plot(epochs, f1History, 'c--', 'LineWidth', 2);
    legend('Accuracy', 'F1 Score', 'Location', 'northeastoutside', 'FontSize', 12);
    title('Validation Accuracy & F1 Score', 'FontSize', 18, 'FontWeight', 'bold');
    xlabel('Epoch', 'FontSize', 14); ylabel('Score', 'FontSize', 14);
    grid on;

    %% Save
    set(gcf, 'Color', 'w');
    exportgraphics(fig, 'wide_training_progress_premium.png', 'Resolution', 300);
    exportgraphics(fig, 'wide_training_progress_premium.pdf', 'ContentType', 'vector');
    close(fig);
end
