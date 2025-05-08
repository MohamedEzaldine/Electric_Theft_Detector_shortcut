function mapk = computeMAPK(trueLabels, predictedScores, K)
    % computeMAPK - Compute Mean Average Precision at K (MAP@K)
    % Inputs:
    %   trueLabels      - Binary array of true labels (1 for positive, 0 for negative)
    %   predictedScores - Array of predicted scores (higher = more confident)
    %   K               - Number of top predictions to consider
    %
    % Output:
    %   mapk - Mean Average Precision at K

    % === Input Validation ===
    if length(trueLabels) ~= length(predictedScores)
        error('trueLabels and predictedScores must have the same length.');
    end
    if K <= 0
        error('K must be a positive integer.');
    end
    N = length(trueLabels);  % Total number of samples
    K = min(K, N);           % Ensure K does not exceed the number of samples

    % === Sort by Predicted Scores ===
    % Sort in descending order based on predicted scores
    [~, sortedIdx] = sort(predictedScores, 'descend');
    topK = sortedIdx(1:K);   % Get indices of top K predictions

    % === Compute Precision at Each Rank ===
    correct = trueLabels(topK);  % True labels of top K predictions
    numCorrect = sum(correct);   % Total number of correct labels in top K

    % Initialize variables for precision calculation
    precisionAtI = 0;
    numHits = 0;

    % Loop through top K predictions
    for i = 1:K
        if correct(i)  % If the i-th prediction is correct
            numHits = numHits + 1;  % Increment hit count
            precisionAtI = precisionAtI + numHits / i;  % Update precision
        end
    end

    % === Compute MAP@K ===
    if numCorrect == 0
        mapk = 0;  % No correct predictions
    else
        mapk = precisionAtI / min(K, numCorrect);  % Normalize by min(K, positives)
    end
end