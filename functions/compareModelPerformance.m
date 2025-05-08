function compareModelPerformance(myAUC, myMAP100, myMAP200)

    % === Define Paper Results ===
    models = {
        'TSR (1)',        0.5705, 0.5056, 0.5140;
        'TSR (2)',        0.5903, 0.5755, 0.5577;
        'LR (Logistic)',  0.6773, 0.6442, 0.5669;
        'SVM',            0.7183, 0.6862, 0.5919;
        'RF',             0.7317, 0.9078, 0.8670;
        'Wide',           0.6751, 0.8013, 0.7675;
        'CNN',            0.7636, 0.9059, 0.8835;
        'Wide & Deep CNN (paper)', 0.7760, 0.9404, 0.8961;
    };

    % === Append Your Model ===
    yourModel = {'Wide & Deep CNN (yours)', myAUC, myMAP100, myMAP200};
    allModels = [models; yourModel]

    % === Convert to Table ===
    T = cell2table(allModels, 'VariableNames', {'Model', 'AUC', 'MAP100', 'MAP200'});

    % === Plot ===
    figure('Name','AUC vs MAP','Position',[100 100 1400 600]);

    subplot(1,3,1);
    bar(categorical(T.Model), T.AUC, 'FaceColor','flat');
    title('📐 AUC');
    ylabel('Score'); grid on;
    xtickangle(45);

    subplot(1,3,2);
    bar(categorical(T.Model), T.MAP100, 'FaceColor','flat');
    title('📈 MAP@100');
    ylabel('Score'); grid on;
    xtickangle(45);

    subplot(1,3,3);
    bar(categorical(T.Model), T.MAP200, 'FaceColor','flat');
    title('📊 MAP@200');
    ylabel('Score'); grid on;
    xtickangle(45);

    sgtitle('⚡ Electricity Theft Detection - Model Comparison');
drawnow;
end
