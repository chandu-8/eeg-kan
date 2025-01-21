%% Create labels vector

k=1;
for j=1:21
    for i=1:40
        lab_train(k,1) = j ;
        lab_test(k,1) = j ;
        k=k+1;
    end
end

% labels = transpose(lab_train);
labels = lab_train;
% labels_test = transpose(lab_test);
labels_test = lab_test;

%% Data Engineering

shuffled_indices = randperm(840);
x_train_shuffled = x_train(:, :, shuffled_indices);
labels_shuffled = labels(shuffled_indices);

disp(size(x_train_shuffled));
disp(size(labels_shuffled));

x_train_shuffled = reshape(x_train_shuffled, [7, 78, 1, 840]);
x_test = reshape(x_test, [7, 78, 1, 840]);

[trainIdx, valIdx] = dividerand(size(x_train_shuffled, 4), 0.8, 0.2);


XTrain = x_train_shuffled(:, :, :, trainIdx);
YTrain = labels_shuffled(trainIdx);
YTrain = categorical(YTrain);

XVal = x_train_shuffled(:, :, :, valIdx);
YVal = labels_shuffled(valIdx);
YVal = categorical(YVal);

XTest = x_test;
YTest = categorical(labels_test);

disp(size(XTrain));
disp(size(YTrain));
disp(size(XVal));
disp(size(YVal));
disp(size(XTest));
disp(size(YTest));

%% Define network architecture and set training options

layers = [
    imageInputLayer([size(x_train_shuffled, 1), size(x_train_shuffled, 2), 1], 'Name', 'input')
    
    convolution2dLayer([3, 13], 128, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    
    maxPooling2dLayer([2, 2], 'Stride', 2, 'Name', 'maxpool1')
    
    convolution2dLayer([2, 13], 256, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    
    maxPooling2dLayer([1, 2], 'Stride', 1, 'Name', 'maxpool2')

    convolution2dLayer([1, 10], 512, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')

    
    fullyConnectedLayer(512, 'Name', 'fc1')
    reluLayer('Name', 'relu4')
    
    % dropoutLayer(0.15)

    fullyConnectedLayer(numel(unique(labels)), 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
];

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.00001, ...
    'LearnRateSchedule', 'polynomial', ...
    'MaxEpochs', 35, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'Metrics','accuracy');

%% Train the network

net = trainnet(XTrain, YTrain, layers, 'crossentropy', options);

%% Validation

% YPred = classify(net, XVal);
% valid_accuracy = sum(YPred == YVal.') / numel(YVal.');
% disp(['Validation Accuracy: ', num2str(valid_accuracy)]);

XVal_dl = dlarray(XVal, 'SSCB');
YPredProb = predict(net, XVal_dl);
[~, YPredLabels] = max(extractdata(YPredProb), [], 1);
YPredLabels = categorical(YPredLabels);
accuracy = sum(YPredLabels == YVal.') / numel(YVal.');
disp(['Validation Accuracy: ', num2str(accuracy * 100), '%']);

%% Test

% YPred_test = classify(net, XTest);
% test_accuracy = sum(YPred_test == YTest.') / numel(YTest.');
% disp(['Test Accuracy: ', num2str(test_accuracy)]);

XTest_dl = dlarray(XTest, 'SSCB');
YPredProb_test = predict(net, XTest_dl);
[~, YPredLabels_test] = max(extractdata(YPredProb_test), [], 1);
YPredLabels_test = categorical(YPredLabels_test);
test_accuracy = sum(YPredLabels_test == YTest.') / numel(YTest.');
disp(['Test Accuracy: ', num2str(test_accuracy * 100), '%']);