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

% shuffled_indices = randperm(840);
% x_train_shuffled = x_train(:, :, shuffled_indices);
% labels_shuffled = labels(shuffled_indices);
% 
% disp(size(x_train_shuffled));
% disp(size(labels_shuffled));
% 
% x_train_shuffled = reshape(x_train_shuffled, [7, 78, 1, 840]);
% x_test = reshape(x_test, [7, 78, 1, 840]);
% 
% [trainIdx, valIdx] = dividerand(size(x_train_shuffled, 4), 0.8, 0.1, 0.1);
% 
% 
% XTrain = x_train_shuffled(:, :, :, trainIdx);
% YTrain = labels_shuffled(trainIdx);
% YTrain = categorical(YTrain);
% 
% XVal = x_train_shuffled(:, :, :, valIdx);
% YVal = labels_shuffled(valIdx);
% YVal = categorical(YVal);
% 
% XTest = x_test;
% YTest = categorical(labels_test);
% 
% disp(size(XTrain));
% disp(size(YTrain));
% disp(size(XVal));
% disp(size(YVal));
% disp(size(XTest));
% disp(size(YTest));

XTrain = [];
YTrain = [];
XVal = [];
YVal = [];

XTest = x_test;
YTest = labels_test;

[XTrain, YTrain, XVal, YVal] = splitOrderedData(x_train, labels, 0.8, 21);

shuffled_indices = randperm(size(XTrain, 3));
XTrain = XTrain(:, :, shuffled_indices);
XTrain = reshape(XTrain, [size(XTrain, 1), size(XTrain, 2), 1, size(XTrain, 3)]);
YTrain = YTrain(shuffled_indices);
YTrain = categorical(YTrain);

shuffled_indices_val = randperm(size(XVal, 3));
XVal = XVal(:, :, shuffled_indices_val);
XVal = reshape(XVal, [size(XVal, 1), size(XVal, 2), 1, size(XVal, 3)]);
YVal = YVal(shuffled_indices_val);
YVal = categorical(YVal);

XTest = reshape(x_test, [size(XTest, 1), size(XTest, 2), 1, size(XTest, 3)]);
YTest = categorical(labels_test);

disp(size(XTrain));
disp(size(YTrain));
disp(size(XVal));
disp(size(YVal));
disp(size(XTest));
disp(size(YTest));
%% Define network architecture and set training options

layers = [
    imageInputLayer([size(XTrain, 1), size(XTrain, 2), 1], 'Name', 'input')
    
    convolution2dLayer([3, 3], 78, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')

    convolution2dLayer([3, 3], 156, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu2')


    maxPooling2dLayer([2, 2], 'Stride', 2, 'Name', 'maxpool1')
    
    convolution2dLayer([1, 11], 256, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu3')
    convolution2dLayer([1, 11], 256, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu4')
    
    maxPooling2dLayer([1, 2], 'Stride',[1,2], 'Name', 'maxpool2')


    convolution2dLayer([1, 5], 512, 'Padding', 'same', 'Name', 'conv5')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu5')
    convolution2dLayer([1, 5], 1024, 'Padding', 'same', 'Name', 'conv6')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu6')

    
    fullyConnectedLayer(1024, 'Name', 'fc1')
    reluLayer('Name', 'relu6')
    fullyConnectedLayer(2048, "Name", 'fc2')
    reluLayer("Name", 'relu7')

    
    dropoutLayer(0.15)

    
    fullyConnectedLayer(numel(unique(labels)), 'Name', 'fc3')
    softmaxLayer('Name', 'softmax')
];

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.000001, ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 5, ...
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