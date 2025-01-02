%% Create labels vector

k=1;
for j=1:21
    for i=1:40
        lab_train(k,1) = j ;
        k=k+1;
    end
end

labels = transpose(lab_train);

%% Data Engineering

shuffled_indices = randperm(840);
x_train_shuffled = x_train(:, :, shuffled_indices);
labels_shuffled = labels(shuffled_indices);

disp(size(x_train_shuffled));
disp(size(labels_shuffled));

x_train_shuffled = reshape(x_train_shuffled, [7, 78, 1, 840]);


[trainIdx, valIdx] = dividerand(size(x_train_shuffled, 4), 0.8, 0.2);


XTrain = x_train_shuffled(:, :, :, trainIdx);
YTrain = labels_shuffled(trainIdx);
YTrain = categorical(YTrain);

XVal = x_train_shuffled(:, :, :, valIdx);
YVal = labels_shuffled(valIdx);
YVal = categorical(YVal);


disp(size(XTrain));
disp(size(YTrain));
disp(size(XVal));
disp(size(YVal));

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
    
    dropoutLayer(0.15)

    fullyConnectedLayer(numel(unique(labels)), 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% Train the network

net = trainNetwork(XTrain, YTrain, layers, options);

%% Validation

YPred = classify(net, XVal);
accuracy = sum(YPred == YVal) / numel(YVal);
disp(['Validation Accuracy: ', num2str(accuracy)]);
