function [XTrain, YTrain, XVal, YVal] = splitOrderedData(data, labels, trainProportion, numClasses)

    XTrain = [];
    YTrain = [];

    XVal = [];
    YVal = [];

    % Assuming equal number of samples per class
    classSize = size(data, 3) / numClasses;
    endTrainIdx = ceil(trainProportion * classSize);

    disp(classSize);

    for i = 1:numClasses
        startIdx = (i-1)*classSize + 1;
        endIdx  = i*classSize;

        XTrain = cat(3, XTrain, data(:, :, startIdx:startIdx + endTrainIdx - 1));
        YTrain = cat(1, YTrain, labels(startIdx:startIdx + endTrainIdx - 1, 1));

        XVal = cat(3, XVal, data(:, :, startIdx + endTrainIdx:endIdx));
        YVal = cat(1, YVal, labels(startIdx + endTrainIdx:endIdx, 1));

    end 