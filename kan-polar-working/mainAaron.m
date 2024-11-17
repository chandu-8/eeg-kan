%.   Kolmogorov-Arnold model for machine learning
%.   See (Poluektov and Polar, arXiv:2305.08194, May 2023)
%.   Code has been written by Michael Poluektov (University of Dundee, Department of Mathematical Sciences and Computational Physics)

clear variables;
close all;

%% Load preprocessed EEG data
% Directory containing the preprocessed data
dataDir = '/Users/hkonduru/Documents/Rig Das Research/eeg-kan/kan-polar-working/Test Subject Preprocessed';  % Update with your actual data path

% Get lists of all .mat files for Session 1 and Session 2
session1Files = dir(fullfile(dataDir, 'Processed_data_received_*_S1.mat'));
session2Files = dir(fullfile(dataDir, 'Processed_data_received_*_S2.mat'));

% Initialize storage for training data (Session 1)
x_train = [];
y_train = [];  % Replace with actual labels if available

% Load and combine all Session 1 data for training
for i = 1:length(session1Files)
    session1File = session1Files(i).name;
    session1Data = load(fullfile(dataDir, session1File));
    
    % Reshape the 3D matrix into a 2D matrix with samples as rows
  %  reshapedData = reshape(session1Data.Template_1, size(session1Data.Template_1, 1), []);
  %  x_train = [x_train; reshapedData];  % Concatenate data from each file
  if i==1
  x_train=cell2mat(struct2cell(session1Data));
  else
  x_train=cat(3,x_train,cell2mat(struct2cell(session1Data)));
  end



    % Replace the random placeholder with actual labels if available
    % Example: Use real EEG labels or calculated features
    % y_train = [y_train; session1Data.labels];  % Uncomment and use if labels are provided
end

% Normalize x_train to the range [0, 1]
%x_train = (x_train - min(x_train(:))) / (max(x_train(:)) - min(x_train(:)));

% Check size of x_train for verification
%disp(['Size of x_train: ', mat2str(size(x_train))]);

% Label records for training (1)
k=1;
for j=1:21
    for i=1:40
        lab_train(k,1) = j ;
        k=k+1;
    end
end
lab_train_2=transpose(lab_train);

save('data_label.mat', "lab_train","x_train");


% Set numerical parameters and limits
alp = 0.01;  % Adjusted learning rate for stability
lam = 0.001;  % Adjusted regularization parameter
Nrun = 50;  % Number of training iterations

% Ensure ymin and ymax are scalars
ymin = min(y_train(:));
ymax = max(y_train(:));

% Debug print to confirm ymin and ymax are scalars
disp(['ymin: ', num2str(ymin), ', ymax: ', num2str(ymax)]);

m = size(x_train, 2);  % Number of input features
n = 6;  % Number of nodes at the bottom
q = 6;  % Number of nodes at the top
p = 5;  % Number of bottom operators

% Print values of n, m, and p for verification
fprintf('n: %d, m: %d, p: %d\n', n, m, p);

%% Build K.-A.
tic;
[fnB0, fnT0] = buildKA_init(m, n, q, p, ymin, ymax);

% Set batch size for training
batchSize = 20;  % Adjust based on available memory and dataset size
numBatches = ceil(size(x_train, 1) / batchSize);

% Initialize tracking variables for RMSE
RMSE_train = zeros(Nrun, 1);
t_min_all_train = zeros(Nrun, p);
t_max_all_train = zeros(Nrun, p);

% Training loop with batch processing
for run = 1:Nrun
    disp(['Training iteration ', num2str(run), ' of ', num2str(Nrun)]);
    
    for batchIdx = 1:numBatches
        % Get batch indices
        startIdx = (batchIdx - 1) * batchSize + 1;
        endIdx = min(batchIdx * batchSize, size(x_train, 1));
        
        % Extract mini-batch data
        x_batch = x_train(startIdx:endIdx, :);
        y_batch = y_train(startIdx:endIdx, :);
        lab_batch = lab_train(startIdx:endIdx, :);
        
        % Train on mini-batch
        [yhat_batch, fnB, fnT, RMSE_batch, t_min_batch, t_max_batch] = solveMinGauss(x_batch, y_batch, lab_batch, 1, 1, alp, lam, 1, xmin, xmax, ymin, ymax, fnB0, fnT0);
        
        % Aggregate results (optional: store or average batch results)
        RMSE_train(run) = mean(RMSE_batch);
        t_min_all_train(run, :) = min(t_min_batch);
        t_max_all_train(run, :) = max(t_max_batch);
    end
    
    % Display iteration progress
    fprintf('Iteration %d completed. RMSE: %f\n', run, RMSE_train(run));
end

toc;
disp('Training complete.');

%% Test the Model on Session 2 Data
x_test = [];
y_test = [];  % Replace with actual labels if available

for i = 1:length(session2Files)
    session2File = session2Files(i).name;
    session2Data = load(fullfile(dataDir, session2File));
    
    % Reshape the 3D matrix into a 2D matrix with samples as rows
    reshapedData = reshape(session2Data.Template_1, size(session2Data.Template_1, 1), []);
    x_test = [x_test; reshapedData];  % Concatenate data from each file

    % Replace with actual test labels if available
    % y_test = [y_test; session2Data.labels];  % Uncomment and use if labels are provided
end

% Normalize x_test using the same scaling as x_train
x_test = (x_test - min(x_train(:))) / (max(x_train(:)) - min(x_train(:)));

% Label records for testing (2)
lab_test = ones(size(x_test, 1), 1) * 2;

% Precompute spline matrices for testing
Mn = splineMatrix(n);
Mq = splineMatrix(q);

% Test the trained model on Session 2 data
[yhat_test, ~, ~] = modelKA_basisC(x_test, xmin, xmax, ymin, ymax, fnB, fnT, Mn, Mq);

% Calculate and display test results
test_RMSE = sqrt(mean((yhat_test - y_test).^2)) / (ymax - ymin);
disp(['Test RMSE: ', num2str(test_RMSE)]);

%% Plot training RMSE
figure(1);
plot(log(RMSE_train) / log(10));
title('Training RMSE (Session 1)');
xlabel('Number of passes');
ylabel('log_{10}(RMSE)');

%% Plot min and max of intermediate variables for training
figure(2);
hold on;
for jj = 1:p
    plot(t_min_all_train(:, jj), 'b');
    plot(t_max_all_train(:, jj), 'r');
end
hold off;
xlabel('Number of passes');
ylabel('min/max of intermediate variables');
