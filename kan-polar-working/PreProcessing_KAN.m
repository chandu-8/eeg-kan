clear;
input_path = "C:\Users\aaron\OneDrive\Desktop\Researcg\Test Subject\EEG Data Subject-1\EEG Data Subject-1";
output_path = "C:\Users\aaron\OneDrive\Desktop\Researcg\Test Subject Preprocessed";
if ~exist(output_path, 'dir')
    mkdir(output_path);
end
Files = dir(fullfile(input_path, '*.mat'));

% Bandpass Filter Design       
OSR = 250;
p = 1;
q = 2;
DSR = OSR * (p / q);
d = fdesign.bandpass(8.1, 8.4, 12, 12.3, 40, 0.2, 40, OSR);
Hd = design(d);

% Process each file individually and save the result
for i = 1:length(Files)
    filename = Files(i).name;
    data_received = load(fullfile(input_path, filename));
    Data = data_received.data_received;

    % Generate biometric template
    Template_1 = template_generation(Data, Hd, p, q);

    % Save preprocessed template for the file in the output path
    output_filename = fullfile(output_path, strcat("Processed_", filename));
    save(output_filename, 'Template_1');
end

%%---------------------------------------------------------%%

function [EC_Template_1] = template_generation(data_received_d1, Hd, p, q)
    Data_1 = data_received_d1(1:45000, [6 7 9 10 14 15 16]);
    EC_Data_1 = Data_1(1:10000, :)';
    EO_Data_1 = Data_1(20001:30000, :)';
    MI_RH_Data_1 = Data_1(32001:42000, :)';

    [EC_PreP_1, DS_EC_PreP_1] = PreProcessing(EC_Data_1, Hd, p, q);
    
    temp_EC_Template_1 = reshape(DS_EC_PreP_1(:,:), [7, 125, 40]);
    EC_Template_1 = temp_EC_Template_1(:, 1:78, :);
end

%------------------------- Data Pre-Processing -----------------------%%

function [temp_EEG, Downsampled_EEG] = PreProcessing(temp_EEG, Hd, p, q)
    % Spectral Filtering 0.4-40 Hz
    nchan = 7;
    temp_EEG(1:nchan, :) = filtfilt(Hd.Numerator, 1, temp_EEG(1:nchan, :)')';

    % Z-score Normalization
    temp_EEG(1:nchan, :) = reshape(zscore(reshape(temp_EEG(1:nchan, :), [], 1)), nchan, []);

    % DeTrend
    temp_EEG(1:nchan, :) = (detrend(temp_EEG(1:nchan, :)'))';

    % Downsampling to 125 Hz
    Downsampled_EEG(1:nchan, :) = resample(temp_EEG(1:nchan, :)', p, q, 50)';
end
