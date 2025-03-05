% Clear all variables, close all figures, and clear the command window
clear all
close all
clc


% Set the figure size and font size
figSize = [10, 10];  % Figure size in inches [width, height]
fontSize = 12;      % Font size for labels, titles, and legends

fileName{1} = 'TBMORE_SW_FFT_1';
fileName{2} = 'TBMORE_SW_FFT_2';
fileName{3} = 'TBMORE_SW_FFT_3';
fileName{4} = 'TBMORE_SW_FFT_4';
fileName{5} = 'TBMORE_SW_FFT_5';
fileName{6} = 'TBMORE_SW_FFT_6';
fileName{7} = 'TBMORE_SW_FFT_7';

% Initialize arrays to store FID data
freq = [];
numFiles = length(fileName);
signalStrength = [];

baselineCorrectionValues = zeros(1, numFiles);

for i = 1:numFiles
    fftData = dlmread(fileName{i});
    % Extract the columns
    freq = fftData(:, 1);

    %baseline correction so no values of the signal are negative. Done by
    %taking the most negative value in one of the files and shifting the
    %rest of the files by that 
    %baselineCorrectionValues(i) = abs(min(fftData(:,2)));

    signalStrength{i} = (fftData(:, 2)); 
end

%baselineCorrection = max(baselineCorrectionValues)

% Apply baseline correction to all elements in signalStrength
%for i = 1:numFiles
%    signalStrength{i} = signalStrength{i} + baselineCorrection;
%end

% Initially plotting the FID data 
figure('Units', 'inches', 'Position', [0, 0, figSize]);
hold on;
for i = 1:numFiles
    plot(freq, signalStrength{i}, 'o-', LineWidth=1.5);
end

% Add labels and title
xlabel('Frequency (Hz)', 'FontSize', fontSize);
ylabel('Amplitude', 'FontSize', fontSize);
title('FFT Plot From SpinWorks', 'FontSize', fontSize);

% Create legend entries
legendEntries = cell(1, numFiles);
for i = 1:numFiles
    legendEntries{i} = sprintf('FFT %d', i);
end

% Add legend
legend(legendEntries, 'FontSize', fontSize);


% Prompt the user to select N peaks visually using ginput, multiplying by
% three because each peak as a left and righthand background region we also
% want to specify 
N = input('Enter the number of peaks to select: ')*3;

% Initialize the peakLocations array
peakLocations = zeros(N, 3);

disp('When selecting peaks: first pick the peak itself, then the left and right background regions')

% Loop to allow user to select N peaks
for i = 1:N
    % Prompt user to zoom in and select peak
    disp(['Zoom in and select peak ', num2str(i), ' using ginput']);
    
    % Dynamically adjust plot limits based on zoom selection
    xlim('auto');
    ylim('auto');
    zoom on;
    pause; % Wait for user to zoom in
    
    % Prompt user to click on the peak
    disp(['Select peak ', num2str(i), ' using ginput']);
    [x, ~] = ginput(1);
    
    % Store the first frequency value
    peakLocations(i, 2) = x;
    
    % Prompt user to click on the second frequency value
    disp(['Select second frequency value for peak ', num2str(i), ' using ginput']);
    [x, ~] = ginput(1);
    
    % Store the second frequency value
    peakLocations(i, 3) = x;
    
    % Store the peak number
    peakLocations(i, 1) = i;
end

% Display the peakLocations array
disp('Selected peak locations:');
disp(peakLocations);

% Display the peakLocations array
disp('Selected peak locations:');
disp(peakLocations);

%ft = 0.1
ft = abs(freq(1)-freq(2))*2 %defining frequency step as 3 times the spacing between two individual frequencies

% Set grid on
grid on;
numFrequencyBinIntervals = abs((freq(end) - freq(1)) / ft); %here we define the number of frequency intervals across the FFT  
current_interval = 0; 
current_values = 0; 

interval_values = []; 
riemannSumAcrossSamples = []; 
riemannSumArray = []
final = []; 
sampleTimeFrequencyDataArrray = [];

%this nested for loop identifies a frequency bin for each sample and computes the Riemann sum and correlates it to a
%frequency value (this frequency value is simply the center of the Riemann)
disp(freq(1))
disp(freq(end))
subplot_count = 0; 
for j = 1:numFiles % for each file
    riemannSumValues = []; 
    for i = freq(end):ft:freq(1)
        current_leftpoint = i;
        current_rightpoint = i + ft;
        current_interval = [num2str(current_leftpoint) ' ', num2str(current_rightpoint)];
        indices = find(freq >= current_leftpoint & freq <= current_rightpoint);

        % now, in the selected FFT region we create an array that holds all of the signal strengths
        %values_fft = signalStrength{1, j}{:, 1};
        values_fft = signalStrength{j};
        values_in_interval = values_fft(indices); 

        % now, we can calculate the number of rectangles within a frequency bin
        numPointsInInterval = length(values_in_interval);

        totalArea = 0;
        % compute the Riemann sum for within the accepted frequency window
        for i = 1:length(values_in_interval)
            %widthOfRectangle = ft / numPointsInInterval;
            %areaOfRectangle = widthOfRectangle * values_in_interval(i);
            areaOfRectangle = values_in_interval(i);
            totalArea = totalArea + areaOfRectangle;
        end
        % we need to assign this Riemann sum with a corresponding frequency point, we'll choose the centermost value
        centerOfValues = ceil(numPointsInInterval / 2);

        riemannFrequency = freq(indices(centerOfValues));

        riemannSumValues = [riemannSumValues; {riemannFrequency, totalArea}];

        %%Movie to plot the current interval and its Riemann value
%         plot(freq(indices), values_fft(indices)); % plot the interval
%         hold on;
%         plot(riemannFrequency, totalArea, 'ro'); % plot the maximum value
%         hold off;
%         % Set labels and title for the plot
%         xlabel('Frequency');
%         ylabel('Amplitude');
%         title(['Riemann Sum vs Frequency Position']);
%         % Pause to observe each plot
%         pause(0.01);

    end
    riemannSumAcrossSamples = [riemannSumAcrossSamples; {j, riemannSumValues}]
end


%now we can compute the average signal strength across each FFT 
averageRiemannSumArray = [];
for j = 1:length(riemannSumValues)
    storeRiemannValue = 0; % Initialize storeRiemannValue as a scalar

    for i = 1:numFiles
        riemannFFT = riemannSumAcrossSamples{i, 2}; % Accessing data within the cell array
        riemannValueInFFT = riemannFFT{j, 2}; % Accessing specific elements within the cell array

        storeRiemannValue = storeRiemannValue + riemannValueInFFT;
    end

    averageRiemannValue = storeRiemannValue / numFiles; % Compute average
    averageRiemannSumArray(j,1) = averageRiemannValue; % Store the average in the array
end

SDOMExpected = 0; 
getMean = 0; 
SDOMDifferenceSquared = 0; 
SDOMArray = zeros(ceil(numFrequencyBinIntervals), 1); % Initialize SDOMArray with zeros

for j = 1:ceil(numFrequencyBinIntervals) % Iterate over all elements of SDOMArray
    SDOMSummation = 0; % Initialize the summation variable
    getMean = averageRiemannSumArray(j,1);
    for i = 1:numFiles % Assuming numFiles is the total number of files
        SDOMExpected = cell2mat(riemannSumAcrossSamples{i,2}(j,2));
        SDOMDifferenceSquared = (getMean - SDOMExpected)^2;
        SDOMSummation = SDOMSummation + SDOMDifferenceSquared; % Accumulate the squared differences
    end
    SDOMArray(j,1) = (SDOMSummation / numFiles).^0.5; % Calculate the SDOM for the current frequency bin interval
end

finalArray = [];
finalArray = [finalArray; cell2mat(riemannSumValues(:,1)) ,averageRiemannSumArray, SDOMArray]

%now, to make our life easier - we will return the arrays for each peak
%region we selected

signalStrengthInEachPeak = [];

for k = 1:length(peakLocations(:, 1))

    %resetting all necessary arrays 
    signal_strength_in_interval = []; 
    uncert_signal_strength_in_interval = []; 

    % array of frequency regions specified in the finalArray
    frequency_region = finalArray(:,1);

    % matching the user selected regions to an actual frequency region in our finalArray
    getLeftUserFreqRegion = peakLocations(k,2);
    getRightUserFreqRegion = peakLocations(k,3);
    current_freq_interval = [num2str(getLeftUserFreqRegion) ' ', num2str(getRightUserFreqRegion)];
    getIndices = find(frequency_region >= getLeftUserFreqRegion & frequency_region <= getRightUserFreqRegion);

    % now get all of our uncertainties and signal strengths within our indices
    signalStrengthArray = finalArray(:,2);
    signal_strength_in_interval = signalStrengthArray(getIndices);
    
    %we can simply sum the signal strength in the interval
    signal_strength_in_interval = sum(signal_strength_in_interval)

    uncertaintyStrengthArray = finalArray(:,3)
    uncert_signal_strength_in_interval = uncertaintyStrengthArray(getIndices);

    % adding uncertainties in quadrature
    uncert_signal_strength_in_interval = sqrt(sum(uncert_signal_strength_in_interval.^2));

    signalStrengthInEachPeak = [signalStrengthInEachPeak; k, {signal_strength_in_interval}, {uncert_signal_strength_in_interval}];
end

