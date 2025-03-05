% Clear all variables, close all figures, and clear the command window
clear all
close all
clc
start_time = 0.01;
dt = 0.5;
ft = .1; %frequency bin slices of FFT. Mustn't be so small that Riemann rectangles do not fit within bin. >0.2

% Here, we are using the 'addNoiseToFIDFile' function to generate random
% noise on the user specified FID file. In actual experimental FFTSignalSpectraForTimeBin,
% multiple measurements must be taken with the same sample to get
% systematic error quantification.
%addNoiseToFIDFile('test.txt', 'test_N', .21)

% Set the figure size and font size
figSize = [10, 10];  % Figure size in inches [width, height]
fontSize = 12;      % Font size for labels, titles, and legends

fileName{1} = 'TB_Less_FID_Data_1.txt';
fileName{2} = 'TB_Less_FID_Data_2.txt';
fileName{3} = 'TB_Less_FID_Data_3.txt';
fileName{4} = 'TB_Less_FID_Data_4.txt';
fileName{5} = 'TB_Less_FID_Data_5.txt';
fileName{6} = 'TB_Less_FID_Data_6.txt';
fileName{7} = 'TB_Less_FID_Data_7.txt';

% Initialize arrays to store FID FFTSignalSpectraForTimeBin
timeFIDArray = [];
numMeasurementsOfSample = length(fileName);
fid = cell(1, numMeasurementsOfSample);
for i = 1:numMeasurementsOfSample
    % Prompt the user to specify the file name
    %fileName = '';
    % while ~exist(fileName, 'file')
    %     fileName = input(sprintf('Enter the file name for FID %d: ', i), 's');
    %     if ~exist(fileName, 'file')
    %         fprintf('File does not exist. Please try again.\n');
    %     end
    % end
    % 
    % Read the FID FFTSignalSpectraForTimeBin from the file

    fidData = dlmread(fileName{i});
    
    % Extract the columns
    timeFIDArray = fidData(:, 1);
    fid{i} = fidData(:, 2);
end

% Initially plotting the FID FFTSignalSpectraForTimeBin 
figure('Units', 'inches', 'Position', [0, 0, figSize]);
hold on;
for i = 1:numMeasurementsOfSample
    plot(timeFIDArray, fid{i}, 'LineWidth', 1.5);
end

% Add labels and title
xlabel('timeFIDArray (s)', 'FontSize', fontSize);
ylabel('Amplitude', 'FontSize', fontSize);
title('FID Plot', 'FontSize', fontSize);

% Create legend entries
legendEntries = cell(1, numMeasurementsOfSample);
for i = 1:numMeasurementsOfSample
    legendEntries{i} = sprintf('FID %d', i);
end

% Add legend
legend(legendEntries, 'FontSize', fontSize);

% Set grid on
grid on;

% Prompt the user to specify the timeFIDArray interval. This interval will
%dt = input('Enter the timeFIDArray interval (in seconds): ');

% Prompt the user to specify the start timeFIDArray for FFT. This is important
% because sometimes the beginning 0.01s of FID FFTSignalSpectraForTimeBin is erratic, and
% frustrates the initial timeFIDArray bin measurement 
%start_time = input('Enter the start timeFIDArray for FFT (in seconds): ');

% Compute the number of intervals based on dt and start_time
% This is the total number of timeFIDArray bins the FID will be sectisampleTimeFrequencyDataArrrayd off into 
numTimeBinIntervals = floor((max(timeFIDArray) - start_time) / dt);
fprintf('Number of time bin intervals is:', numTimeBinIntervals)

% Create a color map for distinguishing different FTs
colors = parula(numTimeBinIntervals);

% Compute the frequency axis
fs = 1 / (timeFIDArray(2) - timeFIDArray(1)); % Calculate the sampling frequency, simply obtained by reciprocal of the difference of the timeFIDArray step interval in the FFTSignalSpectraForTimeBin itself 
tmax = timeFIDArray(end);  
nfft = round(length(timeFIDArray) * (dt / tmax)); 

% Initialize the combined Fourier transform arrays
time_binned_FFT_arrays = cell(numMeasurementsOfSample, 1);
for i = 1:numMeasurementsOfSample
	%for each FID file, perform a fourier transform and store in an array "time_binned_FFT_arrays". 
	%in each cell lies the FFT for timeFIDArray bin 1, 2, 3... and so on depending on the length of numTimeBinIntervals
    time_binned_FFT_arrays{i} = zeros(numTimeBinIntervals, nfft);
end

freq = linspace(-fs/2, fs/2, nfft); % Frequency axis
freq = freq * (3.57*10^-3) - 8.53; %Here, we shifted and scaled the frequency axis based off of the SpinWorks profile. We wanted to match our FFT to the sampleTimeFrequencyDataArrray there. 

%initializing our arrays
avg_time_array = [];
chi_squared_array = [];


% Perform Fourier transform for each timeFIDArray interval and plot
for i = 1:numTimeBinIntervals
	%in this for loop we are taking the FFT of the FID in increments of dt, and then plotting each FFT 
    startTime = start_time + (i-1) * dt;
    endTime = startTime + dt;

    % Find the indices corresponding to the specified timeFIDArray interval.
    indices = find(timeFIDArray >= startTime & timeFIDArray <= endTime);

    % Compute the average timeFIDArray for the selected
    avg_time = mean(timeFIDArray(indices));
    
    % Add the average timeFIDArray to the array
    avg_time_array = [avg_time_array; avg_time];

    % Extract the FID FFTSignalSpectraForTimeBin within the specified interval
    fid_interval = cell(1, numMeasurementsOfSample);
    for j = 1:numMeasurementsOfSample
        fid_interval{j} = fid{j}(indices);
    end
    
    % Plot the Fourier transformed bin for the current interval
    figure('Units', 'inches', 'Position', [0, 0, figSize]);
    hold on;
    %mean_fft_fid_cell = cell(1, numMeasurementsOfSample);
    for j = 1:numMeasurementsOfSample
	%fft_current_FID is the FFT of the nth sample in the range of numMeasurementsOfSample 
	%time_binned_FFT_arrays is an array that stores the FFT FFTSignalSpectraForTimeBin of all seven samples. For each sample, there are (total_time)/dt number of cells that correspond to the decaying FFT signal over time
		fft_current_FID = fftshift(fft(fid_interval{j}, nfft));
        time_binned_FFT_arrays{j}(i, :) = abs(fft_current_FID);
       
		%now plotting the FFT spectra for each time interval, should see decaying spectra 
        plot(freq, time_binned_FFT_arrays{j}(i, :), 'o-');
    end

    xlabel('Frequency (Hz)', 'FontSize', fontSize);
    ylabel('Magnitude', 'FontSize', fontSize);
    title(sprintf('Fourier Transform for Interval %.2f-%.2f s', startTime, endTime), 'FontSize', fontSize);
    grid on;

    % Create a legend for the FIDs
    legendEntries = cell(numMeasurementsOfSample, 1);
    for j = 1:numMeasurementsOfSample
        legendEntries{j} = sprintf('FID %d', j);
    end
    legend(legendEntries, 'FontSize', fontSize);
end

% Prompt the user to select N peaks visually using ginput, multiplying by
% three because each peak as a left and righthand background region we also
% want to specify 
N = input('Enter the number of peaks to select: ')*3;

% Initialize the peakLocations array
peakLocations = zeros(N, 3);

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


% Set grid on
grid on;
numFrequencyBinIntervals = (freq(end) - freq(1)) / ft; %here we define the number of frequency intervals across the FFT  
%add peak picker before plotting the ===============================
current_interval = 0; 
current_values = 0; 

interval_values = []; 
riemannSumFreqBin = []; 
riemannSumArray = []
final = []; 
sampleTimeFrequencyDataArrray = [];

%this nested for loop block acts to identify the peak values associated
%with each sample, in each time bin, in each frequency bin 
%CHANGE ==============
%this nested for loop identifies a frequency bin for each sample in each
%time binned interval, computes the Riemann sum and correlates it to a
%frequency value (this frequency value is simply the center of the Riemann)

subplot_count = 0; 
for j = 1:numMeasurementsOfSample % for each file
    for k = 1:numTimeBinIntervals % for each timeFIDArray bin
        % figure;
        for i = freq(1):ft:freq(end) 
            current_leftpoint = i;
            current_rightpoint = i + ft;
            current_interval = [num2str(current_leftpoint) ' ', num2str(current_rightpoint)];
            indices = find(freq >= current_leftpoint & freq <= current_rightpoint);

            % now, in the selected FFT region we create an array that holds all of the signal strengths 
            values_fid = time_binned_FFT_arrays{j, 1}(k, :);
            values_in_interval = values_fid(indices);

            % now, we can calculate the number of rectangles within a frequency bin 
            numPointsInInterval = length(values_in_interval);

            totalArea = 0;
            % compute the Riemann sum for within the accepted frequency window 
            for i = 1:length(values_in_interval)
                widthOfRectangle = ft / numPointsInInterval;
                areaOfRectangle = widthOfRectangle * values_in_interval(i); 
                totalArea = totalArea + areaOfRectangle;
            end
            % we need to assign this Riemann sum with a corresponding frequency point, we'll choose the centermost value
            centerOfValues = ceil(numPointsInInterval / 2);

            riemannFrequency = freq(indices(centerOfValues)); 

            riemannSumFreqBin = [riemannSumFreqBin; {indices, totalArea}];
            
            % Movie to plot the current interval and its Riemann value
            % plot(freq(indices), values_fid(indices)); % plot the interval
            % hold on;
            % plot(riemannFrequency, totalArea, 'ro'); % plot the maximum value
            % hold off;
            % % Set labels and title for the plot
            % xlabel('Frequency');
            % ylabel('Amplitude');
            % title(['Riemann Sum vs Frequency Position (Time Bin ', num2str(k), ')']);
            % % Pause to observe each plot
            % pause(0.01);

            % totalArea, riemannFrequency and sampleTimeFrequencyDataArrray are arrays to help append our values and frequencies to other arrays
            % totalArea is an array that calculates the Riemann sum value within a
            % frequency region within a time bin. riemannSumArray pairs
            % the totalArea in an array with its time bin 

            %saveas(gcf, ['plot_file', num2str(j), '_bin', num2str(k), '.png']);
        end

        riemannSumArray = [riemannSumArray; {k, riemannSumFreqBin}]
        riemannSumFreqBin = [];
    end
    sampleTimeFrequencyDataArrray = [sampleTimeFrequencyDataArrray; {j, riemannSumArray}];
    riemannSumArray = [];
end


FFTSignalStrengthSpectraForTimeBin = cell(numTimeBinIntervals, 1);
averageSignalStrengthAcrossSamples = cell(numTimeBinIntervals, 1);

%this nested for loop block acts to look in sampleTimeFrequencyDataArrray,
%which is organized in the following way:
%numMeasurementsOfSample (1,numMeasurementsOfSample) -> timeBins (1, numTimeBinIntervals) ->
%numFrequencyBinIntervals (1, numFrequencyBinIntervals) -> signal strength
%at a specific frequency 

%below, for each time bin across all samples, we will compute the average
%signal strength. Ex) if we have 7 samples which are partitioned into 4
%time bins - the code below averages the FFT of the first time bin of
%all of the samples and so on. 

%we do this because we take seven measurements of the same sample 

for j = 1:numTimeBinIntervals
    %FFTSignalSpectraForTimeBin looks at a specific sample, a specific time bin interval and
    %collects all of the FFT signal strength values for that spectra
    iAverageSignalStrengthAcrossSamples = [];
    iFFTSignalStrengthSpectraForTimeBin = {};

    for k = 1:numMeasurementsOfSample
        %here, we are computing the average of the signal across each time
        %bin for each sample. The end result is several FFT arrays for
        %each decaying sample. This is the mean (average value).
        FFTSignalSpectraForTimeBin = sampleTimeFrequencyDataArrray{k, 2}{j, 2}(:, 2);
        iFFTSignalStrengthSpectraForTimeBin = [iFFTSignalStrengthSpectraForTimeBin; {FFTSignalSpectraForTimeBin}];
        iAverageSignalStrengthAcrossSamples = computeAverageFFTAcrossTimeBins(iFFTSignalStrengthSpectraForTimeBin);
    end

    % Assign the computed arrays to the cell arrays
    FFTSignalStrengthSpectraForTimeBin{j} = iFFTSignalStrengthSpectraForTimeBin;
    averageSignalStrengthAcrossSamples{j} = iAverageSignalStrengthAcrossSamples;

    %repeat process for each time bin interval 
end


%now we will compute the SDOM for each peak value (Standard Deviation of Mean)
computeSDOM = [];
arraySDOM = [];
sumRiemannValues = [];
r = zeros(numTimeBinIntervals, 1); % Initialize the 'r' vector to store RMS values for each numInterval

for z = 1:ceil(numFrequencyBinIntervals) %looping over the number of frequency bins in a given time bin (round up because number is not integer)  
    for i = 1:numTimeBinIntervals %looping over each time bin interval
        getMean = averageSignalStrengthAcrossSamples{i,:}(z, :);
        for k = 1:numMeasurementsOfSample %looping over run number for each sample
            addRiemannValues = FFTSignalStrengthSpectraForTimeBin{i,:}{k, 1}(z, 1);
            %here we are calculating the SDOM
            r(i) = r(i) + (cell2mat(addRiemannValues) - getMean)^2;
        end 
     end
            computeSDOM = (r/numMeasurementsOfSample).^(0.5);  % Compute the RMS values for all numTimeBinIntervals
            arraySDOM = [arraySDOM, computeSDOM]; % Add the computed RMS values to arrayRMS as a new column
            r = zeros(numTimeBinIntervals, 1); % Reset the 'r' vector for the next FFTSignalSpectraForTimeBin point
end

mean_array = [];
plot_STD_array = [];
M0_uncertainty_array = [];

% Define the frequency range
freq_start = freq(1);
freq_end = freq(end);

% Calculate the number of steps
numFrequencyBinIntervals = (freq_end - freq_start) / ft;

% Define subplot dimensions
num_rows = 2;
num_cols = 2;

% Loop over each frequency step iteratively and plot the corresponding
% function of the decaying form 
try
    % Loop over each frequency step iteratively and plot the corresponding
    % function of the decaying form 
    for i = 1:numFrequencyBinIntervals
        % Calculate the current frequency region
        current_freq_start = freq_start + (i - 1) * ft;
        current_freq_end = current_freq_start + ft;
        current_interval = [num2str(current_freq_start) ' ', num2str(current_freq_end)];

        % Create a new figure for the current frequency region if needed
        if mod(i-1, num_rows*num_cols) == 0
            figure; % Create a new figure
        end

        subplot(num_rows, num_cols, mod(i-1, num_rows*num_cols) + 1); % Create subplots
        hold on;
        


        % Plot peak values for each FID file and compute mean value for the current frequency region
        for j = 1:numTimeBinIntervals
            find_mean = averageSignalStrengthAcrossSamples{j}(i, :);
            mean_array = [mean_array; find_mean];
            find_RMS = arraySDOM(j, i);
            plot_STD_array = [plot_STD_array; find_RMS];
            plot(j, find_mean, 'ro'); % Plot the mean value
            errorbar(j, find_mean, find_RMS, 'r'); % Draw RMS error bar atop of the find_mean point 

            for k = 1:numMeasurementsOfSample
                find_V = cell2mat(FFTSignalStrengthSpectraForTimeBin{j}{k}(i, :));
                plot(j, find_V, 'bx'); % Plot the peak values for the current FID file
            end

        end
        
        % Create a guess 
        guessM0 = mean_array(1);
        guessB = mean_array(end);
        
        % Here, we are doing a linear plot to the data, which we expect to
        % be exponential. We linearize the y axis which is the mean_array
        % of values. We get some inf and NaN returns sometimes using this
        % method to guess our paramaters, so an alternative is provided
       
        %coefficients = polyfit(avg_time_array, log(mean_array), 1);

        % Extract the coefficient for the expsampleTimeFrequencyDataArrrayntial term
        %guessTAU = -1 / coefficients(1);
        guessTAU = -1/((log(mean_array(end)) - log(mean_array(1)))/(avg_time_array(end)-avg_time_array(1)));

        testarray = (1:numTimeBinIntervals)';
        initialguess = [guessM0, guessTAU, guessB];

        results = nonlinear_regression_plot(testarray, mean_array, initialguess, plot_STD_array);
       
        M0_uncertainty_array = [M0_uncertainty_array; results.uncertainty_M0_t0, results.Mt0, {current_interval}];

        mean_array = [];
        plot_STD_array = []; 

        % Add labels and title for the plot
        xlabel('timeFIDArray Bin');
        ylabel('Peak Value');
        title(['Plot for frequency region [', num2str(current_freq_start), ' Hz - ', num2str(current_freq_end), ' Hz]']);
    end
catch ME
    disp('An error occurred during the execution of the loop:');
    disp(ME.message);
end
function results = nonlinear_regression_plot(xdata, ydata, beta0, uncertainties)
    % Nonlinear regression function form: M(t) = M(0) * exp(-t/tau) + B

    % Create the anonymous function for the model
    model = @(beta, t) beta(1) * exp(-t / beta(2)) + beta(3);

    % Perform the nonlinear regression using nlinfit
    [beta_fit, residuals, J, cov_beta, mse] = nlinfit(xdata, ydata, model, beta0, 'Weights', 1 ./ uncertainties.^2);

    % Calculate chi-squared goodness of fit
    chiSq = sum(residuals.^2 ./ uncertainties.^2);
    degFreedom = length(xdata) - numel(beta_fit);
    
    % Add chi-squared and degrees of freedom to results
    results.chiSq = chiSq;
    results.degFreedom = degFreedom;

    % Calculate uncertainty in M(0) at t = 0 using error propagation
    t0 = 0; % Value of t at which we want to make the extrapolation
    dMdM0 = exp(-t0 / beta_fit(2));
    dMdtau = -(beta_fit(1) * t0 / beta_fit(2)^2) * exp(-t0 / beta_fit(2));
    dMdB = 1;

    % 3x3 matrix
    % sigmaM0       -            -
    %   -        sigma_tau
    %   -           -          sigma_B

    sigma_M0 = sqrt(cov_beta(1, 1));
    sigma_tau = sqrt(cov_beta(2, 2));
    sigma_B = sqrt(cov_beta(3, 3));

    % Calculate M(t) at t = 0 and uncertainty
    Mt0 = beta_fit(1) + beta_fit(3);
    uncertainty_M0_t0 = sqrt((dMdM0^2 * sigma_M0^2) + (dMdtau^2 * sigma_tau^2) + (dMdB^2 * sigma_B^2));

    %calculate percent error 
    percent_error = uncertainty_M0_t0/Mt0 * 100

    %calculate ChiSq GOF
    %experimental = [];
    %observed = [];
    %for i = 1:length(xdata);
    %    experimental = beta(1) * exp(-i / beta(2)) + beta(3)
        %observed = ydata(i)
    %end
            
    % Store the results in the output structure
    results.beta_fit = beta_fit;
    results.residuals = residuals;
    results.J = J;
    results.cov_beta = cov_beta;
    results.mse = mse;
    results.uncertainty_M0_t0 = uncertainty_M0_t0;
    results.Mt0 = Mt0;

    % Convert the functional form and Mt0 with uncertainty to LaTeX strings
    beta1_latex = sprintf('%.4f', beta_fit(1));
    beta2_latex = sprintf('%.4f', beta_fit(2));
    beta3_latex = sprintf('%.4f', beta_fit(3));
    Mt0_latex = sprintf('%.4f', Mt0);
    percent_error_latex = sprintf('%.4f', percent_error); 
    uncertainty_M0_t0_latex = sprintf('%.4f', uncertainty_M0_t0);

    % Plot the curve fit and add legend
    t_fit = linspace(min(xdata), max(xdata), 100); % Generate points for the fitted curve
    M_fit = model(beta_fit, t_fit); % Evaluate the fitted curve using the estimated parameters

    % Plot the fitted curve without adding it to the legend
    plot(t_fit, M_fit, 'r-', 'DisplayName', '$\beta(1)\exp(-t/\beta(2)) + \beta(3)$');

    % Plot FFTSignalSpectraForTimeBin points with error bars and add them to the legend
    errorbar(xdata, ydata, uncertainties, 'bo', 'DisplayName', 'FFTSignalSpectraForTimeBin Points with Error Bars');

    % Add grid lines for better readability
    grid on;

    % Adjust font size for axis labels and ticks
    set(gca, 'FontSize', 14);

    % Add the legend with the functional form and Mt0 with uncertainty in LaTeX form
    legend_str = {...
    '$\beta(1)\exp(-t/\beta(2)) + \beta(3)$', ...
    ['M(t=0) = $', Mt0_latex, '\pm', uncertainty_M0_t0_latex, '$'], ...
    ['\textup{ Percent Error: }', percent_error_latex], ...
    ['$\chi^{2} = $', num2str(chiSq)], ...
    ['Degrees of Freedom = ', num2str(degFreedom)]...
};

    % Plot the legend with updated strings
    legend(legend_str, 'Interpreter', 'latex');
    return
end

%this function acts to, given an array of riemann sums across multiple timebins, average them 
function output_array = computeAverageFFTAcrossTimeBins(input_array)
    % Initialize the maxSignalStrengthArray array to store the sum of elements
    accruing_array = zeros(size(input_array{1}));

    % Loop through each cell in the input array
    for i = 1:numel(input_array)
        % Get the values from the current cell
        current_cell = cell2mat(input_array{i});

        % Add values to the maxSignalStrengthArray array
        accruing_array = accruing_array + current_cell;
    end

    % Divide the sum by the number of elements in the input_array
    mean_array = accruing_array / numel(input_array);

    % Get the name of the input variable
    input_var_name = inputname(1);

    % Create the output variable name with '_mean' suffix
    output_var_name = [input_var_name, '_mean'];

    % Assign the mean_array to the output variable with the constructed name
    assignin('caller', output_var_name, mean_array);

    % Set the function output (optional, as the user can access it from the workspace)
    output_array = mean_array;
end
