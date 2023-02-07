clear all; close all; clc

% -------------------------------------------------------------------------------------------------------------------
% LOAD DATASET

X1 = csvread('data_all_norm_corrected.txt');
X1 = [X1(1:28, :); X1(38+1:end-1, :)]; % 28 ALC and NOR such that 58 samples for 4fold cross validation

order = 3;
remove_col1 = 1;
remove_col2 = 3;

X21 = csvread(strcat('arx_coeff/alc_coeff_', num2str(order), '.csv')); % This only has the samples for 29ALC and 29NOR 
X22 = csvread(strcat('arx_coeff/nor_coeff_', num2str(order), '.csv')); % NOT 38 ALC and 29 NOR
X2 = [X21; X22];
X2(29, :) = []; X2(end, :) = [];
%%%%%%%%%%%%X2 = [X2(1:28, :); X2(38+1:end-1, :)]; % 28 ALC and NOR such that 58 samples for 4fold cross validation
X2(:, remove_col1) = []; % no variation in these coeffs, leads to div by 0 err during normalization
X2(:, remove_col2 - 1) = []; % the minus 1 is added since after removing a col in the prev line, dims change
X2 = normalize_no_save(X2);

X = [X1 X2];


m1 = 28; m2 = 28; % Total numner of ALC and NOR samples as per updated X
y0 = zeros(m1, 1);
y1 = ones(m2, 1);
y = [y0; y1];

% -------------------------------------------------------------------------------------------------------------------
% K-FOLD
 k = 8;
 ind = crossvalind('Kfold',m1+m2,k); % Labelling should not be the same for all 100 iterations, else nothing changes in each of the 100 loops, except for minor errors due to SMO algorithm

cs = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigmas = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
count = zeros(length(cs), length(sigmas));
for loop = 1:200
    disp('#########################');
    overall_err = zeros(length(cs), length(sigmas));
    k = 8;
    ind = crossvalind('Kfold',m1+m2,k); % Labelling must change for every full/overall iteration 

    for c_ind = 1:length(cs)
        disp('**********************');
        C = cs(c_ind)
        for sig_ind = 1:length(sigmas)
            disp('~~~~~~~~~~~~~~~~~~~~~~~');
            loop
            C
            sigma = sigmas(sig_ind)
            
            % k = 8;
            % ind = crossvalind('Kfold',m1+m2,k); % Labelling must stay the same while testing against all C,sigma pairs
            errors = zeros(1, k);
            for i = 1:k
                X_x = X(ind~=i, :);
                Y_y = y(ind~=i);
                Xval = X(ind==i, :);
                yval = y(ind==i);
                
                % [C, sigma] = pick_c_sigma( X_x, Y_y, Xval, yval);%pick_c_sigma_builtIn( X_x, Y_y, Xval, yval);
                % Ck(i) = C; sigmak(i) = sigma;
                model= svmTrain(X_x, Y_y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
                % save the model here, rather than in pick_c_sigma
                predictions = svmPredict(model, Xval);
                errors(i) =  mean(double(predictions ~= yval));
            end
            overall_err(c_ind, sig_ind) = mean(errors);
        
        end
    end

    %csvwrite('svm_err_outer_c_sigma/err_c_sigma_vary_5.csv', overall_err);
    %[[0 sigmas'];[cs overall_err]]
    
    %[colwise_mins, row_inds] = min(overall_err, [], 1);
    [rowwise_mins, col_inds] = min(overall_err, [], 2); 
    
    for l = 1:length(col_inds)
        %r = row_inds(l); % don't take all combinations of the row and col (thus not two loops), but take as pairs of elements
        c = col_inds(l);
        count(l, c) = count(l,c) + 1;        
    end
    % In case I kill the process, of Octave crashes, I have the saved version. I can just rerun remaining number of loops
    % and add the count matrix created in each instance
%    csvwrite('svm_err_outer_c_sigma/count_arx_order3.csv', count);
%    csvwrite('svm_err_outer_c_sigma/overall_err_arx_order3.csv', overall_err);
%    csvwrite('svm_err_outer_c_sigma/loop_arx_number_order3.csv', loop);
    
end
