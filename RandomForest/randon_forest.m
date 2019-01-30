tic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name - call_generic_random_forests
% Creation Date - 7th July 2015
% Author - Soumya Banerjee
% Website - https://sites.google.com/site/neelsoumya/
%
% Description - Function to load data and call generic random forests function
%
% Parameters -
%	Input
%
%	Output
%               BaggedEnsemble - ensemble of random forests
%               Plots of out of bag error
%		Example prediction
%
% Example -
%		call_generic_random_forests()
%
% Acknowledgements -
%           Dedicated to my mother Kalyani Banerjee, my father Tarakeswar Banerjee
%				, my wife Joyeeta Ghose and my friend Irene Egli.
%
% License - BSD
%
% Change History -
%                   7th July 2015 - Creation by Soumya Banerjee
%                   12th July 2017 - Modified by Soumya Banerjee to try different leaf node
%                                   and estimate feature importance
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%load fisheriris
%X = meas;
%Y = species;
%BaggedEnsemble = generic_random_forests(X,Y,60,'classification')
%predict(BaggedEnsemble,[5 3 5 1.8])

% load breast cancer data
data_matrix=xlsread('file.xlsx');
icol = size(data_matrix,2)
data_predictor = data_matrix(:,1:icol-1); % predictors matrix
label = data_matrix(:,end); % last column is 2 for benign, 4 for malignant

BaggedEnsemble = generic_random_forests(data_predictor, label, 500, 'classification')
%%predict(BaggedEnsemble, [5 3 5 1.8])

% Model says that x6 (single epithelial cell size) is most important
% predictor
toc
