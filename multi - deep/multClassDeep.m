data = xlsread('file1.xlsx');
Y = data(:,end);
X = data(:,1:end-1);

meas = X;
species = Y;

t = templateSVM('Standardize',true,'SaveSupportVectors',true);
predictorNames = {'petalLength','petalWidth'};
responseName = 'irisSpecies';
classNames = {'setosa','versicolor','virginica'}; % Specify class order
Mdl = fitcecoc(X,Y,'Learners',t,'ResponseName',responseName,...
     'PredictorNames',predictorNames,'ClassNames',classNames)
