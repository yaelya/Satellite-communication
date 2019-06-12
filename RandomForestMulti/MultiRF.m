% https://www.mathworks.com/help/stats/select-predictors-for-random-forests.html
tic

clc; clear all
data=xlsread('34_35_1.xlsx');
AtmosLossdB=data(1:end,3);
RainLossdB = data(1:end,4);
CloudsFogLossdB = data(1:end,5);
FreqDopplerShiftkHz = data(1:end,7);
Y = data(:,end);
CNdB = data(1:end,12);

X = table(AtmosLossdB,RainLossdB,CloudsFogLossdB,FreqDopplerShiftkHz);
str = ["AtmosLossdB","RainLossdB","CloudsFogLossdB","FreqDopplerShiftkHz"];
countLevels = categories(categorical(str));
C=categorical(str)%%%%%%%%%      ????????????
func=@(x)numel(str)%%%%%%%%%     ????????????
numLevels = varfun(func,X,'OutputFormat','uniform');
 
% data=xlsread('34_35.xlsx');
% CloudsFogLossdB = data(1:end-1,4);
% AtmosLossdB = data(1:end-1,3);
% RcvdFrequencyGHz = data(1:end-1,7);
% X = table(AtmosLossdB,CloudsFogLossdB,RcvdFrequencyGHz);

figure
bar(numLevels)
title('Number of Levels Among Predictors')
xlabel('Predictor variable')
ylabel('Number of levels')
h = gca;
h.XTickLabel = X.Properties.VariableNames(1:end-1);
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';
 
t = templateTree('NumVariablesToSample','all',...
      'PredictorSelection','interaction-curvature','Surrogate','on');
rng(1); % For reproducibility
Mdl = fitrensemble(X,Y,'Method','Bag','NumLearningCycles',200, ...
  'Learners',t);% ???????? ------> I put Y because MPG is not work
 

 
yHat = oobPredict(Mdl);
R2 = corr(Mdl.Y,yHat)^2


impOOB = oobPermutedPredictorImportance(Mdl);
figure
bar(impOOB)
title('Unbiased Predictor Importance Estimates')
xlabel('Predictor variable')
ylabel('Importance')
h = gca;
h.XTickLabel = Mdl.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';
 
 
[impGain,predAssociation] = predictorImportance(Mdl);

figure
plot(1:numel(Mdl.PredictorNames),[impOOB' impGain'])
title('Predictor Importance Estimation Comparison')
xlabel('Predictor variable')
ylabel('Importance')
h = gca;
h.XTickLabel = Mdl.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';
legend('OOB permuted','MSE improvement')
grid on

figure
imagesc(predAssociation)
title('Predictor Association Estimates')
colorbar
h = gca;
h.XTickLabel = Mdl.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';
h.YTickLabel = Mdl.PredictorNames;

predAssociation(1,2)
toc