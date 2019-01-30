function BaggedEnsemble = generic_random_forests(X,Y,iNumBags,str_method)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name - generic_random_forests
% Creation Date - 6th July 2015
% Author - Soumya Banerjee
% Website - https://sites.google.com/site/neelsoumya/
%
% Description - Function to use random forests
%
% Parameters - 
%	Input	
%		X - matrix
%		Y - matrix of response
%		iNumBags - number of bags to use for boostrapping
%		str_method - 'classification' or 'regression'
%
%	Output
%               BaggedEnsemble - ensemble of random forests
%               Plots of out of bag error
%
% Example -
%
%	 load fisheriris
% 	 X = meas;
%	 Y = species;
%	 BaggedEnsemble = generic_random_forests(X,Y,60,'classification')
%	 predict(BaggedEnsemble,[5 3 5 1.8])
%
%
% Acknowledgements -
%           Dedicated to my mother Kalyani Banerjee, my father Tarakeswar Banerjee
%				and my wife Joyeeta Ghose.
%
% License - BSD
%
% Change History - 
%                   7th July 2015 - Creation by Soumya Banerjee
%                   12th July 2017 - Modified by Soumya Banerjee to try different leaf node
%                                   and estimate feature importance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% from https://uk.mathworks.com/help/stats/ensemble-methods.html#bsx62vu
% find optimal leaf size
leaf = [5 10 20 50 100];
col = 'rbcmy';
figure
for i=1:length(leaf)
    b = TreeBagger(50,X,Y,'Method',str_method,'OOBPred','On',...
            'MinLeafSize',leaf(i));
    plot(oobError(b),col(i))
    hold on
end
xlabel 'Number of Grown Trees'
ylabel 'Mean Squared Error'
legend({'5' '10' '20' '50' '100'},'Location','NorthEast')
hold off
min_leaf_size = 5 % TODO: to be computed automatically from above, now hard-coded
BaggedEnsemble = TreeBagger(iNumBags,X,Y,'OOBPred','On','Method',str_method)
% plot out of bag prediction error
oobErrorBaggedEnsemble = oobError(BaggedEnsemble);
figID = figure;
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';
print(figID, '-dpdf', sprintf('randomforest_errorplot_%s.pdf', date));
oobPredict(BaggedEnsemble)
% view trees
view(BaggedEnsemble.Trees{1}) % text description
view(BaggedEnsemble.Trees{1},'mode','graph') % graphic description
% estimate feature importance
b = TreeBagger(iNumBags,X,Y,'Method',str_method,'OOBVarImp','On',...
    'MinLeafSize',min_leaf_size);
figure
plot(oobError(b))
xlabel 'Number of Grown Trees'
ylabel 'Out-of-Bag Mean Squared Error'
figure
bar(b.OOBPermutedVarDeltaError)
xlabel 'Feature Number'
ylabel 'Out-of-Bag Feature Importance'
idxvar = find(b.OOBPermutedVarDeltaError>0.7)