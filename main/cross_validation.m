function [Y, prob] = cross_validation( classifier, X, y, nTree, nFold)
% perform cross validation using "classifier" and return the results
[labels,~,y] = unique(y(:));
c = randi(nFold,size(y));
Y = zeros(size(y));
prob = zeros(length(y),length(labels));
for i = 1:nFold
  msk1 = (c~=i);
  msk2 = (c==i);
  C    = classifier.train (X(msk1,:),y(msk1),[],nTree);
  [y1, p1]     = C.predict(X(msk2,:));
  prob(msk2,:) = p1;
  Y   (msk2)   = y1;
end
Y = labels(Y);