function AdaBoost_demo_plot(classifier, X, y)
nClass = length(unique(y));
%% Prepare plot
range = [-10, 10];
k = 200;
xVec = linspace(range(1), range(2), k);
yVec = linspace(range(1), range(2), k);
[Xg, Yg] = meshgrid(xVec, yVec);
Xp = [Xg(:), Yg(:)];
%% train and test classifier
[Y, P] = classifier.predict(Xp);
Y = reshape(Y, k, k);
P = reshape(P(:,end), k, k);
%% Plot
figure( 'Position', [245 659 1175 439])
clf
subplot(1,2,1)
colormap(jet)
imagesc(xVec, yVec, Y);
hold on
set (gca, 'YDir', 'normal');
ttl = sprintf('Test with %i classes using %s', nClass, ...
  class(classifier.weak_learner));
title({ttl, 'Decision boundaries'},'Interpreter','none')
if nClass>2
  plot(X(y==1,1), X(y==1,2), 'g.');
  plot(X(y==2,1), X(y==2,2), 'm.');
  plot(X(y==3,1), X(y==3,2), 'y.');
else
  plot(X(y==1,1), X(y==1,2), 'g.');
  plot(X(y==2,1), X(y==2,2), 'y.');
end
subplot(1,2,2)
colormap(jet)
imagesc(xVec, yVec, P);
colorbar
hold on
set (gca, 'YDir', 'normal');
%contour(xVec, yVec, P, [1 1]/2, 'LineWidth', 3, 'LineColor', [0 0 0]);
title({ttl, sprintf('Probability of class #%i', nClass)},'Interpreter','none')
if nClass>2
  plot(X(y==1,1), X(y==1,2), 'g.');
  plot(X(y==2,1), X(y==2,2), 'm.');
  plot(X(y==3,1), X(y==3,2), 'y.');
else
  plot(X(y==1,1), X(y==1,2), 'g.');
  plot(X(y==2,1), X(y==2,2), 'y.');
end
