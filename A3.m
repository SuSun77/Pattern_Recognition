%1.1
%load dataset1
hw3dataset1 = readmatrix('hw3_dataset1.csv');
x1 = hw3dataset1(:,1:2);
y1 = hw3dataset1(:,3);

figure(1)
%plot to visualize the dataset
plot(x1(find(y1 == 1), 1), x1(find(y1 == 1), 2), 'ro','MarkerFaceColor', 'y')
hold on;
plot(x1(find(y1 == 0), 1), x1(find(y1 == 0), 2), 'rs', 'MarkerFaceColor', [1 .6 .6])
hold off;
legend('labelone', 'labelzero', 'Location', 'northwest')
xlabel('The first column')
ylabel('The second column')
title('hw3dataset1')

%load dataset2
hw3dataset2 = readmatrix('hw3_dataset2.csv');
x2 = hw3dataset2(:,1:2);
y2 = hw3dataset2(:,3);

figure(2)
%plot to visualize the dataset
plot(x2(find(y2 == 1), 1), x2(find(y2 == 1), 2), 'ro','MarkerFaceColor', 'y')
hold on;
plot(x2(find(y2 == 0), 1), x2(find(y2 == 0), 2), 'rs', 'MarkerFaceColor', [1 .6 .6])
hold off;
legend('labelone', 'labelzero', 'Location', 'northwest')
xlabel('The first columnn')
ylabel('The second columnn')
title('hw3dataset2')

%1.2
%Train linear SVM classifier using toolbox LibSVM-weight-3.24
SVM11 = svmtrain([], y1, x1, '-t 0, -c 0.001');
SVM12 = svmtrain([], y1, x1, '-t 0, -c 0.01');
SVM13 = svmtrain([], y1, x1, '-t 0, -c 0.1');
SVM14 = svmtrain([], y1, x1, '-t 0, -c 1');

figure(3)
%plot to show different decision boundaries
subplot(2,2,1)
plotDecisionBoundary(x1, y1, SVM11, 1, '-b', true)
legend('labelone', 'labelzero', 'Margin', 'Margin', 'Decision boundary', 'Location', 'northwest')
xlabel('The first column')
ylabel('The second column')
title('hw3dataset1 with C = 0.001')

subplot(2,2,2)
plotDecisionBoundary(x1, y1, SVM12, 1, '-b', true)
xlabel('The first column')
ylabel('The second column')
title('hw3dataset1 with C = 0.01')

subplot(2,2,3)
plotDecisionBoundary(x1, y1, SVM13, 1, '-b', true)
xlabel('The first column')
ylabel('The second column')
title('hw3dataset1 with C = 0.1')

subplot(2,2,4)
plotDecisionBoundary(x1, y1, SVM14, 1, '-b', true)
xlabel('The first column')
ylabel('The second column')
title('hw3dataset1 with C = 1')

%Train linear SVM classifier using toolbox LibSVM-weight-3.24
SVM21 = svmtrain([], y2, x2, '-t 0, -c 0.001');
SVM22 = svmtrain([], y2, x2, '-t 0, -c 0.01');
SVM23 = svmtrain([], y2, x2, '-t 0, -c 0.1');
SVM24 = svmtrain([], y2, x2, '-t 0, -c 1');

%plot to show different decision boundaries
figure(4)
subplot(2,2,1)
plotDecisionBoundary(x2, y2, SVM21, 1, '-b', 1)
legend('labelone', 'labelzero', 'Margin', 'Margin', 'Decision boundary', 'Location', 'northwest')
xlabel('The first column')
ylabel('The second column')
title('hw3dataset2 with C = 0.001')

subplot(2,2,2)
plotDecisionBoundary(x2, y2, SVM22, 1, '-b', 1)
xlabel('The first column')
ylabel('The second column')
title('hw3dataset2 with C = 0.01')

subplot(2,2,3)
plotDecisionBoundary(x2, y2, SVM23, 1, '-b', 1)
xlabel('The first column')
ylabel('The second column')
title('hw3dataset2 with C = 0.1')

subplot(2,2,4)
plotDecisionBoundary(x2, y2, SVM24, 1, '-b', 1)
xlabel('The first column')
ylabel('The second column')
title('hw3dataset2 with C = 1')

%Conduct standardizition
x1 = (x1 - mean(x1)) ./ std(x1);
SVM11 = svmtrain([], y1, x1, '-t 0, -c 0.001');
SVM12 = svmtrain([], y1, x1, '-t 0, -c 0.01');
SVM13 = svmtrain([], y1, x1, '-t 0, -c 0.1');
SVM14 = svmtrain([], y1, x1, '-t 0, -c 1');

figure(5)
subplot(2,2,1)
plotDecisionBoundary(x1, y1, SVM11, 1, '-b', 0)
legend('labelone', 'labelzero', 'Decision boundary', 'Location', 'northwest')
xlabel('The first column')
ylabel('The second column')
title('hw3dataset1 with C = 0.001')

subplot(2,2,2)
plotDecisionBoundary(x1, y1, SVM12, 1, '-b', 0)
xlabel('The first column')
ylabel('The second column')
title('hw3dataset1 with C = 0.01')

subplot(2,2,3)
plotDecisionBoundary(x1, y1, SVM13, 1, '-b', 0)
xlabel('The first column')
ylabel('The second column')
title('hw3dataset1 with C = 0.1')

subplot(2,2,4)
plotDecisionBoundary(x1, y1, SVM14, 1, '-b', 0)
xlabel('The first column')
ylabel('The second column')
title('hw3dataset1 with C = 1')

%Conduct standardizition
x2 = (x2 - mean(x2)) ./ std(x2);
SVM21 = svmtrain([], y2, x2, '-t 0, -c 0.001');
SVM22 = svmtrain([], y2, x2, '-t 0, -c 0.01');
SVM23 = svmtrain([], y2, x2, '-t 0, -c 0.1');
SVM24 = svmtrain([], y2, x2, '-t 0, -c 1');

figure(6)
subplot(2,2,1)
plotDecisionBoundary(x2, y2, SVM21, 1, '-b', 0)
legend('labelone', 'labelzero', 'Decision boundary', 'Location', 'northwest')
xlabel('The first column')
ylabel('The second column')
title('hw3dataset2 with C = 0.001')

subplot(2,2,2)
plotDecisionBoundary(x2, y2, SVM22, 1, '-b', 0)
xlabel('The first column')
ylabel('The second column')
title('hw3dataset2 with C = 0.01')

subplot(2,2,3)
plotDecisionBoundary(x2, y2, SVM23, 1, '-b', 0)
xlabel('The first column')
ylabel('The second column')
title('hw3dataset2 with C = 0.1')

subplot(2,2,4)
plotDecisionBoundary(x2, y2, SVM24, 1, '-b', 0)
xlabel('The first column')
ylabel('The second column')
title('hw3dataset2 with C = 1')

%1.3
%decision boundary for dataset1 after conduct standardizition
figure(7)
plotDecisionBoundary(x1, y1, SVM11, 1, '-g', 0)
hold on
plotDecisionBoundary(x1, y1, SVM12, 0, '-b', 0)
hold on
plotDecisionBoundary(x1, y1, SVM13, 0, '-k', 0)
hold on
plotDecisionBoundary(x1, y1, SVM14, 0, '-m', 0)
hold off
legend('labelone', 'labelzero', 'C = 0.001','C = 0.01', 'C = 0.1', 'C = 1')
xlabel('The first column')
ylabel('The second column')
title('hw3dataset1 with standardization')

%decision boundary for dataset2 after conduct standardizition
figure(8)
plotDecisionBoundary(x2, y2, SVM21, 1, '-g', 0)
hold on
plotDecisionBoundary(x2, y2, SVM22, 0, '-b', 0)
hold on
plotDecisionBoundary(x2, y2, SVM23, 0, '-k', 0)
hold on
plotDecisionBoundary(x2, y2, SVM24, 0, '-m', 0)
hold off
legend('labelone', 'labelzero', 'C = 0.001','C = 0.01', 'C = 0.1', 'C = 1')
xlabel('The first column')
ylabel('The second column')
title('hw3dataset2 with standardization')

%Plot decision boundary with origin dataset
clear
hw3dataset1 = readmatrix('hw3_dataset1.csv');
x1 = hw3dataset1(:,1:2);
y1 = hw3dataset1(:,3);
SVM11 = svmtrain([], y1, x1, '-t 0, -c 0.001');
SVM12 = svmtrain([], y1, x1, '-t 0, -c 0.01');
SVM13 = svmtrain([], y1, x1, '-t 0, -c 0.1');
SVM14 = svmtrain([], y1, x1, '-t 0, -c 1');
figure(9)
plotDecisionBoundary(x1, y1, SVM11, 1, '-g', 0)
hold on
plotDecisionBoundary(x1, y1, SVM12, 0, '-b', 0)
hold on
plotDecisionBoundary(x1, y1, SVM13, 0, '-k', 0)
hold on
plotDecisionBoundary(x1, y1, SVM14, 0, '-m', 0)
hold off
legend('labelone', 'labelzero', 'C = 0.001','C = 0.01', 'C = 0.1', 'C = 1')
xlabel('The first column')
ylabel('The second column')
title('hw3dataset1 without standardization')

hw3dataset2 = readmatrix('hw3_dataset2.csv');
x2 = hw3dataset2(:,1:2);
y2 = hw3dataset2(:,3);
SVM21 = svmtrain([], y2, x2, '-t 0, -c 0.001');
SVM22 = svmtrain([], y2, x2, '-t 0, -c 0.01');
SVM23 = svmtrain([], y2, x2, '-t 0, -c 0.1');
SVM24 = svmtrain([], y2, x2, '-t 0, -c 1');
figure(10)
plotDecisionBoundary(x2, y2, SVM21, 1, '-g', 0)
hold on
plotDecisionBoundary(x2, y2, SVM22, 0, '-b', 0)
hold on
plotDecisionBoundary(x2, y2, SVM23, 0, '-k', 0)
hold on
plotDecisionBoundary(x2, y2, SVM24, 0, '-m', 0)
hold off
legend('labelone', 'labelzero', 'C = 0.001','C = 0.01', 'C = 0.1', 'C = 1')
xlabel('The first column')
ylabel('The second column')
title('hw3dataset2 without standardization')

%1.4
%Use svmpredict function to calculate accuracy with different values of C
svmpredict(y1, x1, SVM11);
svmpredict(y1, x1, SVM12);
svmpredict(y1, x1, SVM13);
svmpredict(y1, x1, SVM14);

svmpredict(y2, x2, SVM21);
svmpredict(y2, x2, SVM22);
svmpredict(y2, x2, SVM23);
svmpredict(y2, x2, SVM24);

%% 
%2.1
%load data
classA = readmatrix('classA.csv');
classB = readmatrix('classB.csv');

%combine the two classes
x = [classA; classB];
numx = length(x);
yt = (1 : numx);
y = yt';
numA = length(classA);
y(1 : numA) = 1;
y(numA + 1 : numx) = 0;

%plot to visualize
figure(1)
plot(x(find(y == 1), 1), x(find(y == 1), 2), 'ro','MarkerFaceColor', 'y')
hold on;
plot(x(find(y == 0), 1), x(find(y == 0), 2), 'rs', 'MarkerFaceColor', [1 .6 .6])
hold off;
legend('classA', 'classB', 'Location', 'northwest')
xlabel('The first columnn')
ylabel('The second columnn')
title('classA&classB')
 
%2.2
%conduct standardizition
x = (x - mean(x)) ./ std(x);
%Train linear SVM classifier using toolbox LibSVM-weight-3.24
SVM1 = svmtrain([], y, x, '-t 0, -c 0.1');
SVM2 = svmtrain([], y, x, '-t 0, -c 1');
SVM3 = svmtrain([], y, x, '-t 0, -c 10');
SVM4 = svmtrain([], y, x, '-t 0, -c 100');

%report the accuracy based on 10-times-10-fold cross validation
res1 = CV(x, y, 0.1);
acc = mean(res1);
accuracy1 = mean(acc);

res2 = CV(x, y, 1);
acc = mean(res2);
accuracy2 = mean(acc);

res3 = CV(x, y, 10);
acc = mean(res3);
accuracy3 = mean(acc);

res4 = CV(x, y, 100);
acc = mean(res4);
accuracy4 = mean(acc);
fprintf('The accuracy based on 10-times-10-fold cross validation with C = {0.1, 1, 10, 100} is \n%f, %f, %f, %f respectively', accuracy1, accuracy2, accuracy3, accuracy4);

%plot to show the decision boundary with the best model-highest accuracy
figure(2)
%subplot(2,2,1)
%plotDecisionBoundary(x, y, SVM1, 1, '-b', 1)
%legend('ClassA', 'ClassB', 'Decision boundary', 'Location', 'northwest')
%xlabel('The first column')
%ylabel('The second column')
%title('C = 0.1')

%subplot(2,2,2)
%plotDecisionBoundary(x, y, SVM2, 1, '-b', 1)
%legend('ClassA', 'ClassB', 'Decision boundary', 'Location', 'northwest')
%xlabel('The first column')
%ylabel('The second column')
%title('C = 1')

%subplot(2,2,3)
plotDecisionBoundary(x, y, SVM3, 1, '-b', 1)
legend('ClassA', 'ClassB', 'Margin', 'Margin', 'Decision boundary', 'Location', 'northwest')
xlabel('The first column')
ylabel('The second column')
title('C = 10')

%subplot(2,2,4)
%plotDecisionBoundary(x, y, SVM4, 1, '-b', 1)
%legend('ClassA', 'ClassB', 'Margin', 'Margin', 'Decision boundary', 'Location', 'northwest')
%xlabel('The first column')
%ylabel('The second column')
%title('C = 100')

%2.3
%see function function [accuracy,  adaRes] = adaboost(trainX, trainY, testX, testY, C)

%2.4
classA = readmatrix('classA.csv');
classB = readmatrix('classB.csv');

x = [classA; classB];
numx = length(x);
yt = (1 : numx);
y = yt';
numA = length(classA);
y(1 : numA) = 1;
y(numA + 1 : numx) = 0;

%report the mean and variance
% C = 0.1
fprintf('Calculating the mean and variance of accuracy with C = 0.1');
mean11 = mean(adaboostCV(x, y, 0.1));
meanAdaboostCV1 = mean(mean11)
var11 = var(adaboostCV(x, y, 0.1));
varAdaboostCV1 = mean(var11)

% C = 1
fprintf('Calculating the mean and variance of accuracy with C = 1');
mean12 = mean(adaboostCV(x, y, 1));
meanAdaboostCV2 = mean(mean12)
var12 = var(adaboostCV(x, y, 1));
varAdaboostCV2 = mean(var12)

% C = 10
fprintf('Calculating the mean and variance of accuracy with C = 10');
mean13 = mean(adaboostCV(x, y, 10));
meanAdaboostCV3 = mean(mean13)
var13 = var(adaboostCV(x, y, 10));
varAdaboostCV3 = mean(var13)

% C = 100
fprintf('Calculating the mean and variance of accuracy with C = 100');
mean14 = mean(adaboostCV(x, y, 100));
meanAdaboostCV4 = mean(mean14)
var14 = var(adaboostCV(x, y, 100));
varAdaboostCV4 = mean(var14)

%% 
%2.5
clear
classA = readmatrix('classA.csv');
classB = readmatrix('classB.csv');

x = [classA; classB];
numx = length(x);
yt = (1 : numx);
y = yt';
numA = length(classA);
y(1 : numA) = 1;
y(numA + 1 : numx) = 0;

%plot adaboost.m1 decision boundary
x1plot = [min(x(:,1)): (max(x(:,1))-min(x(:,1)))./99: max(x(:,1))];
x1plot = x1plot';
x2plot = [min(x(:,2)): (max(x(:,2))-min(x(:,2)))./99: max(x(:,2))];
x2plot = x2plot';
[X1, X2] = meshgrid(x1plot, x2plot);
%create empty matrix to store data
res = zeros(size(X1));
yvals = zeros(size(X1));

for i = 1:size(X1, 2)
   %[accuracy, adaRes1] = adaboost(x, y, [X1(:, i), X2(:, i)], yvals(:, i), 0.1);
   %res1(:, i) = adaRes1;

   %[accuracy, adaRes2] = adaboost(x, y, [X1(:, i), X2(:, i)], yvals(:, i), 1);
   %res2(:, i) = adaRes2;

   %[accuracy, adaRes3] = adaboost(x, y, [X1(:, i), X2(:, i)], yvals(:, i), 10);
   %res3(:, i) = adaRes3;
   
   [accuracy, adaRes4] = adaboost(x, y, [X1(:, i), X2(:, i)], yvals(:, i), 100);
   res4(:, i) = adaRes4;
end

figure(1)
%use contour to plot decision boundary
plot(x(find(y == 1), 1), x(find(y == 1), 2), 'ro','MarkerFaceColor', 'y')
hold on;
plot(x(find(y == 0), 1), x(find(y == 0), 2), 'rs', 'MarkerFaceColor', [1 .6 .6])
hold on;
xlabel('The first column')
ylabel('The second column')
title('C = 100')
contour(X1, X2, res4, 7, 'k');

%% 
%reference: Stanford-AndrewWu-MachineLearning-Exercise6
function plotDecisionBoundary(x, y, model, ifScatter, color, ifmargin)
    %plot data points
    if ifScatter
        labelone = find(y == 1);
        labelzero = find(y == 0);
        plot(x(labelone, 1), x(labelone, 2), 'ro','MarkerFaceColor', 'y')
        hold on;
        plot(x(labelzero, 1), x(labelzero, 2), 'rs', 'MarkerFaceColor', [1 .6 .6])
        hold on;
    end
    %define coefficient and constant
    if (model.Label(1) == -1)
        w = -model.SVs' * model.sv_coef;
        b = model.rho;
    else
        w = model.SVs' * model.sv_coef;
        b = -model.rho;
    end
    
    xplot = [min(x(:,1)): (max(x(:,1))-min(x(:,1)))./99: max(x(:,1))];
    coe1 = -w(1) / w(2);
    coe2 = -b / w(2);
    yplot = coe1 * xplot + coe2;
    points = find(yplot > min(x(:,2)) & yplot < max(x(:,2)));
    %handle 2.2 when C is 100
    if coe1 > 20
        xplot = xplot(points);
        yplot = yplot(points);
    end
    %plot margin
    if ifmargin
        margin = 1 / sqrt(sum(model.sv_coef.^2));
        upper = yplot + sqrt(1 + coe1.^2) * margin;
        lower = yplot - sqrt(1 + coe1.^2) * margin;
        plot(xplot, upper, '-.m')
        hold on
        plot(xplot, lower, '-.m')
        hold on
    end
    plot(xplot, yplot, color)
    hold off
end

function [accuracy,  adaRes] = adaboost(trainX, trainY, testX, testY, C)
    %initialize distribution
    m = length(trainY);
    D(1 : m) = 1 / m;
    D = D';
    % initialize a vector to store beta value
    Beta = (1:50);
    prediction = zeros(length(testY), 50);
    weakLearner = svmtrain(D, trainY, trainX, sprintf('-t 0, -c %f -q', C));
    
    %conduct adaboost
    T = 1;
    while T <= 50
        %choose 100 samples to train classifier
        randata = [trainX, trainY];
        sequence = randperm(length(trainX));
        sample = randata(sequence(1:100),:);
        %train a classifier
        model = svmtrain(D(sequence(1:100)), sample(:, 3), sample(:,1:2), sprintf('-t 0, -c %f -q', C));
        %get training error
        correct = find(svmpredict(trainY, trainX, model,'-q') == trainY);
        e = sum(D(svmpredict(trainY, trainX, model,'-q') ~= trainY));
        %discard if error is higher than 50%
        if e > 0.5
            continue
        end
                
        weakLearner(T) = model;
        %set beta
        Beta(T) = e / (1 - e);
        %update distribution using normalization constant
        D(correct) = D(correct) * Beta(T);
        Z = sum(D);
        D = D / Z;
        %predict results
        prediction(:,T) = svmpredict(testY, testX, weakLearner(T), '-q');
        T = T + 1;
    end
    
    rangeT = 1:length(testY);
    adaRes = rangeT;
    adaRes = adaRes';
    for index = rangeT
        classA = prediction(index,:) * log(1./Beta)';
        %get final hypothesis 
        argMax = sum(log(1./Beta))/2;
        if classA > argMax
            adaRes(index) = 1;
        else
            adaRes(index) = 0;
        end
    end
    
    lenTy = length(testY);
    accuracy = sum(testY == adaRes) / lenTy;
    
end
         
function res = adaboostCV(x, y, C)
     res = eye(10);
     dataset = [x, y];
     numSample = round(length(x) / 10);
     for time = 1:10
         data = dataset(randperm(length(x)),:);
         for fold = [1:10]         
             if fold > 10  
                 VC = (((fold - 1) * numSample + 1):length(x));
             else
                 VC = (((fold - 1) * numSample + 1):fold * numSample);
             end
             Vdata = data(VC, :);
             Tdata = data;
             Tdata(VC,:) = [];

             trainX = Tdata(:,1:2);
             trainY = Tdata(:,3);
             testX = Vdata(:,1:2);
             testY = Vdata(:,3);
             accuracy = adaboost(trainX, trainY, testX, testY, C);            
             res(fold, time) = accuracy;
         end
     end
end

function res = CV(x, y, C)
    res = eye(10);
    dataset = [x, y];
    numSample = round(length(x) / 10);
    for time = 1:10
        data = dataset(randperm(length(x)),:);
        for fold = 1:10
            %except last fold
            if fold > 9
                VC = (((fold - 1) * numSample + 1):length(x));
            else
                VC = (((fold - 1) * numSample + 1):fold * numSample);
            end

            Vdata = data(VC, :);
            Tdata = data;
            Tdata(VC,:) = [];
            X = Tdata(:, 3);
            Y = Tdata(:, 1:2);
            S = sprintf('-t 0, -c %f -q', C);
            model = svmtrain([], X, Y, S);
            [prediction, accuracy, dValue] = svmpredict(Vdata(:, 3), Vdata(:, 1:2), model);
            res(fold, time) = accuracy(1);
        end
    end
end
