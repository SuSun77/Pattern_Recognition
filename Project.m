%4.2
%load data
Xtrain = readmatrix('X_train.txt');
ytrain = readmatrix('y_train.txt');
Xtest = readmatrix('X_test.txt');
ytest = readmatrix('y_test.txt');
%conduct PCA
Sigma = (Xtrain' * Xtrain) / size(Xtrain,1);
[U, S, V] = svd(Sigma);
diagVec = diag(S);
for k = 1:size(Xtrain, 2)
    if sum(diagVec(1:k))/sum(diagVec) >= .99
        Ureduce = U(:, 1:k);
        Z = Xtrain * Ureduce;
        break
    end
end
%conduct 10-times-10-folds cross validation
numFeature = size(Z, 2);
res = eye(10);
dataset = [Z, ytrain];
numSample = round(length(Z) / 10);
   
for time = 1:10
    data = dataset(randperm(length(Z)),:);
    for fold = 1:10
        if fold > 9
            VC = (((fold - 1) * numSample + 1):length(Z));
        else
            VC = (((fold - 1) * numSample + 1):fold * numSample);
        end
        Vdata = data(VC, :);
        Tdata = data;
        Tdata(VC,:) = [];
        
        Xtrain1 = Tdata(:, 1:numFeature);
        ytrain1 = Tdata(:, numFeature+1);
        Xtest1 =  Vdata(:, 1:numFeature);
        ytest1 =  Vdata(:, numFeature+1);
        
        [accuracy01,label] = KNNSVM(Xtrain1, ytrain1, Xtest1, ytest1, 0.1, 5);
        res01(fold, time) = accuracy01;
        [accuracy1,label] = KNNSVM(Xtrain1, ytrain1, Xtest1, ytest1, 1, 5);
        res1(fold, time) = accuracy1;
        [accuracy10,label] = KNNSVM(Xtrain1, ytrain1, Xtest1, ytest1, 10, 5);
        res10(fold, time) = accuracy10;
        [accuracy100,label] = KNNSVM(Xtrain1, ytrain1, Xtest1, ytest1, 100, 5);
        res100(fold, time) = accuracy100;
         
        [accuracy4,label] = KNNSVM(Xtrain1, ytrain1, Xtest1, ytest1, 100, 4);
        res4(fold, time) = accuracy4;
        [accuracy5,label] = KNNSVM(Xtrain1, ytrain1, Xtest1, ytest1, 100, 5);
        res5(fold, time) = accuracy5;
        [accuracy6,label] = KNNSVM(Xtrain1, ytrain1, Xtest1, ytest1, 100, 6);
        res6(fold, time) = accuracy6;
        [accuracy8,label] = KNNSVM(Xtrain1, ytrain1, Xtest1, ytest1, 100, 8);
        res8(fold, time) = accuracy8;
    end
end

%4.3
%calculate accuracy with different parameter to choose the best set
acc01 = mean(mean(res1))
acc1 = mean(mean(res1))
acc10 = mean(mean(res10))
acc100 = mean(mean(res100))
         
acc4 = mean(mean(res4))
acc5 = mean(mean(res5))
acc6 = mean(mean(res6))
acc8 = mean(mean(res8))

%plot linechart to better campare
figure(1)
x = [1:1:4];
acc = [acc01, acc1, acc10, acc100];
plot(x, acc,'-or');
set(gca,'xtick',[1 2 3 4],'xticklabel',[0.1, 1, 10, 100])
xlabel('The value of C')
ylabel('Accuracy')
title('Parameter C')
         
figure(2)
x = [1:1:4];
acc = [acc4, acc5, acc6, acc8];
plot(x, acc,'-or');
set(gca,'xtick',[1 2 3 4],'xticklabel',[4, 5, 6, 8])
xlabel('The value of k')
ylabel('Accuracy')
title('Parameter k')

%4.4
[accuracy,label] = KNNSVM(Xtrain, ytrain, Xtest, ytest, 100, 5);
figure(3)
C = confusionmat(label,ytest);
confusionchart(C,'RowSummary','row-normalized', 'XLabel','Predicted', 'YLabel','True');

function [ksAccuracy,label] = KNNSVM(Xtrain, ytrain, Xtest, ytest, C, k)
    %independent knn classifier
    kmodel = fitcknn(Xtrain, ytrain,'NumNeighbors', k,'Standardize',1);
    [label, b, c] = predict(kmodel, Xtest);
         
    %independent svm classifier
    S = sprintf('-t %f, -c %f -q', 2, C);
    smodel = svmtrain([], ytrain, Xtrain, S);
    [svmLabel, svmAccuracy, dValue] = svmpredict(ytest, Xtest, smodel, '-q');
         
    %knnsvm classifier
    [a, b] = find(c == 0);
    multiClass = setxor(a, (1: length(Xtest)));
    label(multiClass) = svmLabel(multiClass);
         
    %calculate and output the accuracy of the three methods erspectively
    knnAccuracy = sum(label == ytest) / length(ytest);
    fprintf('knn accuracy is %f\n', knnAccuracy);
    fprintf('svm accuracy is %f\n', svmAccuracy(1)/100);
    ksAccuracy = sum(label == ytest)/length(ytest);
    fprintf('knn&svm accuracy is %f\n\n', ksAccuracy);
end
