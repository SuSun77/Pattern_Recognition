%1a
%refer:https://stats.stackexchange.com/questions/120179/generating-data-with-a-given-sample-covariance-matrix
%class1
Sigma1 = [1 0;0 1];

%generate 1000 data samples
n = 1000;
d = 2;
rng(31);
X1 = randn(n,d)*chol(Sigma1);

%assume zero mean
mean = [0,0];
figure(1);
subplot(1,2,1);
plot(X1(:,1)+mean(1),X1(:,2)+mean(2),'b.');
title('Matrix A');
hold on

%class2
Sigma2 = [2,-2;-2,3];
X2 = randn(n,d)*chol(Sigma2);
subplot(1,2,2);
plot(X2(:,1)+mean(1),X2(:,2)+mean(2),'g.');
title('Matrix B');

%1b
%refer:https://blog.csdn.net/qinminss/article/details/71246607
%calculate the direction of contour
[V1,D1] = eig(Sigma1);

%sort the eigenvalue in descend order
[D1Sort,index1] = sort(diag(D1),'descend');
V1Sort = V1(:,index1);
long1 = sort(V1Sort);
tan1 = long1(2)/long1(1);

%comnpute the radian
radian1 = atan(tan1);
%change into degree
degree1 = rad2deg(radian1);

[V2,D2] = eig(Sigma2);
[D2Sort,index2] = sort(diag(D2),'descend');
V2Sort = V2(:,index2);
long2 = sort(V2Sort);
tan2 = long2(2)/long2(1);
radian2 = atan(tan2);
degree2 = rad2deg(radian2);

%draw contour
%refer:https://www.mathworks.com/help/map/ref/ellipse1.html
[elat1,elon1] = ellipse1(0, 0, [sqrt(D1Sort(1)), axes2ecc(sqrt(D1Sort))], degree1);
figure(1);
hold on
subplot(1,2,1);
plot(elat1,elon1,'r','linewidth',1.2);
[elat2,elon2] = ellipse1(0, 0, [sqrt(D2Sort(2)), axes2ecc(sqrt(D2Sort))], degree2);
subplot(1,2,2);
hold on
plot(elat2,elon2,'r','linewidth',1.2);

%1c
%class 1
vector1(1:size(X1,1)) = 1;
mean1 = (vector1*X1)/size(X1,1);
meanSubtract1 = X1-mean1(vector1,:);
covA = (meanSubtract1.' * meanSubtract1)/(size(X1,1)-1)

%class 2
vector2(1:size(X2,1)) = 1;
mean2 = (vector2*X2)/size(X2,1);
meanSubtract2 = X2-mean2(vector2,:);
covB = (meanSubtract2.' * meanSubtract2)/(size(X2,1)-1)

%2a
P1 = 0.2;
mean1 = [3; 2];
Sigma1 = [1 -1; -1 2];

P2 = 0.7;
mean2 = [5; 4];
Sigma2 = [1 -1; -1 2];

P3 = 0.1;
mean3 = [2; 5];
Sigma3 = [0.5 0.5; 0.5 3];

x = -1:0.1:8;
y = -1:0.1:8;
length(x)
length(y)
[coordx,coordy] = meshgrid(x,y);
coord = [coordx(:) coordy(:)];

mvn1 = reshape(mvnpdf(coord, mean1', Sigma1),91,91);
mvn2 = reshape(mvnpdf(coord, mean2', Sigma2),91,91);
mvn3 = reshape(mvnpdf(coord, mean3', Sigma3),91,91);

ML = MLclassifier(mvn1,mvn2,mvn3);
%plot ML
figure(1);
title('Decision boundaries for a ML classifier');
map = [0.2 0.1 0.5
0.1 0.5 0.8
0.2 0.7 0.6];
colormap(map);
contourf(coordx,coordy,ML,'y','LineWidth',1.2);
hold on
                      
%plot mean
plot(3,2,'yo');
plot(5,4,'yo');
plot(2,5,'yo');
hold on
                      
%first standard deviation contour
[elat1, elon1] = drawContour(mean1, Sigma1);
[elat2, elon2] = drawContour(mean2, Sigma2);
[elat3, elon3] = drawContour(mean3, Sigma3);
plot(elat1, elon1, 'y--', 'LineWidth',1.2);
plot(elat2, elon2, 'y--', 'LineWidth',1.2);
plot(elat3, elon3, 'y--', 'LineWidth',1.2);
                      
MAP = MAPclassifier(mvn1,mvn2,mvn3);
%plot MAP
figure(2);
title('Decision boundaries for a MAP classifier');
map = [0.2 0.7 0.6
0.8 0.7 0.3
0.9 1 0];
colormap(map);
contourf(coordx,coordy,MAP,'b','LineWidth',1.2);
hold on
                      
%plot mean
plot(3,2,'bo');
plot(5,4,'bo');
plot(2,5,'bo');
hold on
                      
%first standard deviation contour
[elat1, elon1] = drawContour(mean1, Sigma1);
[elat2, elon2] = drawContour(mean2, Sigma2);
[elat3, elon3] = drawContour(mean3, Sigma3);
plot(elat1, elon1, 'b--', 'LineWidth',1.2);
plot(elat2, elon2, 'b--', 'LineWidth',1.2);
plot(elat3, elon3, 'b--', 'LineWidth',1.2);
                      
%discuss differnence
figure(3);
title('The differences between the decision boundaries');
contour(coordx,coordy,ML,'y','LineWidth',1.2);
hold on
contour(coordx,coordy,MAP,'b','LineWidth',1.2);
hold on

%2b
SampleNumber = 3000;
SampleNumber1 = SampleNumber * P1
SampleNumber2 = SampleNumber * P2
SampleNumber3 = SampleNumber * P3
                      
data1 = mvnrnd(mean1, Sigma1, 600);
data2 = mvnrnd(mean2, Sigma2, 2100);
data3 = mvnrnd(mean3, Sigma3, 300);
SampleSpace = [data1;data2;data3];
                      
%draw dataset
figure(1);
contour(coordx,coordy,ML,'y','LineWidth',1.2);
hold on
contour(coordx,coordy,MAP,'b','LineWidth',1.2);
plot(data1(:,1),data1(:, 2),'gx',data2(:,1),data2(:, 2),'m+',data3(:,1),data3(:, 2),'yo');
                     
Given = ones(SampleNumber,1);
Given(1: 600) = 1;
Given(601: 2700) = 2;
Given(2701: 3000) = 3;
mvn1b = mvnpdf(SampleSpace, mean1', Sigma1);
mvn2b = mvnpdf(SampleSpace, mean2', Sigma2);
mvn3b = mvnpdf(SampleSpace, mean3', Sigma3);
                      
%calculate a confusion matrix
confusionML = confusionmat(Given, MLclassifier(mvn1b,mvn2b,mvn3b))
confusionMAP = confusionmat(Given, MAPclassifier(mvn1b,mvn2b,mvn3b))

%calculate the experimental P(ε)
sumML = confusionML(1,1) + confusionML(2,2) + confusionML(3,3);
errorML = (3000 - sumML) / 3000;
disp(errorML);
sumMAP = confusionMAP(1,1) + confusionMAP(2,2) + confusionMAP(3,3);
errorMAP = (3000 - sumMAP) / 3000;
disp(errorMAP);

%3a
%refer: https://github.com/liruoteng/MNIST-classification/blob/master/PCA.m
%refer: https://github.com/liruoteng/MNIST-classification/blob/master/PCA.m
%load MNIST data
X = (loadMNISTImages('train-images-idx3-ubyte'))';
Y = loadMNISTLabels('train-labels-idx1-ubyte');
Sigma = cov(X);
               
%use SVD function
[U,S,V] = svd(Sigma);
diagVec = diag(S);
%choose d = 100
Ureduce = U(:,1:100);
z = X * Ureduce *  Ureduce';
               
figure(1);
subplot(1,2,1), displayData(X(8,:));
title('784*N');
subplot(1,2,2), displayData(z(8,:));
title('100*N');

%3b
%refer: https://github.com/liruoteng/MNIST-classification/blob/master/PCA.m
sz = size(diagVec,1);
tr = trace(S);
info = 0;
idx = 0;
               
%find the eigen value that preserves over 95% total information
for i = 1:sz
    info = info + diagVec(i,1);
    if info / tr >= 0.95
        idx = i;
    break;
end
end
               
if idx == 0
    error('could not find');
end
disp(['Dimension: ', num2str(idx)]);

%3c
%different values of d
d = [1 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340 360 380 400 420 440 460 480 500 520 540 560 580 600 620 640 660 680 700 720 740 760 784];
MSE = zeros(40,1);
               
%calculate the mean square error (MSE)
for i = 1:40
    Ureduce = U(:,1:d(i));
    XRec = X * Ureduce * Ureduce';
    d(i);
    MSE(i) = immse(X, XRec);
end
disp('Average MSE is: ');
               
%calculate the average mean square error (MSE)
disp(mean(MSE));
               
figure(1);
plot(d, MSE,'r','linewidth',1.2);
title('MSE (y-axis) versus d (x-axis)');
xlabel('d');
ylabel('MSE');
grid on
hold on

%3d
%manually find the number '8'
displayData(X(18,:));
X8 = X(18,:);
               
%d= {1, 10, 50, 250, 784}
d3 = [1,10,50,250,784];
               
for j = 1:5
    Ureduce = U(:,1:d3(j));
    z(j,:) = X8 * Ureduce *  Ureduce';
end
               
figure(1);
subplot(2,3,1),displayData(z(1,:));
title('d = 1');
subplot(2,3,2),displayData(z(2,:));
title('d = 10');
subplot(2,3,3),displayData(z(3,:));
title('d = 50');
subplot(2,3,4),displayData(z(4,:));
title('d = 250');
subplot(2,3,5),displayData(z(5,:));
title('d = 784');

%3e
figure(1)
plot([1:1:784], diagVec, 'g', 'linewidth', 1.2);
xlabel('d')
ylabel('eigenvalues')
title('eigenvalues (y-axis) versus d (x-axis)')
grid on

%4b
%refer: https://blog.csdn.net/u010084228/article/details/79400474
%load data
data = load('dataset3.txt');
X = data(:,1:2);
y = data(:,3);
               
%add bias terms
X = [ones(length(X), 1),X];
size(X, 2)
theta = zeros(3, 1);
               
%set iteration number
iter = 400;
%set step size
stepSize = 0.001;
%set lambda
lambda = 1;
totalSize = size(X);
standardDev = std(X);
means = mean(X);
               
%stanford-machine learning-exercise2
%feature scaling
for i = 2:totalSize(1, 2)
    X(:,i) = (X(:,i) - means(1, i)) / standardDev(1, i);
end
Jvalue = zeros(iter,1);
lt = length(theta);
thetaValue = zeros(iter,lt);
index = randperm(length(X));
indexi = index(1);
for i = 1:iter
    h = sigmoid(X(indexi,:) * theta);
    hX = sigmoid(X * theta);
    theta(1,1) = theta(1) - stepSize * (h - y(indexi)) * X(indexi,1)';
    theta(2:end,1) = theta(2:end) - stepSize * (h - y(indexi))* X(indexi,2:end)'  +1/length(X) * lambda * theta(2:end,1);
    Jvalue(i) = 1 * (-y' * log(hX + eps) - (1 - y)' * log(1 - hX + eps))  + (lambda / 2) * (theta' * theta - theta(1)^2);
    thetaValue(i,:) = theta';
end
                                                                                    
[JMin, JMinIndex] = min(Jvalue);
OptTheta = thetaValue(JMinIndex,:)';
disp(OptTheta(1));
disp(OptTheta(2));
disp(OptTheta(3));

%4c
figure(1)
plot(linspace(1,400,400), Jvalue,'b', 'lineWidth', 1.2)
xlabel('No.iter')
ylabel('Jvalue')
title('the Cost Function along the Epochs of the SGD')
grid on
hold on
                                                                                
%4d
train = round(sigmoid(X * OptTheta));
res = find(train == y);
%compute accuracy
accuracy = length(res);
disp(accuracy);
                                                                                
%4e
data = load('dataset3.txt');
Xnew = data(:,1:2);
class1 = find(y == 1);
class2 = find(y == 0);
                                                                                    
figure(2)
subplot(1,2,1);
plot(Xnew(class1,1),Xnew(class1,2),'rx', Xnew(class2,1),Xnew(class2,2),'k+');
title('Original');
grid on;
class3 = find(train== 1);
class4 = find(train == 0);
subplot(1,2,2);
plot(Xnew(class3,1),Xnew(class3,2),'rx', Xnew(class4,1),Xnew(class4,2),'k+');
title('Train');
grid on
                                                                                
%4f
figure(3)
class5 = find(train == 1);
class6 = find(train == 0);
%choose two point to form one line
%coefficient
co1 = (-1./OptTheta(3)).* OptTheta(2);
co2 = (-1./OptTheta(3)).* OptTheta(1);
chosenX = [min(X(:,2)), max(X(:,2))];
chosenY = co1.* chosenX + co2;
plot(X(class5, 2), X(class5,3), 'rx', X(class6, 2), X(class6,3), 'k+', chosenX, chosenY,'g--','linewidth',1.2)
title('Decision Boundary of the Classifier')
grid on

%MAP function
function MAP = MAPclassifier(a,b,c)
post1 = a * 0.2;
post2 = b * 0.7;
post3 = c * 0.1;
for i = 1:size(post1,1)
    for j = 1:size(post1,2)
        if (post1(i,j) >= post2(i,j)) && (post1(i,j) >= post3(i,j))
            MAP(i,j) = 1;
        elseif (post2(i,j) >= post1(i,j)) && (post2(i,j) >= post3(i,j))
            MAP(i,j) = 2;
        elseif (post3(i,j) >= post2(i,j)) && (post3(i,j) >= post1(i,j))
            MAP(i,j) = 3;
        end
    end
end
end

%draw contour function
function [elat, elon] = newContour(mean, Sigma)
[eigVects, eigVals] = eig(Sigma);
[sortEigVals, index] = sort(diag(eigVals), 'descend');
sortEigVects = eigVects(:,index(1));
long = sort(sortEigVects);
tan = long(2) / long(1);
radian = atan(tan);
degree = rad2deg(radian);
[elat, elon] = ellipse1(mean(1),mean(2), [sqrt(sortEigVals(1)), axes2ecc(sqrt(sortEigVals))], degree);
end
                      
%ML funcion
function ML = MLclassifier(a,b,c)
likelihood1 = a;
for i = 1:size(likelihood1,1)
    likelihood2 = b;
    likelihood3 = c;
    for j = 1:size(likelihood1,2)
        if (likelihood1(i,j) >= likelihood2(i,j)) && (likelihood1(i,j) >= likelihood3(i,j))
            ML(i,j) = 1;
        elseif (likelihood2(i,j) >= likelihood1(i,j)) && (likelihood2(i,j) >= likelihood3(i,j))
            ML(i,j) = 2;
        elseif (likelihood3(i,j) >= likelihood1(i,j)) && (likelihood3(i,j) >= likelihood2(i,j))
            ML(i,j) = 3;
        end
    end
end
end

%load MNISITlabel function
%refer:https://ww2.mathworks.cn/matlabcentral/answers/373413-how-do-i-import-and-read-the-contentes-of-an-idx3-ubyte-file
function labels = loadMNISTLabels(filename)
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);
numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');
labels = fread(fp, inf, 'unsigned char');
assert(size(labels,1) == numLabels, 'Mismatch in label count');
fclose(fp);
end

%load MNISITimage function
%refer:https://ww2.mathworks.cn/matlabcentral/answers/373413-how-do-i-import-and-read-the-contentes-of-an-idx3-ubyte-file
function images = loadMNISTImages(filename)
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);
numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');
images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);
fclose(fp);
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
images = double(images) / 255;
end

%displayData function
%refer: stanford-machine learning-exercise7-displayData.m
function [h, display_array] = displayData(X, example_width)
if ~exist('example_width', 'var') || isempty(example_width)
    example_width = round(sqrt(size(X, 2)));
end
colormap(gray);
[m n] = size(X);
example_height = (n / example_width);
display_rows = floor(sqrt(m));
display_cols = ceil(m / display_rows);
pad = 1;
display_array = - ones(pad + display_rows * (example_height + pad), ...
                       pad + display_cols * (example_width + pad));
curr_ex = 1;
for j = 1:display_rows
    for i = 1:display_cols
        if curr_ex > m,
            break;
        end
        max_val = max(abs(X(curr_ex, :)));
        display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), pad + (i - 1) * (example_width + pad) + (1:example_width)) = reshape(X(curr_ex, :), example_height, example_width) / max_val;
        curr_ex = curr_ex + 1;
    end
    if curr_ex > m,
        break;
    end
end
h = imagesc(display_array, [-1 1]);
axis image off
drawnow;
end

%sigmoid function
function g = sigmoid(z)
g = zeros(size(z));
g = 1 ./ (1 + exp(-z));
end
