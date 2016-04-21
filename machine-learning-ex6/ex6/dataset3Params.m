function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
C_vec = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
table=zeros(64,3);
count=0;
for i = 1:length(C_vec)
    for j = 1:length(sigma_vec)
        model=svmTrain(X, y, C_vec(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(j)));
        predictions=svmPredict(model,Xval);
        count=count+1;
        table(count,1)=C_vec(i);
        table(count,2)=sigma_vec(j);
        table(count,3)=mean(double(predictions ~= yval));
    end
end
[minError,minErrorIdx]=min(table(:,3));
C=table(minErrorIdx,1);
sigma=table(minErrorIdx,2);

disp(['minError: ',num2str(minError)]);
disp(['minError Index: ',num2str(minErrorIdx)]);
disp(['C: ',num2str(C)]);
disp(['Sigma: ',num2str(sigma)]);








% =========================================================================

end
