
[spmatrix, tokenlist, category] = readMatrix('MATRIX.TEST');

testMatrix = full(spmatrix);
numTestDocs = size(testMatrix, 1);
numTokens = size(testMatrix, 2);

% Assume nb_train.m has just been executed, and all the parameters computed/needed
% by your classifier are in memory through that execution. You can also assume 
% that the columns in the test set are arranged in exactly the same way as for the
% training set (i.e., the j-th column represents the same token in the test data 
% matrix as in the original training data matrix).

% Write code below to classify each document in the test set (ie, each row
% in the current document word matrix) as 1 for SPAM and 0 for NON-SPAM.

% Construct the (numTestDocs x 1) vector 'output' such that the i-th entry 
% of this vector is the predicted class (1/0) for the i-th  email (i-th row 
% in testMatrix) in the test set.
output = zeros(numTestDocs, 1);

%---------------
% YOUR CODE HERE

% using octave
logtheta = log(theta)';
logtheta_complement = log(1-theta)';
logprior = log(prior);
logprior_complement = log(1-prior);
    
joint_ll = testMatrix * logtheta + (1-testMatrix) * logtheta_complement; 
joint_ll = joint_ll .+ [logprior_complement logprior];

[maxjoint_ll, max_idx] = max(joint_ll,[], 2);

output = max_idx .- 1;

% determine most informative features
% [s,si] = sort(logtheta(:,2) - logtheta(:,1), "descend")
% top_5 = si(1:5)

%---------------


% Compute the error on the test set
error=0;
for i=1:numTestDocs
  if (category(i) ~= output(i))
    error=error+1;
  end
end

%Print out the classification error on the test set
error/numTestDocs


