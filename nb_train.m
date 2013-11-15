
[spmatrix, tokenlist, trainCategory] = readMatrix('MATRIX.TRAIN.1400');

trainMatrix = full(spmatrix);
numTrainDocs = size(trainMatrix, 1);
numTokens = size(trainMatrix, 2);

% trainMatrix is now a (numTrainDocs x numTokens) matrix.
% Each row represents a unique document (email).
% The j-th column of the row $i$ represents the number of times the j-th
% token appeared in email $i$. 

% tokenlist is a long string containing the list of all tokens (words).
% These tokens are easily known by position in the file TOKENS_LIST

% trainCategory is a (1 x numTrainDocs) vector containing the true 
% classifications for the documents just read in. The i-th entry gives the 
% correct class for the i-th email (which corresponds to the i-th row in 
% the document word matrix).

% Spam documents are indicated as class 1, and non-spam as class 0.
% Note that for the SVM, you would want to convert these to +1 and -1.

% YOUR CODE HERE

% using octave

theta = zeros(2, numTokens);
pos = find(trainCategory == 1);
neg = find(trainCategory == 0);

pos_count = size(pos,2)
neg_count = size(neg,2)

size(spmatrix)
size(trainCategory)

prior = pos_count / ( pos_count + neg_count);

theta(1,:) = sum(spmatrix(neg,:),1);
theta(2,:) = sum(spmatrix(pos,:),1);

theta = theta + 1;

theta(1,:) = theta(1,:) ./ sum (theta(1,:));
theta(2,:) = theta(2,:) ./ sum (theta(2,:));