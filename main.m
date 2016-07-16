% A Movie Recommender System Using Collaborative Filtering
% Author: RUI HU
% Web: http://hurui.info/


%% ============== Step 1: Load Traning Data ===============

%  Y - is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 943 users
%  R - is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i

fprintf('Loading training data - movie ratings dataset.\n\n');
load ('movies_training.mat');


%% ============== Step 2: Randomly set movie ratings for a new user that we just observed ===============
%  Note: You can put in your own ratings here.


movieList = loadMovieList();

%  Initialize my ratings
my_ratings = zeros(1682, 1);

% Check the file movie_idx.txt for id of each movie in our dataset
% For example, Toy Story (1995) has ID 1, so to rate it "4", we do
my_ratings(1) = 4;

% Or suppose you did not enjoy Silence of the Lambs (1991), you can set
my_ratings(98) = 2;

% I randomly set some ratings here, but feel free to change them
my_ratings(10) = 3;
my_ratings(12)= 5;
my_ratings(14) = 4;
my_ratings(30)= 5;
my_ratings(40)= 3;
my_ratings(69) = 5;
my_ratings(180) = 4;
my_ratings(230) = 5;
my_ratings(344)= 5;


%% ================== Step 3: Training Recommender Using Collaborative Filtering Algorithm ====================
fprintf('\nTraining Recommender Using Collaborative Filtering Algorithm...\n');

%  Add our own ratings to the data matrix
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

%  Normalize Ratings - scaling
[Ynorm, Ymean] = normalizeRatings(Y, R);

num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;

% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);
initial_parameters = [X(:); Theta(:)];

% Set options for fmincg - 100 iterations
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set Regularization
lambda = 10;

% Run Gradient Descend - fmincg
theta = fmincg (@(t)(cofiCostFunc(t, Y, R, num_users, num_movies, ...
                                num_features, lambda)), ...
                initial_parameters, options);

% Unfold the returned theta back into U and W
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), ...
                num_users, num_features);

fprintf('Recommender learning completed.\n');


%% ================== Step 4: Make recommendation for you ====================
%  After training the model, now make recommendations by computing the predictions matrix.
%

p = X * Theta';
my_predictions = p(:,1) + Ymean;

movieList = loadMovieList();

[r, ix] = sort(my_predictions, 'descend');
fprintf('\nTop recommendations for you:\n');
for i=1:10
    j = ix(i);
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), ...
            movieList{j});
end

fprintf('\n\nOriginal ratings provided:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), ...
                 movieList{i});
    end
end
