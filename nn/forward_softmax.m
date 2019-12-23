function P = forward_softmax(X, which_dim)
% Px = softmax(Y, which_dim)

expX = exp(X - repmat(max(X,[],which_dim),1,size(X,2))); % Use shifts to ensure numerical stability
P = expX ./ repmat(sum(expX, which_dim),1,size(X,2));
P = P + eps; % Avoid explicit zero
% Px = softmax(X);
