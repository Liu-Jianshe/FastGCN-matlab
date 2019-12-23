function dLdX = backward_softmax(P, which_dim, dLdP)
% P = softmax(Y, which_dim)

P = P - eps; % Account for +eps in forward_softmax
dLdX = ( dLdP - repmat(sum(dLdP .* P, which_dim),1,size(P,2)) ) .* P;
