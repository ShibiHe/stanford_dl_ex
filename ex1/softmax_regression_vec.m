function [f,g] = softmax_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  Sigma=exp(theta'*X);
  Sigma(end+1,:)=ones(1,size(Sigma,2));
  Sigma(end+1,:)=sum(Sigma);
  for i=1:m
      k=y(i);
      poss=log(Sigma(k,i)/Sigma(end,i));
      f=f+poss;
  end
  f=-f;
  
  for k=1:9
      for i=1:m
          if y(i)==k
              dot=1;
          else
              dot=0;
          end
          g(:,k)=g(:,k)+X(:,i)*(dot-Sigma(k,i)/Sigma(end,i));
      end
  end
  g=-g;
  
  g=g(:); % make gradient a vector for minFunc
end
