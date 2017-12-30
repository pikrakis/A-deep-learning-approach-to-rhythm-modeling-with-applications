function [f, df] = CG_CLASSIFY_INIT(VV,Dim,w3probs,target,NumOfClasses)
%NumOfClasses=10;
l1 = Dim(1);
l2 = Dim(2);
N = size(w3probs,1);
% Do decomversion.
w_class = reshape(VV,l1+1,l2);
w3probs = [w3probs  ones(N,1)];

targetout = exp(w3probs*w_class);
targetout = targetout./repmat(sum(targetout,2),1,NumOfClasses);
f = -sum(sum( target(:,1:end).*log(targetout))) ;
IO = (targetout-target(:,1:end));
Ix_class=IO;
dw_class =  w3probs'*Ix_class;

df = [dw_class(:)']';

