function y=estimateDerivative(x,k)

% Estimates the derivative of a signal using a local weghted average.
% Used during the extraction of a rhythmic signature.
% (c) Aggelos Pikrakis, pikrakis@unipi.gr

Lx=length(x);
y=zeros(1,length(x));

limi=2;
for i=limi+1:Lx-limi
    if i<=k
        w=[1./[-i+1:-1] 0 1./[1:i-1]];
        y(i)=sum(x(1:2*i-1).*w);
    elseif i>length(x)-k
        w=[1./[-(length(x)-i):-1] 0 1./[1:length(x)-i]];
        y(i)=sum(x(i-(length(x)-i):end).*w);
    else
        w=[1./[-k:-1] 0 1./[1:k]];
        y(i)=sum(x(i-k:i+k).*w);
    end
end
