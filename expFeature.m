function y=expFeature(x,ststep)

minindex=round(60/300/ststep);
nei=round(.1/ststep);
y=x;
y(1:minindex)=0;
y=estimateDerivative(estimateDerivative(x,nei),nei);
y(1:minindex)=0;
y=(y-mean(y))./(std(y));
y=(1./(1+exp(-y)));


