function [f, df] = CG_CLASSIFY(VV,Dim,XX,target,NumOfClasses)
%NumOfClasses=10;

l=Dim;
NumOfLayers=length(Dim)-2;
N = size(XX,1);

% Do decomversion.
w{1} = reshape(VV(1:(l(1)+1)*l(2)),l(1)+1,l(2));
xxx = (l(1)+1)*l(2);
for k=2:NumOfLayers
    w{k} = reshape(VV(xxx+1:xxx+(l(k)+1)*l(k+1)),l(k)+1,l(k+1));
    xxx = xxx+(l(k)+1)*l(k+1);
end
w_class = reshape(VV(xxx+1:xxx+(l(end-1)+1)*l(end)),l(end-1)+1,l(end));


XX = [XX ones(N,1)];
wprobs{1} = 1./(1 + exp(-XX*w{1})); wprobs{1} = [wprobs{1}  ones(N,1)];
for k=2:NumOfLayers
    wprobs{k} = 1./(1 + exp(-wprobs{k-1}*w{k})); wprobs{k} = [wprobs{k} ones(N,1)];
end

targetout = exp(wprobs{end}*w_class);
targetout = targetout./repmat(sum(targetout,2),1,NumOfClasses);
f = -sum(sum( target(:,1:end).*log(targetout))) ;

IO = (targetout-target(:,1:end));
Ix_class=IO;
dw_class =  wprobs{end}'*Ix_class;

Ix{NumOfLayers} = (Ix_class*w_class').*wprobs{end}.*(1-wprobs{end});
Ix{NumOfLayers} = Ix{NumOfLayers}(:,1:end-1);
if NumOfLayers>1
    dw{NumOfLayers} =  wprobs{end-1}'*Ix{NumOfLayers};
elseif NumOfLayers==1
    dw{NumOfLayers} =  XX'*Ix{NumOfLayers};
end

for k=NumOfLayers-1:-1:1
    Ix{k} = (Ix{k+1}*w{k+1}').*wprobs{k}.*(1-wprobs{k});
    Ix{k} = Ix{k}(:,1:end-1);
    if k>1
        dw{k} =  wprobs{k-1}'*Ix{k};
    end
    if k==1
        dw{k} =  XX'*Ix{k};
    end
end

df=[];
for k=1:NumOfLayers
    df=[df dw{k}(:)'];
end
df = [df dw_class(:)']';
