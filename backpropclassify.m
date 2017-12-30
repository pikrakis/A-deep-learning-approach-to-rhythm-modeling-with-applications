function [w,w_class]=backpropclassify(NumOfClasses,maxepoch,vishid,hidbiases,batchdata,batchtargets)

% Back-propagation training of the deep learning architecture after the
% individual RBMs have been trained. This code is a re-engineered version
% of the publicly available Hinton's code on digit recognition. 
% 

[numcases numdims numbatches]=size(batchdata);
N=numcases;

%%%% PREINITIALIZE WEIGHTS OF THE DISCRIMINATIVE MODEL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NumOfLayers=length(vishid); %my


for k=1:NumOfLayers
    w{k}=[vishid{k};hidbiases{k}];
end
w_class = 0.1*randn(size(w{end},2)+1,NumOfClasses);


%%%%%%%%%% END OF PREINITIALIZATION OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k=1:NumOfLayers
    l(k)=size(w{k},1)-1;
end
l(end+1)=size(w_class,1)-1;
l(end+1)=NumOfClasses;
test_err=[];
train_err=[];


for epoch = 1:maxepoch
    
    %%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    err=0;
    err_cr=0;
    counter=0;
    [numcases numdims numbatches]=size(batchdata);
    N=numcases;
    for batch = 1:numbatches
        data = [batchdata(:,:,batch)];
        target = [batchtargets(:,:,batch)];
        data = [data ones(N,1)]; % for the bias
        for k=1:NumOfLayers
            data = 1./(1 + exp(-data*w{k})); data = [data  ones(N,1)]; % for the bias
        end
        targetout = exp(data*w_class); % output layer
        targetout = targetout./repmat(sum(targetout,2),1,NumOfClasses); % divides each element of targetout by the sum of elements of the line
        
        [I J]=max(targetout,[],2); % detect maximum value at each line
        [I1 J1]=max(target,[],2);
        counter=counter+length(find(J==J1));
        err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
    end
    train_err(epoch)=(numcases*numbatches-counter);    
    train_crerr(epoch)=err_cr/numbatches;
    fprintf('epoch = %d, training error = %f\n',epoch,train_err(end)/(numcases*numbatches));
    
    %%%%%%%%%%%%%% END OF COMPUTING TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    tt=0;
    for batch = 1:numbatches/10       
        %%%%%%%%%%% COMBINE 10 MINIBATCHES INTO 1 LARGER MINIBATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tt=tt+1;
        data=[];
        targets=[];
        for kk=1:10
            data=[data
                batchdata(:,:,(tt-1)*10+kk)];
            targets=[targets
                batchtargets(:,:,(tt-1)*10+kk)];
        end
        
        %%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        max_iter=3;
        
        if epoch<6  % First update top-level weights holding other weights fixed.
            N = size(data,1);
            XX = [data ones(N,1)];
            for k=1:NumOfLayers
                XX = 1./(1 + exp(-XX*w{k}));
                if k<NumOfLayers
                    XX = [XX  ones(N,1)]; % for the bias
                end
            end
            
            VV = [w_class(:)']';
            Dim = [l(end-1); l(end)];
            [X, fX] = minimize(VV,'CG_CLASSIFY_INIT',max_iter,Dim,XX,targets,NumOfClasses);
            w_class = reshape(X,l(end-1)+1,l(end));
        else
            VV=[];
            for k=1:NumOfLayers
                VV=[VV w{k}(:)'];
            end
            VV=[VV w_class(:)']';
            
            Dim=l';
            
            [X, fX] = minimize(VV,'CG_CLASSIFY',max_iter,Dim,data,targets,NumOfClasses);
            w{1} = reshape(X(1:(l(1)+1)*l(2)),l(1)+1,l(2));
            xxx = (l(1)+1)*l(2);
            for k=2:NumOfLayers
                w{k} = reshape(X(xxx+1:xxx+(l(k)+1)*l(k+1)),l(k)+1,l(k+1));
                xxx = xxx+(l(k)+1)*l(k+1);
            end
            w_class = reshape(X(xxx+1:xxx+(l(end-1)+1)*l(end)),l(end-1)+1,l(end));
            
        end
        %%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    end    
end



