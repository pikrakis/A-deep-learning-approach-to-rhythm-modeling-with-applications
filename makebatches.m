function [batchdata,batchtargets]=makebatches(TrainingSet,ScratchFolder,lastlag,batchsize)

% This is an essential function which prepares the training set by postprocessing the feature
% files in the scratch folder. It is called by the TrainingAlgorithm.
% m-file.
% Aggelos Pikrakis, pikrakis@unipi.gr

digitdata=[];
targets=[];
genres=detectGenres(TrainingSet);
NumOfClasses=length(genres);
fin=fopen(TrainingSet);

while ~feof(fin)
    filestr=fscanf(fin,'%s\t',1);
    indfilesep=strfind(filestr,filesep);
    indfilesep=indfilesep(end)+1;
    inddot=strfind(filestr,'.');
    if isempty(inddot)
        inddot=length(filestr);
    else
        inddot=inddot(end)-1;
    end
    foutname=[ScratchFolder filesep filestr(indfilesep:inddot) '.features.txt'];
    D=load(foutname);
    D=D(:,1:lastlag);    
    for m=1:size(D,1)
        D(m,:)=expFeature(D(m,:),0.005);
    end
    
    digitdata = [digitdata; D];    
    blabel=fscanf(fin,'%s\n',1);
    i=1;
    while i<=length(genres)
        if strcmp(genres{i},blabel)
            break;
        else
            i=i+1;
        end
    end
    
    labelid=zeros(1,NumOfClasses);
    labelid(i)=1;
    targets = [targets; repmat(labelid, size(D,1), 1)];
    
end
fclose(fin);

totnum=size(digitdata,1);
fprintf(1, 'Size of the training dataset= %5d \n', totnum);

rand('state',sum(100*clock())); 
randomorder=randperm(totnum);
numbatches=floor(totnum/batchsize)-1;
numdims  =  size(digitdata,2);
batchdata = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, NumOfClasses, numbatches);

for b=1:numbatches
    batchdata(:,:,b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    batchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
save([ScratchFolder filesep 'TrainingData'],'genres');



