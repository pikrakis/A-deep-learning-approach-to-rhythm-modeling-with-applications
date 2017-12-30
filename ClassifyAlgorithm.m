function ClassifyAlgorithm(ScratchFolder,TestingSet,OutputListFile)

% This is the classification algorithm.
% ScratchFolder: the scratch folder where the feature files and training data 
% have been stored.
% TestingSet: full path to the file which contains the listing of the test
% files (one test file per line, no header).
% OutputListFile: full path to the file where the results will be stored
% (one result per line, no header).
% (c) Aggelos Pikrakis, pikrakis@unipi.gr

lastlag=800;
load([ScratchFolder filesep 'TrainingData'],'w','w_class','genres');

NumOfClasses=length(genres);
NumOfLayers=length(w);
fin=fopen(TestingSet);
fout=fopen(OutputListFile,'w');
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
    
    % classification takes place here
    maxvalues=zeros(1,length(genres));
    for m=1:size(D,1)
        data_t=expFeature(D(m,:),0.005);
        data_t = [data_t ones(1,1)];
        for k=1:NumOfLayers
            data_t = 1./(1 + exp(-data_t*w{k}));
            data_t = [data_t  ones(1,1)];
        end
        tout = exp(data_t*w_class);
        tout = tout/sum(tout);
        [indiff,indtout]=max(tout);     
        maxvalues(indtout)=maxvalues(indtout)+1;
    end
    [indiff,indtout]=max(maxvalues);     
    fprintf(fout,'%s\t%s\n',filestr,genres{indtout});
    %%%%%%%%%%%%%%%
    
    
end
fclose(fin);
fclose(fout);

