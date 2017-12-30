function TrainingAlgorithm(ScratchFolder,TrainingSet)

% This is the training algorithm.
% ScratchFolder: the folder where the feature files have been stored. This
% is also the location where the training data (output of this algorithm
% will bw stored).
% TrainingSet: full path to the training file which contains a list of the
% audio files used for training.
% The function creates a TrainingData.mat file in the scratch folder.

numhid=400*ones(1,5);
maxepoch=100;
backpropepoch=100;

fprintf('Assembling training set from individual feature files in %s\n',ScratchFolder);
[batchdata,batchtargets]=makebatches(TrainingSet,ScratchFolder,800,100);
fprintf('Done (Assembling training set from individual feature files in %s)\n',ScratchFolder);
lastlag=size(batchdata,2);

for layer=1:length(numhid)
    [numcases numdims numbatches]=size(batchdata);
    fprintf('Pretraining: Layer %d, RBM: %d - %d \n',layer,numdims,numhid(layer));
    [vishid{layer},visbiases{layer},hidbiases{layer},batchposhidprobs{layer}]=rbm(maxepoch,numhid(layer),batchdata,1);
    fprintf('Done (Pretraining: Layer %d, RBM: %d - %d) \n',layer,numdims,numhid(layer));
    batchdata=batchposhidprobs{layer};    
end

fprintf('Re-Assembling training set from individual feature files in %s\n',ScratchFolder);
[batchdata,batchtargets]=makebatches(TrainingSet,ScratchFolder,800,100);
fprintf('Done (Re-Assembling training set from individual feature files in %s)\n',ScratchFolder);

fprintf('Training deep network as a whole with back-propagation\n');
[w,w_class]=backpropclassify(size(batchtargets,2),backpropepoch,vishid,hidbiases,batchdata,batchtargets);
fprintf('Done (Training deep network as a whole with back-propagation)\n');

fprintf('Saving training data in %s \n', ScratchFolder);
save([ScratchFolder filesep 'TrainingData'],'w','w_class','-append');
fprintf('Done (Saving training data in %s)\n', ScratchFolder);

fprintf('Finished Training\n');
fprintf('Created file TrainingData.mat in %s\n', ScratchFolder);
