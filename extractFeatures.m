function extractFeatures(PathToScratchFolder,PathToFeatureExtractionListFile)

% This is the feature extractor.
% PathToScratchFolder: full path to the folder where the feature files will
% be stored (one file for each line in ,PathToFeatureExtractionListFile).
% PathToFeatureExtractionListFile: file which contains a listing of ALL the
% audio files used in the competition (one file per line).
% The algorithms creates one feature file per audio file. 
% IT IS IMPORTANT THAT FILENAMES ARE UNIQUE, OTHERWISE OVERWRITING WILL
% TAKE PLACE.

fin=fopen(PathToFeatureExtractionListFile);
while ~feof(fin)
    audiofile=fgetl(fin)
    FeatureExtractionSingleFile(PathToScratchFolder,audiofile,10,10,0.1,0.005,4,12);
end
fclose(fin);