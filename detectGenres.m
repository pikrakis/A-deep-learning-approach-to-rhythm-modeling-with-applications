function genres=detectGenres(TrainingSet)

fin=fopen(TrainingSet);
counter=0;
while ~feof(fin)
    filestr=fscanf(fin,'%s\t',1);
    counter=counter+1;
    genres{counter}=fscanf(fin,'%s\n',1);    
end
fclose(fin);
genres=unique(genres);
