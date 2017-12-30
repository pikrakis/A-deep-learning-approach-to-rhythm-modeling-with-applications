function FeatureExtractionSingleFile(ScratchFolder,filestr,ltwin,ltstep,stwin,ststep,meter_end,spacing)

indfilesep=strfind(filestr,filesep);
indfilesep=indfilesep(end)+1;

inddot=strfind(filestr,'.');
if isempty(inddot)
    inddot=length(filestr)
else
    inddot=inddot(end)-1;
end


foutname=[ScratchFolder filesep filestr(indfilesep:inddot) '.features.txt'];
fid=fopen(foutname,'w');

[x,Fs]=wavread(filestr,'size');
L=x(1);
maxindex=round(meter_end/ststep)+1;
ltwin=round(ltwin*Fs);
ltstep=round(ltstep*Fs);
lcount=0;
counter=1;
while counter+ltwin-1<=L
    frame=wavread(filestr,[counter counter+ltwin-1]);
    szframe=size(frame);
    if szframe(2)==2
        frame=sum(frame,2);
    end
    lcount=lcount+1;
    fprintf('%d -> ',lcount);
    [Feat] = chroma_based_mfcc(frame+eps, Fs, round(stwin*Fs),round(ststep*Fs),spacing);   
    B=zeros(1,maxindex+1);
    for i=1:maxindex+1
        Eucl=sqrt(sum((Feat(:,i+1:end)-Feat(:,1:end-i)).^2));
        B(i)=mean(Eucl);
    end
    fprintf(fid,'%f ',B);
    fprintf(fid,'\n');    
    counter=counter+ltstep;
end
fclose(fid);

