function [ceps,freqresp,fb,fbrecon,freqrecon] = chroma_based_mfcc(input, samplingRate, winlength,winstep,spacing)
% Computes a version of the MFCCs, according to which the centers of the
% filters of the filterbank coincide with the semitones of the chromatic
% scale. This function is used during the feature extraction stage.
% This code is a variant of Slaney's code.
% Aggelos Pikrakis, pikrakis@unipi.gr


global mfccDCTMatrix mfccFilterWeights

[r c] = size(input);
if (r > c)
    input=input';
end

%	Filter bank parameters
lowestFrequency = 110;
fftSize = winlength;
cepstralCoefficients = 13;
windowSize = winlength;
windowStep = winstep;

if spacing==6
    totalFilters = 33; % up to 5000 Hz for whole-tone spacing.
    freqs=lowestFrequency*2.^([0:totalFilters+2]/6);
elseif spacing==12
    totalFilters =64;
    freqs=lowestFrequency*2.^([0:totalFilters+2]/12);
end

lower = freqs(1:totalFilters);
center = freqs(2:totalFilters+1);
upper = freqs(3:totalFilters+2);

mfccFilterWeights = zeros(totalFilters,fftSize);
triangleHeight = 2./(upper-lower);
fftFreqs = (0:fftSize-1)/fftSize*samplingRate;

for chan=1:totalFilters
    mfccFilterWeights(chan,:) = ...
        (fftFreqs > lower(chan) & fftFreqs <= center(chan)).* ...
        triangleHeight(chan).*(fftFreqs-lower(chan))/(center(chan)-lower(chan)) + ...
        (fftFreqs > center(chan) & fftFreqs < upper(chan)).* ...
        triangleHeight(chan).*(upper(chan)-fftFreqs)/(upper(chan)-center(chan));
end
%semilogx(fftFreqs,mfccFilterWeights')
%axis([lower(1) upper(totalFilters) 0 max(max(mfccFilterWeights))])

hamWindow = 0.54 - 0.46*cos(2*pi*(0:windowSize-1)/windowSize);

if 0					% Window it like ComplexSpectrum
    a = .54;
    b = -.46;
    wr = sqrt(windowStep/windowSize);
    phi = pi/windowSize;
    hamWindow = 2*wr/sqrt(4*a*a+2*b*b)* ...
        (a + b*cos(2*pi*(0:windowSize-1)/windowSize + phi));
end

mfccDCTMatrix = 1/sqrt(totalFilters/2)*cos((0:(cepstralCoefficients-1))' * ...
    (2*(0:(totalFilters-1))+1) * pi/2/totalFilters);
mfccDCTMatrix(1,:) = mfccDCTMatrix(1,:) * sqrt(2)/2;

if 0
    preEmphasized = filter([1 -.97], 1, input);
else
    preEmphasized = input;
end


cols = fix((length(input)-windowSize)/windowStep);
if cols==0
    cols=1;
end
% Allocate all the space we need for the output arrays.
ceps = zeros(cepstralCoefficients, cols);
if (nargout > 1) freqresp = zeros(fftSize/2, cols); end;
if (nargout > 2) fb = zeros(totalFilters, cols); end;

% Invert the filter bank center frequencies.  For each FFT bin
% we want to know the exact position in the filter bank to find
% the original frequency response.  The next block of code finds the
% integer and fractional sampling positions.
if (nargout > 4)
    fr = (0:(fftSize/2-1))'/(fftSize/2)*samplingRate/2;
    j = 1;
    for i=1:(fftSize/2)
        if fr(i) > center(j+1)
            j = j + 1;
        end
        if j > totalFilters-1
            j = totalFilters-1;
        end
        fr(i) = min(totalFilters-.0001, ...
            max(1,j + (fr(i)-center(j))/(center(j+1)-center(j))));
    end
    fri = fix(fr);
    frac = fr - fri;
    
    freqrecon = zeros(fftSize/2, cols);
end

% Ok, now let's do the processing.  For each chunk of data:
%    * Window the data with a hamming window,
%    * Shift it into FFT order,
%    * Find the magnitude of the fft,
%    * Convert the fft data into filter bank outputs,
%    * Find the log base 10,
%    * Find the cosine transform to reduce dimensionality.

% Related to pshychoacoustics
% hann=hanning(windowSize);
% df=samplingRate/windowSize;
% k=0:windowSize/2-1;
% f=k*df;
% % F in Barks
% Z=13*atan(0.00076*k*samplingRate/windowSize)+3.5*(atan(k*samplingRate/windowSize/7500)).^2;
% %Original JND
% Tq=3.64*(f/1000).^(-0.8)-6.5*exp(-0.6*((f/1000)-3.3).^2)+10^(-3)*(f/1000).^4;
%%% eof psychoacoustics

for start=0:cols-1
    first = start*windowStep + 1;
    last = first + windowSize-1;
    
    %ttemp=preEmphasized(first:last);
    %[Ptm,tms,Pnm,tns,Tg,ind]=psAcoustics(ttemp,hann,f,Z,Tq);
    
    %kl=find(ind==1);
    %ind(kl)=[];
    
    %ttemp_fft=fft(ttemp);
    %ttemp_fft(ind)=0;
    %ttemp_fft(length(ttemp_fft)+2-ind)=0;
    %ttemp=real(ifft(ttemp_fft))+eps;
    
    fftData = zeros(1,fftSize);
    fftData(1:windowSize) = preEmphasized(first:last).*hamWindow;
    %fftData(1:windowSize) = ttemp.*hamWindow;
    fftMag = abs(fft(fftData));
    earMag = log10(mfccFilterWeights * fftMag');
    
    ceps(:,start+1) = mfccDCTMatrix * earMag;
    if (nargout > 1) freqresp(:,start+1) = fftMag(1:fftSize/2)'; end;
    if (nargout > 2) fb(:,start+1) = earMag; end
    if (nargout > 3)
        fbrecon(:,start+1) = ...
            mfccDCTMatrix(1:cepstralCoefficients,:)' * ...
            ceps(:,start+1);
    end
    if (nargout > 4)
        f10 = 10.^fbrecon(:,start+1);
        freqrecon(:,start+1) = samplingRate/fftSize * ...
            (f10(fri).*(1-frac) + f10(fri+1).*frac);
    end
end

% OK, just to check things, let's also reconstruct the original FB
% output.  We do this by multiplying the cepstral data by the transpose
% of the original DCT matrix.  This all works because we were careful to
% scale the DCT matrix so it was orthonormal.
if 1 & (nargout > 3)
    fbrecon = mfccDCTMatrix(1:cepstralCoefficients,:)' * ceps;
    %	imagesc(mt(:,1:cepstralCoefficients)*mfccDCTMatrix);
end;


