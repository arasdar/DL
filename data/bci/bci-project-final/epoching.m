
clear all
[signal, states, parameters]=load_data;
timesec=0.8;
fs=256;
N_samples=round(timesec*fs);
N_Channels=16;

Freqmin=0.1;
Freqmax=30;

Filterorder=round(3*(fs/Freqmin));%length of the filter in time domain at leas 3 times the lowest activity (lowest frequency in Hz)
FWeights= fir1(Filterorder,[Freqmin/(fs/2) Freqmax/(fs/2)]);
for i=1:N_Channels
    signal(:,i)=filtfilt(FWeights,1, signal(:,i)); 
end
signal=zscore(signal);
Non_Zero_index=find(states.StimulusBegin);
select=(states.StimulusBegin(Non_Zero_index)-states.StimulusBegin(Non_Zero_index-1))==1;
First_Intensification_Index=Non_Zero_index(select);
IntensificationNum=length(First_Intensification_Index);


% %first intensification only
% select=(states.StimulusBegin(Non_Zero_index)-states.StimulusBegin(Non_Zero_index-1))==1;
% F_Index=Non_Zero_index.*select;
% %the vector First_intensification_index contains the index of the first intensification 
% %First_intensification_index=F_Index(find(F_Index));
% First_Intensification_Index=F_Index(F_Index~=0);
% IntensificationNum=length(First_Intensification_Index);

Intensification_Data=zeros([IntensificationNum N_samples N_Channels]);
Intensification_SType=zeros([IntensificationNum N_samples]);
Intensification_SCode=zeros([IntensificationNum N_samples]);


for counter=1:IntensificationNum  
  Intensification_Data(counter,:,:)=signal((First_Intensification_Index(counter):First_Intensification_Index(counter)+N_samples-1),:);
%   Intensification_SType(counter,:)=states.StimulusType(First_intensification_index(counter):First_intensification_index(counter)+N_samples-1);
%   Intensification_SCode(counter,:)=states.StimulusCode(First_intensification_index(counter):First_intensification_index(counter)+N_samples-1);
 Intensification_SType(counter,:)=states.StimulusType(First_Intensification_Index(counter));
 Intensification_SCode(counter,:)=states.StimulusCode(First_Intensification_Index(counter));

end