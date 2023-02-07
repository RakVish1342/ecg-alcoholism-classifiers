clear all;
close all;

%--------------------------------------------------------------------------------------------------------------------
%pkg load signal;
%disp('package loaded ... \n');

sample_type1 = 'alcoholic/ALC (';
sample_type2 = 'normal/NOR (';
m = 10; %no of ALC or NOR samples
n = 13; %no of features per sample


classifier_data = zeros(2*m, n);
for i = 1:m
  disp(strcat('\n TRAINING---', sample_type1, num2str(i), '... \n') ); 
  features = freq_extract(i, sample_type1);
  classifier_data(i, :) = features;
end

for i = 1:m
  disp(strcat('\n TRAINING---', sample_type2, num2str(i), '... \n') ); 
  features = freq_extract(i, sample_type2);
  classifier_data(m+i, :) = features;
end

csvwrite('data_freq.txt', classifier_data);


disp(' ==== END ==== ');