clear all;
close all;

%--------------------------------------------------------------------------------------------------------------------
pkg load signal;
disp('package loaded ... \n');

sample_type1 = 'alcoholic/ALC (';
sample_type2 = 'normal/NOR (';
m1 = 38; %no of ALC samples
m2 = 29; %no of NOR samples
n = 10; %no of features per sample

classifier_data = zeros(m1+m2, n);
for i = 1:m1
  disp(strcat('\n TRAINING---', sample_type1, num2str(i), '... \n') ); 
  features = iir_butt(i, sample_type1);
  classifier_data(i, :) = features;
end

for i = 1:m2
  disp(strcat('\n TRAINING---', sample_type2, num2str(i), '... \n') ); 
  features = iir_butt(i, sample_type2);
  classifier_data(m1+i, :) = features;
end

csvwrite('data_time_nonlin_corrected.txt', classifier_data);


disp(' ==== END ==== ');