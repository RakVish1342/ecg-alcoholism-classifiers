clear all;
close all;

%--------------------------------------------------------------------------------------------------------------------
pkg load signal;
disp('package loaded ... \n');

sample_type1 = 'alcoholic/ALC (';
sample_type2 = 'normal/NOR (';
m1 = 38; m2 = 29; %no of ALC or NOR samples
n = 10+13; %no of features per sample


classifier_data = zeros(m1+m2, n);
for i = 1:m1
  disp(strcat('\n TRAINING---', sample_type1, num2str(i), '... \n') ); 
  features1 = iir_butt(i, sample_type1);
  features2 = freq_extract(i, sample_type1);
  classifier_data(i, :) = [features1, features2];
end

for i = 1:m2
  disp(strcat('\n TRAINING---', sample_type2, num2str(i), '... \n') ); 
  features1 = iir_butt(i, sample_type2);
  features2 = freq_extract(i, sample_type2);
  classifier_data(m1+i, :) = [features1, features2];
end

csvwrite('data_all.txt', classifier_data);


disp(' ==== END ==== ');