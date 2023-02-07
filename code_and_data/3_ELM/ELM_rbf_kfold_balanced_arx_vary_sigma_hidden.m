function [TV,F,normalized_data] = ELM_rbf_kfold_balanced_arx_vary_sigma_hidden(DataSet, AlcArx, NorArx, Elm_Type, ActivationFunction)
%%%%%%%%%%%
REGRESSION=0;
CLASSIFIER=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dataset=csvread(DataSet);
alc_arx=csvread(AlcArx);
nor_arx=csvread(NorArx);
data_arx=[alc_arx;nor_arx];
dataset2=zeros(size(dataset,1),size(dataset,2)+1);
dataset2(1:size(dataset2,1),2:size(dataset2,2))=dataset;
dataset2(1:39,1)=1;
dataset2(39:67,1)=-1;
dataset2(30:38,:)=[];
dataset2=[dataset2 data_arx];
dataset2(29:30,:)=[];
dataset2(:,25)=[];
dataset2(:,26)=[];
dataset2=dataset2';
min_values=zeros(size(dataset2,1),1);
max_values=zeros(size(dataset2,1),1);
datanew=zeros(size(dataset2,1),size(dataset2,2));
datanew(1,:)=dataset2(1,:);

for i=2:size(dataset2,1)
    min_values(i,1)=min(dataset2(i,:));
    max_values(i,1)=max(dataset2(i,:));
end
for i=2:size(dataset2,1)
    for j=1:size(dataset2,2)
        datanew(i,j)=(((dataset2(i,j))-min_values(i,1))/(max_values(i,1)-min_values(i,1)));
    end
end
datanew=datanew';
normalized_data=datanew;
normalized_data(:,1)=[];
F.final_accuracy=0;
% K fold crossvalidation
k = 56;
no_of_samples = 56;
index = crossvalind('Kfold',no_of_samples,k);
for Sigma=1:0.2:3
    for NumberofHiddenNeurons=1:1:20
        for iterations=1:10
            for no = 1:k
                
                %%%%%%%%%%% Load training dataset
                train_data=datanew(setdiff(1:56,find(index==no)),:);
                T=train_data(:,1)';
                P=train_data(:,2:size(train_data,2));
                P_old=P;
                for i=1:size(P,1)
                    for j=1:size(P,1)
                        temp_train(i,j)=exp(-((sum((P(i,:)-P(j,:)).*(P(i,:)-P(j,:))))/(2*(Sigma*Sigma))));
                    end
                end
                P=temp_train';
                %clear train_data;                                   %   Release raw training data array
                
                
                %%%%%%%%%%% Load testing dataset
                test_data=datanew(index==no,:);
                TV.T=test_data(:,1)';
                TV.P=test_data(:,2:size(test_data,2));
                for i=1:size(TV.P,1)
                    for j=1:size(P,1)
                        temp_test(i,j)=exp(-((sum((TV.P(i,:)-P_old(j,:)).*(TV.P(i,:)-P_old(j,:))))/(2*(Sigma*Sigma))));
                    end
                end
                TV.P=temp_test';
                %clear test_data;                                    %   Release raw testing data array
                
                
                NumberofTrainingData=size(P,2);
                NumberofTestingData=size(TV.P,2);
                NumberofInputNeurons=size(P,1);
                
                if Elm_Type~=REGRESSION
                    %%%%%%%%%%%% Preprocessing the data of classification
                    sorted_target=sort(cat(2,T,TV.T),2);
                    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
                    label(1,1)=sorted_target(1,1);
                    j=1;
                    for i = 2:(NumberofTrainingData+NumberofTestingData)
                        if sorted_target(1,i) ~= label(1,j)
                            j=j+1;
                            label(1,j) = sorted_target(1,i);
                        end
                    end
                    number_class=j;
                    NumberofOutputNeurons=number_class;
                    
                    %%%%%%%%%% Processing the targets of training
                    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
                    for i = 1:NumberofTrainingData
                        for j = 1:number_class
                            if label(1,j) == T(1,i)
                                break;
                            end
                        end
                        temp_T(j,i)=1;
                    end
                    T=temp_T*2-1;
                    
                    %%%%%%%%%% Processing the targets of testing
                    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
                    for i = 1:NumberofTestingData
                        for j = 1:number_class
                            if label(1,j) == TV.T(1,i)
                                break;
                            end
                        end
                        temp_TV_T(j,i)=1;
                    end
                    TV.T=temp_TV_T*2-1;
                    
                end                                                 %   end if of Elm_Type
                
                %%%%%%%%%%% Calculate weights & biases
                start_time_train=cputime;
                
                %%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
                InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
                BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
                tempH=InputWeight*P;
                %clear P;                                            %   Release input of training data
                ind=ones(1,NumberofTrainingData);
                BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
                tempH=tempH+BiasMatrix;
                
                %%%%%%%%%%% Calculate hidden neuron output matrix H
                switch lower(ActivationFunction)
                    case {'sig','sigmoid'}
                        %%%%%%%% Sigmoid
                        H = 1 ./ (1 + exp(-tempH));
                    case {'sin','sine'}
                        %%%%%%%% Sine
                        H = sin(tempH);
                    case {'hardlim'}
                        %%%%%%%% Hard Limit
                        H = double(hardlim(tempH));
                    case {'tribas'}
                        %%%%%%%% Triangular basis function
                        H = tribas(tempH);
                    case {'radbas'}
                        %%%%%%%% Radial basis function
                        H = radbas(tempH);
                        %%%%%%%% More activation functions can be added here
                end
                clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H
                
                %%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
                OutputWeight=pinv(H') * T';                        % implementation without regularization factor //refer to 2006 Neurocomputing paper
                %OutputWeight=inv(eye(size(H,1))/C+H * H') * H * T';   % faster method 1 //refer to 2012 IEEE TSMC-B paper
                %implementation; one can set regularizaiton factor C properly in classification applications
                %OutputWeight=(eye(size(H,1))/C+H * H') \ H * T';      % faster method 2 //refer to 2012 IEEE TSMC-B paper
                %implementation; one can set regularizaiton factor C properly in classification applications
                
                %If you use faster methods or kernel method, PLEASE CITE in your paper properly:
                
                %Guang-Bin Huang, Hongming Zhou, Xiaojian Ding, and Rui Zhang, "Extreme Learning Machine for Regression and Multi-Class Classification," submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence, October 2010.
                
                end_time_train=cputime;
                TrainingTime=end_time_train-start_time_train;        %   Calculate CPU time (seconds) spent for training ELM
                
                %%%%%%%%%%% Calculate the training accuracy
                Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
                if Elm_Type == REGRESSION
                    TrainingAccuracy=sqrt(mse(T - Y));               %   Calculate training accuracy (RMSE) for regression case
                end
                clear H;
                
                %%%%%%%%%%% Calculate the output of testing input
                start_time_test=cputime;
                tempH_test=InputWeight*TV.P;
                clear TV.P;             %   Release input of testing data
                ind=ones(1,NumberofTestingData);
                BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
                tempH_test=tempH_test + BiasMatrix;
                switch lower(ActivationFunction)
                    case {'sig','sigmoid'}
                        %%%%%%%% Sigmoid
                        H_test = 1 ./ (1 + exp(-tempH_test));
                    case {'sin','sine'}
                        %%%%%%%% Sine
                        H_test = sin(tempH_test);
                    case {'hardlim'}
                        %%%%%%%% Hard Limit
                        H_test = hardlim(tempH_test);
                    case {'tribas'}
                        %%%%%%%% Triangular basis function
                        H_test = tribas(tempH_test);
                    case {'radbas'}
                        %%%%%%%% Radial basis function
                        H_test = radbas(tempH_test);
                        %%%%%%%% More activation functions can be added here
                end
                TY=(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data
                end_time_test=cputime;
                TestingTime=end_time_test-start_time_test;           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data
                
                if Elm_Type == REGRESSION
                    TestingAccuracy=sqrt(mse(TV.T - TY));            %   Calculate testing accuracy (RMSE) for regression case
                end
                
                if Elm_Type == CLASSIFIER
                    %%%%%%%%%% Calculate training & testing classification accuracy
                    MissClassificationRate_Training=0;
                    MissClassificationRate_Testing=0;
                    
                    for i = 1 : size(T, 2)
                        [x, label_index_expected]=max(T(:,i));
                        [x, label_index_actual]=max(Y(:,i));
                        if label_index_actual~=label_index_expected
                            MissClassificationRate_Training=MissClassificationRate_Training+1;
                        end
                    end
                    TrainingAccuracy=1-MissClassificationRate_Training/size(T,2);
                    for i = 1 : size(TV.T, 2)
                        [x, label_index_expected]=max(TV.T(:,i));
                        [x, label_index_actual]=max(TY(:,i));
                        if label_index_actual~=label_index_expected
                            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
                        end
                    end
                    TestingAccuracy(no)=1-MissClassificationRate_Testing/size(TV.T,2);
                end
            end
            PreFinalAccuracy=sum(TestingAccuracy)/k;
            if PreFinalAccuracy>F.final_accuracy
                F.final_accuracy=PreFinalAccuracy;
                F.final_InputWeight=InputWeight;
                F.final_BiasofHiddenNeurons=BiasMatrix;
                F.final_OutputWeight=OutputWeight;
                F.final_Sigma=Sigma;
                F.final_NumberofHiddenNeurons=NumberofHiddenNeurons;
                F.final_rbf_centres=P_old;
            end
        end
        fprintf('Number of Hidden Neurons=%i\n',NumberofHiddenNeurons);
    end
    fprintf('Sigma value=%f\n',Sigma);
end
csvwrite('final_InputWeight_k56_rbf_balanced_arx5_vary_sigma_1-3_hidden_1-20.txt', F.final_InputWeight);
csvwrite('final_BiasofHiddenNeurons_k56_rbf_balanced_arx5_vary_sigma_1-3_hidden_1-20.txt', F.final_BiasofHiddenNeurons);
csvwrite('final_OutputWeight_k56_rbf_balanced_arx5_vary_sigma_1-3_hidden_1-20.txt', F.final_OutputWeight);
csvwrite('normalized_data_k56_rbf_balanced_arx5_vary_sigma_1-3_hidden_1-20.txt', normalized_data);
csvwrite('final_Sigma_k56_rbf_balanced_arx5_vary_sigma_1-3_hidden_1-20.txt',F.final_Sigma);
csvwrite('final_NumberofHiddenNeurons_k56_rbf_balanced_arx5_vary_sigma_1-3_hidden_1-20.txt',F.final_NumberofHiddenNeurons);
csvwrite('final_rbf_centres_k56_rbf_balanced_arx5_vary_sigma_1-3_hidden_1-20.txt',F.final_rbf_centres);