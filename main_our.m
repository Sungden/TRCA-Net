
% Training and testing of the proposed deep neural network (DNN) in [1]
% for steady-state visual evoked potentials SSVEP-based BCI

% [1] O. B. Guney, M. Oblokulov and H. Ozkan, "A Deep Neural Network for SSVEP-Based
% Brain-Computer Interfaces," IEEE Transactions on Biomedical Engineering, vol. 69,
% no. 2, pp. 932-944, 2022.

%% Preliminaries
% Please download benchmark [2] and/or BETA [3] datasets
% and add folder that contains downloaded files to the MATLAB path.

% [2] Y. Wang, X. Chen, X. Gao, and S. Gao, “A benchmark dataset for
% ssvep-based brain–computer interfaces,” IEEE Transactions on Neural Systems and
% Rehabilitation Engineering,vol. 25, no. 10, pp. 1746–1752, 2016.

% [3] B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, “Beta: A large
% benchmark database toward ssvep-bci application,” Frontiers in
% Neuroscience, vol. 14, p. 627, 2020.

% [4] Nakanishi, Masaki, et al.
% "Enhancing detection of SSVEPs for a high-speed brain speller using task-related component analysis."
% IEEE Transactions on Biomedical Engineering 65.1 (2017): 104-112.
%% Specifications (e.g. number of character) of datasets
is_ensemble=1; %using TRCA or eTRCA
dataset='data_our'; % 'Bench' or 'BETA' dataset
signal_length=0.4; % Signal length in second
result='result_our/';

two_stages=1;%using two_stages(transfer learning) approach or not
if two_stages==1
    use_transfer='Transfer/';
else
    use_transfer='No_transfer/';
end

f_dir=fullfile(result,dataset,num2str(signal_length),use_transfer);
if ~exist(f_dir,'dir')
    mkdir(f_dir)
end
if strcmp(dataset,'Bench')
    is_dataset=1;
    subban_no=3;
    totalsubject=35; % # of subjects
    totalblock=6; % # of blocks
    totalcharacter=40; % # of characters
    sampling_rate=250; % Sampling rate
    visual_latency=0.14; % Average visual latency of subjects
    visual_cue=0.5; % Length of visual cue used at collection of the dataset
    sample_length=sampling_rate*signal_length; % Sample length
    total_ch=64; % # of channels used at collection of the dataset
    max_epochs=500; % # of epochs for first stage
    max_epochs_first_stage=1000;
    max_epochs_second_stage=2000;
    dropout_first_stage=0.1; % Dropout probabilities of first two dropout layers at first stage
    dropout_second_stage=0.6; % Dropout probabilities of first two dropout layers at second stage
    dropout_final=0.95;
    dropout_M1=0.7;
    all_conf_matrix=zeros(40,40);
    %High cut off frequencies for the bandpass filters (90 Hz for all)
    high_cutoff = ones(1,subban_no)*90;
    %Low cut off frequencies for the bandpass filters (ith bandpass filter low cutoff frequency 8*i)
    low_cutoff =8:8:8*subban_no;

 elseif strcmp(dataset,'Bench_max')
    is_dataset=1;
    subban_no=3;
    totalsubject=10; % # of subjects
    totalblock=6; % # of blocks
    totalcharacter=40; % # of characters
    sampling_rate=250; % Sampling rate
    visual_latency=0.14; % Average visual latency of subjects
    visual_cue=0.5; % Length of visual cue used at collection of the dataset
    sample_length=sampling_rate*signal_length; % Sample length
    total_ch=64; % # of channels used at collection of the dataset
    max_epochs=500; % # of epochs for first stage
    max_epochs_first_stage=1000;
    max_epochs_second_stage=2000;
    dropout_first_stage=0.1; % Dropout probabilities of first two dropout layers at first stage
    dropout_second_stage=0.6; % Dropout probabilities of first two dropout layers at second stage
    dropout_final=0.95;
    dropout_M1=0.7;
    all_conf_matrix=zeros(40,40);
    %High cut off frequencies for the bandpass filters (90 Hz for all)
    high_cutoff = ones(1,subban_no)*90;
    %Low cut off frequencies for the bandpass filters (ith bandpass filter low cutoff frequency 8*i)
    low_cutoff =8:8:8*subban_no;   
    
elseif strcmp(dataset,'data_our')
    is_dataset=1;
    subban_no=3;
    totalsubject=10; % # of subjects
    totalblock=4; % # of blocks
    totalcharacter=40; % # of characters
    sampling_rate=250; % Sampling rate
    visual_latency=0.14; % Average visual latency of subjects
    visual_cue=0; % Length of visual cue used at collection of the dataset
    sample_length=sampling_rate*signal_length; % Sample length
    total_ch=9; % # of channels used at collection of the dataset
    max_epochs=500; % # of epochs for first stage
    max_epochs_first_stage=1000;
    max_epochs_second_stage=2000;
    dropout_first_stage=0.1; % Dropout probabilities of first two dropout layers at first stage
    dropout_second_stage=0.6; % Dropout probabilities of first two dropout layers at second stage
    dropout_final=0.95;
    dropout_M1=0.7;
    all_conf_matrix=zeros(40,40);
    %High cut off frequencies for the bandpass filters (90 Hz for all)
    high_cutoff = ones(1,subban_no)*90;
    %Low cut off frequencies for the bandpass filters (ith bandpass filter low cutoff frequency 8*i)
    low_cutoff =8:8:8*subban_no;
    
elseif strcmp(dataset,'BETA')
    is_dataset=2;
    subban_no=3;
    totalsubject=70;
    totalblock=4;
    totalcharacter=40;
    sampling_rate=250;
    visual_latency=0.13;
    visual_cue=0.5;
    sample_length=sampling_rate*signal_length; %
    total_ch=64;
    max_epochs=800;
    max_epochs_first_stage=1000;
    max_epochs_second_stage=4000;
    dropout_first_stage=0.1; % Dropout probabilities of first two dropout layers at first stage
    dropout_second_stage=0.7;
    dropout_final=0.95;
    dropout_M1=0.7;
    all_conf_matrix=zeros(40,40);
    %High cut off frequencies for the bandpass filters (90 Hz for all)
    high_cutoff = ones(1,subban_no)*90;
    %Low cut off frequencies for the bandpass filters (ith bandpass filter low cutoff frequency 8*i)
    low_cutoff =8:8:8*subban_no;
elseif strcmp(dataset,'shanhuan')
    is_dataset=0;
    subban_no=1;
    totalsubject=30;
    totalblock=20;
    totalcharacter=8;
    sampling_rate=250;
    visual_latency=0.14;
    visual_cue=0;
    sample_length=sampling_rate*signal_length; %
    total_ch=9;
    max_epochs=1000;
    max_epochs_second_stage=1000;
    dropout_first_stage=0.1; % Dropout probabilities of first two dropout layers at first stage
    dropout_second_stage=0.5;
    dropout_final=0.7;
    dropout_final_second=0.8;
    dropout_M1=0.1;
    all_conf_matrix=zeros(8,8);
    %High cut off frequencies for the bandpass filters (90 Hz for all)
    high_cutoff = 65;
    %Low cut off frequencies for the bandpass filters (ith bandpass filter low cutoff frequency 8*i)
    low_cutoff =55;
    %else %if you want to use another dataset please specify parameters of the dataset
    % totalsubject= ... ,
    % totalblock= ... ,
    % ...
end

%% Preprocessing
total_delay=visual_latency+visual_cue; % Total undesired signal length in seconds
delay_sample_point=round(total_delay*sampling_rate); % # of data points correspond for undesired signal length
sample_interval = (delay_sample_point+1):delay_sample_point+sample_length; % Extract desired signal
channels=[48 54 55 56 57 58 61 62 63];% Indexes of 9 channels: (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, and O2)
total_channels=length(channels); % Determine total number of channel
% To use all the channels set channels to 1:total_ch=64;
[AllData,y_AllData]=PreProcess_1(channels,sample_length,sample_interval,subban_no,totalsubject,totalblock,totalcharacter,sampling_rate,dataset);
% Dimension of AllData:
% (# of channels, # sample length,  # of characters, # of blocks, # of subjects)
% (# of channels, # sample length, #subbands, # of characters, # of blocks, # of subjects)

% AllData=AllData(:,:,:,1:3,:);
% y_AllData=y_AllData(:,:,1:3,:);
% totalblock=3;


%% Forming bandpass filters
filter_order=2; % Filter Order of bandpass filters
PassBandRipple_val=1;
bpFilters=cell(subban_no,1); % Form and store bandpass filters
for f_b=1:subban_no
    bpFilt1 = designfilt('bandpassiir','FilterOrder',filter_order, ...
        'PassBandFrequency1',low_cutoff(f_b),'PassBandFrequency2',high_cutoff(f_b),...
        'PassBandRipple',PassBandRipple_val,...
        'DesignMethod','cheby1','SampleRate',sampling_rate);
    bpFilters{f_b}=bpFilt1;
end
%% Evaluations
acc_matrix=zeros(totalsubject,totalblock); % Initialization of accuracy matrix
acc_matrix_1=zeros(totalsubject,totalblock); % Initialization of accuracy matrix of the 1st stage
acc_matrix_2=zeros(totalsubject,totalblock); % Initialization of accuracy matrix of the 2nd stage
acc_trca=zeros(totalsubject,totalblock); % Initialization of accuracy matrix
sizes=size(AllData);%(# of channels, # sample length,  # of characters, # of blocks, # of subjects)
% Leave-one-block-out, each block is used as testing block in return,
% remaining blocks are used for training of the DNN.

for block=1:totalblock
    allblock=1:totalblock;
    allblock(block)=[]; % Exclude the block used for testing
    
    train=AllData(:,:,:,allblock,:); %Getting training data % (# of channels, # sample length,  # of characters, # of blocks, # of subjects)
    test=AllData(:, :, :, block,:);
    train_tmp=zeros(subban_no,sample_length,totalcharacter,totalcharacter,totalblock-1,totalsubject); % Initialization
    test_tmp=zeros(subban_no,sample_length,totalcharacter,totalcharacter,totalsubject); % Initialization
    for subject =1:totalsubject
        sub_data=train(:,:,:,:,subject);
        sub_data=permute(sub_data,[3 1 2 4]);% (# of characters,# of channels, # sample length, # of blocks)
        model = train_trca(sub_data, sampling_rate, subban_no,is_dataset);
        %size(model.W)%3    40     9
        testdata=squeeze(test(:,:,:,subject)); % (# of channels, # sample length,  # of characters)
        testdata = squeeze(permute(testdata,[3 1 2]));% (# of characters, # of channels, # sample length )
        estimated = test_trca(testdata, model, is_ensemble,is_dataset);
        is_correct = (estimated==[1:1:totalcharacter]);
        %accs(block) = mean(is_correct)*100;
        fprintf('Trial %d: Accuracy = %2.2f%%',...
            block, mean(is_correct)*100);
        acc_trca(subject,block)=mean(is_correct);
        sv_name=[f_dir,'/','acc_matrix_trca','.mat'];
        save(sv_name,'acc_trca');
         
        
        processed_signal=zeros(subban_no,sample_length,totalcharacter,totalcharacter,totalblock-1); % Initialization
        filtered_data=zeros(subban_no,total_channels,sample_length,1);
        for cla=1:totalcharacter
            for blo=1:size(sub_data,4)
                for w_n =1:size(model.W,2)
                    for f_b=1:subban_no
                        for cha=1:total_channels
                            filter_data=filtfilt(bpFilters{f_b},squeeze(sub_data(cla,cha,:,blo)));
                            filtered_data(f_b,cha,:,:)=filter_data;
                        end
                        processed_signal(f_b,:,w_n,cla,blo)=squeeze(filtered_data(f_b,:,:))'*squeeze(model.W(f_b,w_n, :));% Filtering raw signal with ith trca filter
                    end
                end
            end
        end
        train_tmp(:,:,:,:,:,subject)=processed_signal;          % (# of n_fb, # sample length, # of w_n, # of characters, # of blocks, # of subjects)%
        
        processed_signal_test=zeros(subban_no,sample_length,totalcharacter,totalcharacter); % Initialization
        filtered_data_test=zeros(subban_no,total_channels,sample_length,1);
        for cla=1:totalcharacter
            for w_n =1:size(model.W,2)
                for f_b=1:size(model.W,1)
                    for cha=1:total_channels
                        filter_data_test=filtfilt(bpFilters{f_b},squeeze(testdata(cla,cha,:)));
                        filtered_data_test(f_b,cha,:,:)=filter_data_test;
                    end
                    processed_signal_test(f_b,:,w_n,cla)=squeeze(filtered_data_test(f_b,:,:))'*squeeze(model.W(f_b,w_n, :));% Filtering raw signal with ith trca filter
                end
            end
        end
        test_tmp(:,:,:,:,subject)=processed_signal_test;          % (# of n_fb, # sample length, # of w_n, # of characters, # of subjects)%
    end
    train_tmp=permute(train_tmp,[3 2 1 4  5 6]);% ( # of w_n,# sample length, # of n_fb,  # of characters, # of blocks, # of subjects)
    sizes=size(train_tmp);
    train=reshape(train_tmp,[sizes(1),sizes(2),sizes(3),totalcharacter*length(allblock)*totalsubject*1]);% ( # of w_n,# sample length, # of n_fb,N)
    train_y=y_AllData(:,:,allblock,:);
    train_y=reshape(train_y,[1,totalcharacter*length(allblock)*totalsubject*1]);
    train_y=categorical(train_y);
    
    test_tmp=permute(test_tmp,[3 2 1 4 5]);% ( # of w_n,# sample length, # of n_fb,  # of characters,# of subjects)
    sizes=size(test_tmp);
    testdata=reshape(test_tmp,[sizes(1),sizes(2),sizes(3),totalcharacter*totalsubject]);
    test_y=y_AllData(:,:,block,:);
    test_y=reshape(test_y,[1,totalcharacter*totalsubject*1]);
    test_y=categorical(test_y);
    
    layers = [ ...
        imageInputLayer([sizes(1),sizes(2),subban_no],'Normalization','none')
        convolution2dLayer([1,1],1,'WeightsInitializer','ones') % If you use MATLAB R2018b or previous releases, you need to delete (,'WeightsInitializer','ones') and add (layers(2,1).Weights=ones(1,1,sizes(3))) to the line 81.
        convolution2dLayer([sizes(1),1],120,'WeightsInitializer','narrow-normal') % If you use MATLAB R2018b or previous releases you need to delete (,'WeightsInitializer','narrow-normal') from all convolution2dLayer and fullyConnectedLayer definitions.
        dropoutLayer(dropout_first_stage)
        convolution2dLayer([1,2],120,'Stride',[1,2],'WeightsInitializer','narrow-normal')
        dropoutLayer(dropout_first_stage)
        reluLayer
        convolution2dLayer([1,10],120,'Padding','Same','WeightsInitializer','narrow-normal')
        dropoutLayer(dropout_final)
        fullyConnectedLayer(totalcharacter,'WeightsInitializer','narrow-normal')
        softmaxLayer
        classificationLayer];
    
    layers(2, 1).BiasLearnRateFactor=0; % At first layer, sub-bands are combined with 1 cnn layer,
    % bias term basically adds DC to signal, hence there is no need to use
    % bias term at first layer. Note: Bias terms are initialized with zeros by default.
    % First stage training
    
    
    % First stage training
    options = trainingOptions('adam',... % Specify training options for first-stage training
        'InitialLearnRate',0.0001,...
        'MaxEpochs',max_epochs,...
        'MiniBatchSize',100, ...
        'Shuffle','every-epoch',...
        'L2Regularization',0.001,...
        'ExecutionEnvironment','gpu',...
        'Plots','training-progress');
    
    if two_stages==1
        main_net = trainNetwork(train,train_y,layers,options);
        model_name=[f_dir,'/','main_net_',int2str(block),'.mat'];
        save(model_name,'main_net'); % Save the trained model
    end
    
    %      model=load(model_name);%load the trained model
    %      main_net=model.main_net;
    
    % Second stage training
    for s=1:totalsubject
        layers = [ ...
            imageInputLayer([sizes(1),sizes(2),subban_no],'Normalization','none')
            convolution2dLayer([1,1],1)
            convolution2dLayer([sizes(1),1],120)
            dropoutLayer(dropout_second_stage)
            convolution2dLayer([1,2],120,'Stride',[1,2])
            dropoutLayer(dropout_second_stage)
            reluLayer
            convolution2dLayer([1,10],120,'Padding','Same')
            dropoutLayer(dropout_final)
            fullyConnectedLayer(totalcharacter)
            softmaxLayer
            classificationLayer];
        if two_stages==1
            % Transfer the weights that learnt in the first-stage training
            layers(2, 1).Weights = main_net.Layers(2, 1).Weights;
            layers(3, 1).Weights = main_net.Layers(3, 1).Weights;
            layers(5, 1).Weights = main_net.Layers(5, 1).Weights;
            layers(8, 1).Weights = main_net.Layers(8, 1).Weights;
            layers(10, 1).Weights = main_net.Layers(10, 1).Weights;
            
            layers(2, 1).BiasLearnRateFactor=0;
            layers(3, 1).Bias = main_net.Layers(3, 1).Bias;
            layers(5, 1).Bias = main_net.Layers(5, 1).Bias;
            layers(8, 1).Bias = main_net.Layers(8, 1).Bias;
            layers(10, 1).Bias = main_net.Layers(10, 1).Bias;
            
            options = trainingOptions('adam',... % Specify training options for second-stage training
                'InitialLearnRate',0.0001,...
                'MaxEpochs',max_epochs_first_stage,...
                'MiniBatchSize',totalcharacter*(totalblock-1), ...
                'Shuffle','every-epoch',...
                'L2Regularization',0.001,...
                'ExecutionEnvironment','gpu');
        else
            options = trainingOptions('adam',... % Specify training options for second-stage training
                'InitialLearnRate',0.0001,...
                'MaxEpochs',max_epochs_second_stage,...
                'MiniBatchSize',totalcharacter*(totalblock-1), ...
                'Shuffle','every-epoch',...
                'L2Regularization',0.001,...
                'ExecutionEnvironment','gpu');
        end
        
        % Getting the subject-specific data
        train_tmp_2=train_tmp(:,:,:,:,:,s);% ( # of w_n,# sample length, # of n_fb,  # of characters, # of blocks)
        sizes=size(train_tmp_2);
        train=reshape(train_tmp_2,[sizes(1),sizes(2),sizes(3),totalcharacter*length(allblock)*1]);% ( # of w_n,# sample length, # of n_fb,N)
        train_y=y_AllData(:,:,allblock,s);
        train_y=reshape(train_y,[1,totalcharacter*length(allblock)*1]);
        train_y=categorical(train_y);
        
        test_tmp_2=test_tmp(:,:,:,:,s);% ( # of w_n,# sample length, # of n_fb,  # of characters, # of blocks)
        sizes=size(test_tmp_2);
        testdata=reshape(test_tmp_2,[sizes(1),sizes(2),sizes(3),totalcharacter*1]);% ( # of w_n,# sample length, # of n_fb,N)
        test_y=y_AllData(:,:,block,s);
        test_y=reshape(test_y,[1,totalcharacter*1]);
        test_y=categorical(test_y);
        
        net = trainNetwork(train,train_y,layers,options);
        
        [YPred,~] = classify(net,testdata);
        acc=mean(YPred==test_y');
        acc_matrix(s,block)=acc;
        all_conf_matrix=all_conf_matrix+confusionmat(test_y,YPred);
        
        if two_stages==1
            [YPred_1,~] = classify(main_net,testdata);
            acc_1=mean(YPred_1==test_y');
            acc_matrix_1(s,block)=acc_1;
        end
    end
    if two_stages==1
        sv_name=[f_dir,'/','confusion_mat_',int2str(block),'.mat'];
        save(sv_name,'all_conf_matrix');
        sv_name=[f_dir,'/','acc_matrix','.mat'];
        save(sv_name,'acc_matrix');
        sv_name=[f_dir,'/','acc_matrix_1','.mat'];
        save(sv_name,'acc_matrix_1');
    else
        sv_name=[f_dir,'/','confusion_mat_no_transfer_',int2str(block),'.mat'];
        save(sv_name,'all_conf_matrix');
        sv_name=[f_dir,'/','acc_matrix_no_transfer','.mat'];
        save(sv_name,'acc_matrix');
    end
end
if two_stages==1
    itr_matrix=itr(acc_matrix,totalcharacter,0.5+signal_length);
    sv_name=[f_dir,'/','itr','.mat'];
    save(sv_name,'itr_matrix');
    
    itr_matrix_trca=itr(acc_trca,totalcharacter,0.5+signal_length);
    trca_sv_name=[f_dir,'/','trca_itr','.mat'];
    save(trca_sv_name,'itr_matrix_trca');
    
    itr_matrix_1=itr(acc_matrix_1,totalcharacter,0.5+signal_length);
    sv_name=[f_dir,'/','itr_1','.mat'];
    save(sv_name,'itr_matrix_1');
else
    itr_matrix=itr(acc_matrix,totalcharacter,0.5+signal_length);
    sv_name=[f_dir,'/','itr_no_transfer','.mat'];
    save(sv_name,'itr_matrix');
    
    itr_matrix_trca=itr(acc_trca,totalcharacter,0.5+signal_length);
    trca_sv_name=[f_dir,'/','trca_itr','.mat'];
    save(trca_sv_name,'itr_matrix_trca');
end
