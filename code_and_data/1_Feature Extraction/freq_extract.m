function output = freq_extract(i, sample_type)

    str1 = '../';
    str2 = '.edf';
    [header, sig] = edfread(strcat(str1, sample_type, num2str(i), str2));   %Reading the EDF file
    Fs=1000; %Sampling Rate
    t_start=1;
    t_stop=length(sig);
    y=sig(t_start:t_stop);
    time=(0:length(sig)-1)/Fs;
    t=(1:length(y));
    figure
    subplot(4,1,1)
    plot(time(t_start:t_stop),y);   %Plotting the original signal
    grid on
    title('ORIGINAL SIGNAL')
    xlabel('TIME')
    ylabel('AMPLITUDE')

    %%Performing WAVELET TRANSFORMATION%%
    [C,L] = wavedec(y,8,'db5');
    A8=wrcoef('a',C,L,'db5',8);

    subplot(4,1,2)
    plot(time(t_start:t_stop),A8) 
    grid on
    title('LEVEL 8 APPROXIMATION OF THE SIGNAL')
    xlabel('TIME')
    ylabel('AMPLITUDE')

    subplot(4,1,3)
    y1=y-(A8);
    plot(time(t_start:t_stop),y1)
    grid on
    title('REQUIRED GRAPH')
    xlabel('TIME')
    ylabel('AMPLITUDE')

    %%DETECTING THE PEAKS%%
    subplot(4,1,4)
    [pks,time1]=findpeaks(y1,'MinPeakHeight',1.0); %find_peaks(y1, t_start, t_stop);
    plot(time(t_start:t_stop),y1,time(time1+t_start),pks,'or')
    grid on 
    title('RR PEAK DETECTED')
    xlabel('TIME')
    ylabel('AMPLITUDE')

    %%PLOTTING THE PSD%%
    figure
    [pxx1,f]=pwelch(y1,Fs);
    plot(f,pxx1)
    xlabel('FREQUENCY')
    ylabel('POWER')
    title('POWER SPECTRAL DENSITY')

    %%CALCULATING THE FEATURES%%
    ind0_1=find(f<0.040);                           %Finding the index of all VLF values
    X=(f(1:ind0_1(end)));                           %Extracting all the frequency values in the VLF range
    val=pxx1(1:length(X));                          %Finding all the power values in the VLF range
    abs_pw_vlf=max(val);                            %Finding the Peak value
    ind=find(pxx1==abs_pw_vlf);                     %Finding the peak value index
    pk_freq_vlf=f(ind);                             %Find Peak Frequency for VLF
    %a=0:0.0016:X(end);
    ab_pow_vlf=trapz(X,val);                        %Absolute-power in the VLF range
    %plot(X,val);

    ind1_1=find(0.15>f & f>0.04);                   %Finding the index of all LF values
    X1=(f(ind0_1(end):ind1_1(end)));                %Extracting all the frequency values in the LF range
    val1=pxx1(ind0_1(end):ind0_1(end)+length(X1));  %Finding all the power values in the LF range
    abs_pw_lf=max(val1);                            %Finding the Peak value
    ind1=find(pxx1==abs_pw_lf);                     %Finding the peak value index
    pk_freq_lf=f(ind1);                             %Find Peak Frequency for LF
    ab_pow_lf=trapz(val1);                          %Absolute-power in the LF range

    ind2_1=find(0.40>f & f>0.15);                   %Finding the index of all HF values
    X2=(f(ind1_1(end):ind2_1(end)));                %Extracting all the frequency values in the HF range
    val2=pxx1(ind1_1(end):ind1_1(end)+length(X2));  %Finding all the power values in the HF range
    abs_pw_hf=max(val2);                            %Finding the Peak value
    ind2=find(pxx1==abs_pw_hf);                     %Finding the peak value index
    pk_freq_hf=f(ind2);                             %Find Peak Frequency for HF
    ab_pow_hf=trapz(val2);                          %Absolute-power in the HF range

    pw_ttl=ab_pow_vlf+ab_pow_lf+ab_pow_hf;          %Total power
    rp_vlf=ab_pow_vlf/pw_ttl;                       %Relative Power in each range
    rp_lf=ab_pow_lf/pw_ttl;
    rp_hf=ab_pow_hf/pw_ttl;

    norm_lf=ab_pow_lf/(pw_ttl-ab_pow_vlf);          %Normalized power in LF and HF
    norm_hf=ab_pow_hf/(pw_ttl-ab_pow_vlf);

    ratio=ab_pow_lf/ab_pow_hf;                      %Ratio of band powers in LF and HF
    freq_fea=[pk_freq_vlf, pk_freq_lf, pk_freq_hf, ab_pow_vlf, ab_pow_lf, ab_pow_hf, pw_ttl, rp_vlf, rp_lf, rp_hf, norm_lf, norm_hf, ratio];
    
    output = freq_fea;
end    