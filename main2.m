clc; clear all; close all;

for load_file = 1
    %% initializing variables
    no_part = 5.;
    %% to load file
    switch load_file
        case 1
            file = 'ecoli-0-1_vs_2-3-5';
            test_start =121;
            FunPara.c1=10;
            FunPara.c3=10;
            FunPara.kerfPara.pars=1;
        otherwise
            continue;
    end

    %% Change the file path and dataset here
    filename = strcat('./newd/',file,'.mat');
    A = load(filename);
    [m,n] = size(A);

    for i=1:m
        if A(i,n)==0
            A(i,n)=-1;
        end
    end
    test = A(test_start:m,:);
    train = A(1:test_start-1,:);


    [no_input,no_col] = size(train);

    x1 = train(:,1:no_col-1);
    y1 = train(:,no_col);

    [no_test,no_col] = size(test);
    xtest0 = test(:,1:no_col-1);
    ytest0 = test(:,no_col);

    %Combining all the column in one variable
    A=[x1 y1];    %training data
    [m,n] = size(A);
    A_test=[xtest0,ytest0];    %testing data
    p=0;
    for i=1:m
        if A(i,n)==1
            p=p+1;
        end
    end
    ir=(m-p)/(p);
    FunPara.ir=ir;
    FunPara.kerfPara.type = 'rbf';
    %% initializing crossvalidation variables

    [lengthA,n] = size(A);
    min_err = -10^-10.;


    ind = -5:5;
    cvs1= 10.^(ind);
    eps=cvs1;
    mus=10.^(ind);
    cvs0=[0.5,1,1.5,2,2.5];

    for C1 = 1:length(cvs1)
        FunPara.c1 = cvs1(C1)
        for C3 = 1:length(cvs1)
            FunPara.c3 = cvs1(C3)
            for mv = 1:length(mus)
                FunPara.kerfPara.pars = mus(mv)

                avgerror = 0;
                block_size = lengthA/(no_part*1.0);
                part = 0;
                t_1 = 0;
                t_2 = 0;
                while ceil((part+1) * block_size) <= lengthA
                    %% seprating testing and training datapoints for
                    % crossvalidation
                    t_1 = ceil(part*block_size);
                    t_2 = ceil((part+1)*block_size);
                    B_t = [A(t_1+1 :t_2,:)];
                    Data = [A(1:t_1,:); A(t_2+1:lengthA,:)];
                    %% testing and training
                    [time,predicted_class,output_struct] = CS_LS_IFTWSVM_CIL(B_t,Data,FunPara);
                    [accuracy_with_zero,time] = rflstsvm(Data,B_t,c,c0,mu,ir);
                    avgerror = avgerror + accuracy_with_zero;
                    part = part+1
                end

                if avgerror > min_err
                    min_err = avgerror;
                    OptPara=FunPara;

                end
                %
            end
        end
    end


    [accuracy,time] = CS_LS_IFTWSVM_CIL(A_test,A,OptPara)

end