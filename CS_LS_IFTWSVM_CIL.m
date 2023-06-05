
function [accuracy] = CS_LS_IFTWSVM_CIL(TestX,Data,FunPara)
DataTrain.A=Data(Data(:,end)==1,1:end-1);
DataTrain.B=Data(Data(:,end)~=1,1:end-1);

FunPara.mu=FunPara.kerfPara.pars;
[S1,S2]=non_linear_score_values(Data,FunPara.mu);
S1=diag(S1);
S2=diag(S2);

Xpos = DataTrain.A;
Xneg = DataTrain.B;
m1=size(Xpos,1);
m2=size(Xneg,1);

D1=m2/(m1+m2);
D2=m1/(m1+m2);

cpos = FunPara.c1;
cpos=cpos*D1;
cneg = FunPara.c1;
cneg = cneg*D2;

eps1 = FunPara.c3;
eps1=eps1*D1;
eps2 = FunPara.c3;
eps2=eps2*D2;

if strcmp(FunPara.kerfPara.type,'rbf')
    FunPara.kerfPara.pars=FunPara.mu;
else
    FunPara.kerfPara.pars=1;
end

kerfPara = FunPara.kerfPara;

e1=-ones(m1,1);
e2=-ones(m2,1);

if strcmp(kerfPara.type,'lin')
    H1=[Xpos,-e1];
    G1=[Xneg,-e2];
else
    X=[DataTrain.A;DataTrain.B];
    H1=[kernelfun(Xpos,kerfPara,X),-e1];
    G1=[kernelfun(Xneg,kerfPara,X),-e2];
end

H=H1;G=S2*G1;
HH=H'*H;
GtG=G'*G;
HH = HH + eps1*eye(size(HH))+cpos*(GtG);
vpos = HH\(cpos*(G'*S2*e2));
G=G1;H=S1*H1;
QQ=G'*G;
HtH=H'*H;
QQ=QQ + eps2*eye(size(QQ))+cneg*(HtH);
vneg=QQ\(cneg*(H'*S1*e1));
vneg=-vneg;
w1=vpos(1:(length(vpos)-1));
b1=vpos(length(vpos));
w2=vneg(1:(length(vneg)-1));
b2=vneg(length(vneg));


xtest0=TestX(:,1:end-1);
no_test=size(xtest0,1);
K = zeros(no_test,X);
for i =1: no_test
    for j =1: m3
        nom = norm( xtest0(i ,:) - X(j ,:) );
        K(i,j )=exp(-mew1*nom*nom);
    end
end
K=[K ones(no_test,1)];
preY1=K*w1/norm(w1(1:size(w1,1)-1,:));preY2=K*w2/norm(w2(1:size(w2,1)-1,:));
predicted_class=[];
for i=1:no_test
    if abs(preY1(i))< abs(preY2(i))
        predicted_class=[predicted_class;1];
    else
        predicted_class=[predicted_class;-1];
    end

end

%%%%%%% accuracy
no_test=m_test;
classifier=predicted_class;
obs1=TestX(:,end);
match = 0.;
match1=0;

posval=0;
negval=0;

for i = 1:no_test
    if(obs1(i)==1)
        if(classifier(i) == obs1(i))
            match = match+1;
        end
        posval=posval+1;
    elseif(obs1(i)==-1)
        if(classifier(i) ~= obs1(i))
            match1 = match1+1;
        end
        negval=negval+1;
    end
end
if(posval~=0)
    a_pos=(match/posval)
else
    a_pos=0;
end

if(negval~=0)
    am_neg=(match1/negval)
else
    am_neg=0;
end

AUC=(1+a_pos-am_neg)/2;

accuracy=AUC*100
end