function [S1,S2,alpha_d,mew]=non_linear_score_values(A,mew)
[no_input,no_col]=size(A);
A1=A(A(:,end)==1,1:end-1);
B1=A(A(:,end)~=1,1:end-1);
K1 = exp(-(1/(mew^2))*(repmat(sqrt(sum(A1.^2,2).^2),1,size(A1,1))-2*(A1*A1')+repmat(sqrt(sum(A1.^2,2)'.^2),size(A1,1),1)));
K2 = exp(-(1/(mew^2))*(repmat(sqrt(sum(B1.^2,2).^2),1,size(B1,1))-2*(B1*B1')+repmat(sqrt(sum(B1.^2,2)'.^2),size(B1,1),1)));
A_temp=A(:,1:end-1);
K3 = exp(-(1/(mew^2))*(repmat(sqrt(sum(A_temp.^2,2).^2),1,size(A_temp,1))-2*(A_temp*A_temp')+repmat(sqrt(sum(A_temp.^2,2)'.^2),size(A_temp,1),1)));

radiusxp=sqrt(1-2*mean(K1,2)+mean(mean(K1)));
radiusmaxxp=max(radiusxp);
radiusxn=sqrt(1-2*mean(K2,2)+mean(mean(K2)));
radiusmaxxn=max(radiusxn);
alpha_d=max(radiusmaxxn,radiusmaxxp);
mem1=ones(size(radiusxp,1),1)-(radiusxp/(radiusmaxxp+10^-4));
mem2=ones(size(radiusxn,1),1)-(radiusxn/(radiusmaxxn+10^-4));
ro=[];
DD=sqrt(2*(ones(size(K3))-K3));
for i=1:no_input
    temp=DD(i,:)';
    B1=A(temp<alpha_d,:);
    
    [x3,~]=size(B1);
    count=sum(A(i,end)*ones(size(B1,1),1)~=B1(:,end));
    
    x5=count/x3;
    ro=[ro;x5];
end

A2=[A(:,no_col) ro];
ro2=A2(A2(:,1)==-1,2);
ro1=A2(A2(:,1)~=-1,2);
v1=(ones(size(mem1))-mem1).*ro1;
v2=(ones(size(mem2))-mem2).*ro2;

S1=[];
S2=[];
for i=1:size(v1,1)
    if v1(i)==0
        S1=[S1;mem1(i)];
    elseif (mem1(i)<=v1(i))
        S1=[S1;0];
    else
        S1=[S1;(1-v1(i))/(2-mem1(i)-v1(i))];
    end
end

for i=1:size(v2,1)
    if v2(i)==0
        S2=[S2;mem2(i)];
    elseif (mem2(i)<=v2(i))
        S2=[S2;0];
    else
        S2=[S2;(1-v2(i))/(2-mem2(i)-v2(i))];
    end
end

end