df<-read.csv(file.choose(),header = TRUE)
str(df)
df$Loan_ID<-NULL
summary(df)
#clean missing values from gender
which(df$Gender=="")
summary(df$Gender)
df$Gender[df$Gender==""]="Male"
df$Gender<-factor(droplevels(df$Gender))
df$Gender<-factor(df$Gender,levels = c("Male","Female"))

#clean missing values from married
summary(df$Married)
df$Married[df$Married==""]="Yes"
df$Married<-factor(droplevels(df$Married))
df$Married<-factor(df$Married,levels = c("Yes","No"))

#clean missing values from dependents
summary(df$Dependents)
df$Dependents[df$Dependents==""]=0
df$Dependents<-factor(droplevels(df$Dependents))

#clean missing values from self_employed
summary(df$Self_Employed)
df$Self_Employed[df$Self_Employed==""]="No"
df$Self_Employed<-factor(droplevels(df$Self_Employed))
df$Self_Employed<-factor(df$Self_Employed,levels = c("Yes","No"))

summary(df)
# clean missing values from loanamount
summary(df$LoanAmount)
hist(df$LoanAmount)
df$LoanAmount<-ifelse(is.na(df$LoanAmount),mean(df$LoanAmount,na=T),df$LoanAmount)

#clean missing values from loan_amount_term
summary(df$Loan_Amount_Term)
df$Loan_Amount_Term<-ifelse(is.na(df$Loan_Amount_Term),mean(df$Loan_Amount_Term,na=T),
                            df$Loan_Amount_Term)

#clean missing values from credit_histroy
df$Credit_History<-ifelse(is.na(df$Credit_History),median(df$Credit_History,na=T),
                          df$Credit_History)

df$Loan_Status<-factor(df$Loan_Status,levels = c("Y","N"))

df$Credit_History<-as.factor(df$Credit_History)
str(df)
summary(df)
###############################
#outliers from ApplicantIncome
box1<-boxplot(df$ApplicantIncome,main="applicantincome")
summary(box1$out)
outlier1<-box1$out
hist(outlier1)
df$ApplicantIncome<-ifelse(df$ApplicantIncome>10408,mean(outlier1),df$ApplicantIncome)
summary(df$ApplicantIncome)

#outliers from CoapplicantIncome
box2<-boxplot(df$CoapplicantIncome,main="coapplcant")
summary(box2$out)
outlier2<-box2$out
hist(outlier2)
df$CoapplicantIncome<-ifelse(df$CoapplicantIncome>=6250,median(outlier2),df$CoapplicantIncome)
summary(df$CoapplicantIncome)

#outliers from loanamount
box3<-boxplot(df$LoanAmount,main="loan")
summary(box3$out)
outlier3<-box3$out
outlier3
sum(outlier3>370)
df$LoanAmount<-ifelse(df$LoanAmount>=265,360.0,df$LoanAmount)
summary(df$LoanAmount)
summary(df)

library(ggplot2)
ggplot(data = df,aes(x=LoanAmount,fill=Loan_Status))+geom_bar()
ggplot(data = df,aes(x=ApplicantIncome,fill=Loan_Status))+geom_bar(position = "fill")

library(psych)
describe(df)

#crosstab
library(gmodels)
CrossTable(df$Gender,df$Loan_Status,prop.r =F,prop.chisq = F,
           prop.c= F,prop.t = T,chisq = T,
           format = "SPSS",digits = 2)
?CrossTable
library(caret)
cont<-trainControl(method = "repeatedcv",
                   number = 5,
                   repeats = 5,
                   classProbs = TRUE,
                   savePredictions = TRUE)
summary(cont)
cont2<-trainControl(classProbs = TRUE,
                    savePredictions = TRUE)
summary(cont2)

#model building
#cross-validation svm
svmmodel1<-train(Loan_Status~.,data=df,method="svmLinear",trControl=cont,
                 metric="ROC")
svmmodel1

#BASE SVM
svmmodel2<-train(Loan_Status~.,data = df,
                 method="svmLinear",
                 trControl=cont2,metric="ROC")
svmmodel2

#cross-validated SVM: non-linear
svmRcv<-train(Loan_Status~.,data = df,method="svmRadial",
              trControl=cont,metric="ROC")
svmRcv

svmR<-train(Loan_Status~.,data = df,method="svmRadial",
            trControl=cont2,metric="ROC")
svmR

#importing test data for predication futher analysis
test<-read.csv(file.choose(),header = TRUE)
str(test)
test$Loan_ID<-NULL
summary(test)

#missing value from gender
which(test$Gender=="")
test$Gender[test$Gender==""]<-"Male"
test$Gender<-factor(droplevels(test$Gender))
test$Gender<-factor(test$Gender,levels = c("Male","Female"))

test$Married<-factor(test$Married,levels = c("Yes","No"))

#missing value from dependents
which(test$Dependents=="")
test$Dependents[test$Dependents==""]<-0
test$Dependents<-factor(droplevels(test$Dependents))

#missing value from self_Employed
which(test$Self_Employed=="")
test$Self_Employed[test$Self_Employed==""]<-"No"
test$Self_Employed<-factor(droplevels(test$Self_Employed))
test$Self_Employed<-factor(test$Self_Employed,levels = c("Yes","No"))
summary(test)

#missing value from credit_history
str(test$Credit_History)
which(test$Credit_History==1)
test$Credit_History<-ifelse(is.na(test$Credit_History),1,test$Credit_History)
test$Credit_History<-as.factor(test$Credit_History)

#outliers from applocantincome
boxT1<-boxplot(test$ApplicantIncome)
outlierT1<-boxT1$out
summary(outlierT1)
test$ApplicantIncome<-ifelse(test$ApplicantIncome>=8449,median(outlierT1),test$ApplicantIncome)
boxT1$out

#outliers from coapplicantincome
boxT2<-boxplot(test$CoapplicantIncome)
outlierT2<-boxT2$out
summary(outlierT2)
test$CoapplicantIncome<-ifelse(test$CoapplicantIncome>=6414,mean(outlierT2),test$CoapplicantIncome)
boxplot(test$CoapplicantIncome)

#outliers from loanamount
summary(test$LoanAmount)
sum(is.na(test$LoanAmount))
test$LoanAmount<-ifelse(is.na(test$LoanAmount),median(test$LoanAmount,na=T),test$LoanAmount)
summary(test)
boxT3<-boxplot(test$LoanAmount)
outliersT3<-boxT3$out
summary(outliersT3)
test$LoanAmount<-ifelse(test$LoanAmount>=254,310,test$LoanAmount)
boxplot(test$LoanAmount)

#outliers from loan_amount_term
summary(test$Loan_Amount_Term)
median(test$Loan_Amount_Term,na=T)
test$Loan_Amount_Term<-ifelse(is.na(test$Loan_Amount_Term),median(test$Loan_Amount_Term,na=T),
                              test$Loan_Amount_Term)
summary(test)

#predication for confusion matrix
svmpredmodel1<-predict(svmmodel1,newdata = test)
svmpredmodel2<-predict(svmmodel2,newdata = test)
SVMPREDR1<-predict(svmRcv,newdata = test)
SVMPREDR2<-predict(svmR,newdata = test)

#PREDICATION FOR TEST DATA
test$preprob<-predict(svmmodel1,test,type = "prob")
test$prob<-ifelse(test$preprob>0.5,"Y","N")
str(test)

Lcvprob<-predict(svmmodel1,newdata=test,type="prob")
Lcvprob
library(pROC)

#####################
#roc 
plot.roc(test$preprob$Y,test$prob,print.auc=T,auc.pplygon=T,main="glm model")



###########################
library(dlookr)
describe(df)
plot_normality(df)
###########################

