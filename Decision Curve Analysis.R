library(readxl)
library(tidyverse)
library(rms)
library(foreign)

clinical <- read.table("D:/Desktop/stroke3_chen/R code/dataset/stroke2_clinic.csv",header = T,sep = ',')
str(clinical)
# clinical$mRS<-factor(clinical$mRS,levels = c("0","1"))
clinical$sex<-factor(clinical$sex,levels = c("1","2"))
clinical$Hyper<-factor(clinical$Hyper,levels = c("1","2"))
clinical$dia<-factor(clinical$dia,levels = c("1","2"))
clinical$CHD<-factor(clinical$CHD,levels = c("1","2"))
clinical$HCY<-factor(clinical$HCY,levels = c("1","2"))
clinical$AF<-factor(clinical$AF,levels = c("1","2"))
clinical$smoke<-factor(clinical$smoke,levels = c("1","2"))
clinical$drink<-factor(clinical$drink,levels = c("1","2"))
clinical$TOAST<-factor(clinical$TOAST,levels = c("1","2","3","4","5"))
clinical$OCSP<-factor(clinical$OCSP,levels = c("1","2","3","4"))
str(clinical)

pyrad <- read.table("D:/Desktop/stroke3_chen/R code/dataset/pyrad_lasso_5value_redscore.csv",header = T,sep = ',')
str(pyrad)

mydata<- read.table("D:/Desktop/stroke3_chen/R code/dataset/stroke2_joint.csv",header = T,sep = ',')
str(mydata)
# mydata$mRS<-factor(mydata$mRS,levels = c("0","1"))
mydata$sex<-factor(mydata$sex,levels = c("1","2"))
# mydata$dia<-factor(mydata$dia,levels = c("1","2"))
# mydata$OCSP<-factor(mydata$OCSP,levels = c("1","2","3","4"))
str(mydata)

library(rmda)
uPCX<- decision_curve(mRS~age+sex+TC+LDL+Hyper+dia+CHD+HCY+AF+
                        smoke+drink+TOAST+OCSP+inNIHSS+date,data = clinical,
                      family = binomial(link ='logit'),#模型类型，这里是二分类
                      thresholds= seq(0,1, by = 0.01),
                      confidence.intervals =0.95,#95可信区间
                      study.design = 'cohort')#研究类型，这里是队列研究
uPCX1<- decision_curve(mRS~LDL,data = clinical,
                      family = binomial(link ='logit'),#模型类型，这里是二分类
                      thresholds= seq(0,1, by = 0.01),
                      confidence.intervals =0.95,#95可信区间
                      study.design = 'cohort')#研究类型，这里是队列研究

clinicalparameters<-decision_curve(mRS~x1+x2+x3+x4+x5,data = pyrad, family = binomial(link ='logit'),
                                   thresholds= seq(0,1, by = 0.01),
                                   confidence.intervals =0.95,study.design ='cohort')

all<- decision_curve(mRS~sex+inNIHSS+date,data = mydata,
                     family = binomial(link='logit'),
                     thresholds= seq(0,1, by = 0.01),
                     confidence.intervals =0.95,study.design ='cohort')

List<-list(uPCX,uPCX1,clinicalparameters,all)

# plot_decision_curve(List,curve.names= c('all','clinical','pyrad'),
#                     ylim = c(-0.05, 0.15),
#                     cost.benefit.axis =FALSE,col = c('red','blue','green'),
#                     confidence.intervals =FALSE,standardize = FALSE,
#                     legend.position = "topright")

windowsFonts(myFont = windowsFont("Times New Roman"))
plot_decision_curve(List,curve.names= c('ALL','clinical','D','pyrad'),
                    ylim = c(-0.02, 0.15),
                    confidence.intervals =FALSE,standardize = FALSE ,
                    cost.benefit.axis =FALSE,
                    legend.position = 'none')
                   
