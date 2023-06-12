library(readxl)
library(tidyverse)
library(rms)
library(foreign)

mydata <- read.table("D:/Desktop/stroke3_chen/R code/dataset/stroke2_joint_train.csv",header = T,sep = ',')
str(mydata)
mydata$mRS<-factor(mydata$mRS,levels = c("0","1","2","3"))
mydata$sex<-factor(mydata$sex,levels = c("1","2"))
mydata$dia<-factor(mydata$dia,levels = c("1","2"))
mydata$OCSP<-factor(mydata$OCSP,levels = c("1","2","3","4"))
str(mydata)
# attach(mydata)

mydata1 <- read.table("D:/Desktop/stroke3_chen/R code/dataset/stroke2_joint_test.csv",header = T,sep = ',')
str(mydata1)
mydata1$mRS<-factor(mydata1$mRS,levels = c("0","1","2","3"))
mydata1$sex<-factor(mydata1$sex,levels = c("1","2"))
mydata1$dia<-factor(mydata1$dia,levels = c("1","2"))
mydata1$OCSP<-factor(mydata1$OCSP,levels = c("1","2","3","4"))
str(mydata1)
# attach(mydata1)
# fit1<-glm(mRS~sex+dia+inNIHSS+date+Red_score+OCSP,
#                 data = mydata,x=TRUE, y=TRUE,family = binomial(link='logit'),
#                 control=list(maxit=100))
# model.result<-summary(fit1)

dev = mydata
vad = mydata1

dd<-datadist(dev)
options(datadist='dd')

fit.dev<-lrm(mRS~sex+dia+Redscore+OCSP+Deepsurv,,data=dev,x=T,y=T)

fun2 <- function(x) plogis(x-fit.dev$coef[1]+fit.dev$coef[2])
fun3 <- function(x) plogis(x-fit.dev$coef[1]+fit.dev$coef[3])

nom.ord <- nomogram(fit.dev, fun=list('Prob Y>=1'=plogis,
                                'Prob Y>=2'=fun2,
                                'Prob Y=3'=fun3),
                    lp=F,
                    fun.at=c(.01,seq(.1,.9,by=.2),.95,.99))
plot(nom.ord, lmgp=.2, cex.axis=.6)


tt<-datadist(vad)
options(datadist='tt')

fit.vad<-lrm(mRS~sex+dia+Redscore+OCSP+Deepsurv,,data=vad,x=T,y=T)

fun4 <- function(x) plogis(x-fit.vad$coef[1]+fit.vad$coef[2])
fun5 <- function(x) plogis(x-fit.vad$coef[1]+fit.vad$coef[3])

nom.vad<- nomogram(fit.vad, fun=list('Prob Y>=1'=plogis,
                                      'Prob Y>=2'=fun4,
                                      'Prob Y=3'=fun5),
                    lp=F,
                    fun.at=c(.01,seq(.1,.9,by=.2),.95,.99))
plot(nom.vad, lmgp=.2, cex.axis=.6)

