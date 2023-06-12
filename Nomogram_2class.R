library(readxl)
library(tidyverse)
library(rms)
library(foreign)

mydata <- read.table("D:/Desktop/stroke3_chen/R code/dataset/stroke2_joint.csv",header = T,sep = ',')
str(mydata)
mydata$mRS<-factor(mydata$mRS,levels = c("0","1"))
mydata$sex<-factor(mydata$sex,levels = c("0","1"))
mydata$dia<-factor(mydata$dia,levels = c("0","1"))
mydata$OCSP<-factor(mydata$OCSP,levels = c("1","2","3","4"))
str(mydata)
# attach(mydata)

mydata1 <- read.table("D:/Desktop/stroke3_chen/R code/dataset/stroke2_joint.csv",header = T,sep = ',')
str(mydata1)
mydata1$mRS<-factor(mydata1$mRS,levels = c("0","1"))
mydata1$sex<-factor(mydata1$sex,levels = c("0","1"))
mydata1$dia<-factor(mydata1$dia,levels = c("0","1"))
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

fit.dev<-lrm(mRS~sex+dia+Red_score+OCSP,,data=dev,x=T,y=T)

nom.dev<- nomogram(fit.dev, fun=plogis,fun.at=c(.001, .01, seq(.1,.9, by=.3), .95, .99, .999),lp=F, funlabel="Probability of early recurrence")
plot(nom.dev)

cal1 <- calibrate(fit.dev, method='boot', B=1000)
# plot(cal1,xlim=c(0,1.0),ylim=c(0,1.0),col = c("black","red","green"))
plot(cal1,
     xlim = c(0,1),
     xlab = "Predicted Probability",
     ylab = "Observed Probability",
     legend = FALSE,
     subtitles = FALSE)
abline(0,1,col = "black",lty = 2,lwd = 2)
lines(cal1[,c("predy","calibrated.orig")], type = "l",lwd = 2,col="red",pch =16)
lines(cal1[,c("predy","calibrated.corrected")], type = "l",lwd = 2,col="green",pch =16)
legend(0.4,0.5,
       c("Apparent","Ideal","Bias-corrected"),
       lty = c(2,1,1),
       lwd = c(2,1,1),
       col = c("black","red","green"),
       bty = "n") # "o"为加边框

tt<-datadist(vad)
options(datadist='tt')

fit.vad<-lrm(mRS~sex+dia+Red_score+OCSP,,data=vad,x=T,y=T)

nom.vad<- nomogram(fit.vad, fun=plogis,fun.at=c(.001, .01, seq(.1,.9, by=.3), .95, .99, .999),lp=F, funlabel="Probability of early recurrence")
plot(nom.vad)

cal2 <- calibrate(fit.vad, method='boot', B=1000)
# plot(cal1,xlim=c(0,1.0),ylim=c(0,1.0),col = c("black","red","green"))
plot(cal2,
     xlim = c(0,1),
     xlab = "Predicted Probability",
     ylab = "Observed Probability",
     legend = FALSE,
     subtitles = FALSE)
abline(0,1,col = "black",lty = 2,lwd = 2)
lines(cal2[,c("predy","calibrated.orig")], type = "l",lwd = 2,col="red",pch =16)
lines(cal2[,c("predy","calibrated.corrected")], type = "l",lwd = 2,col="green",pch =16)
legend(0.2,0.2,
       c("Apparent","Ideal","Bias-corrected"),
       lty = c(2,1,1),
       lwd = c(2,1,1),
       col = c("black","red","green"),
       bty = "n") # "o"为加边框