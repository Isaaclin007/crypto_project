rm(list=ls())
library(readr)
library(ggplot2)
library(grid)
library(gridExtra)
library(Hmisc)
library(tidyverse)
library(lubridate)
library(plyr)
library(urca)
library(TTR)
library(xts)


start_insample = 1538392290 # oct
end_insample = 1541070690 # nov

##### Clean Data ####

cex_eth_prices = read_csv("./prices/prices_cex.csv")
cex_eth_prices = cex_eth_prices %>% 
  filter(utc>start_insample)

as.character(cex_eth_prices$utc, format = "%y-%m-%d %H:%M")
cex_eth_prices['utc'] = as.character(cex_eth_prices$utc, format = "%y-%m-%d %H:%M")


quoinex_eth_prices = read_csv("./prices/prices_quoinex.csv")
quoinex_eth_prices = quoinex_eth_prices %>% 
  filter(utc>start_insample) 
quoinex_eth_prices['utc'] = as.character(quoinex_eth_prices$utc, format = "%y-%m-%d %H:%M")

dt = merge(quoinex_eth_prices, cex_eth_prices,by = c("utc")) # full sample
dt_train = dt[1:70000,]
dt_out = dt[70000:length(dt),]

##### test ####
y = dt$last.y # cex close
x = dt$last.x # liquid close
result = lm(y ~ x -1)
beta_ols =coefficients(result)[1]
et = residuals(result)
random.data = residuals(result)
summary(et)
plot.ecdf(et) # very stationary - look at normal

jotest=ca.jo(data.frame(y,x), type="trace", K=2, ecdet="none", spec="longrun")
summary(jotest) # cointegrate

##### check the robustness of OLS estimator and Johansen estimatpr - almost the same
s = 1.000*y  -1.0034*x
plot(s, type="l")
plot(et, type="l")
compare = data.frame(dt$utc,et,s)
par(mfrow=c(1,2)) # Partition or create a 2*2 matrix plot for 4 graphs 
for(i in 2:3)
{
  qqnorm(compare[,i], datax=TRUE, main=i)  #logical=[True, False] datax->sample quantiles is in x
  qqline(compare[,i], datax=T) # add a 45 degree line
}


##### determine the lookback period ####
y.lag <- c(random.data[2:length(random.data)], 0)
y.lag <- y.lag[1:length(y.lag)-1] # As shifted vector by -1, remove anomalous element at end of vector
random.data <- random.data[1:length(random.data)-1] # Make vector same length as vector y.lag
y.diff <- random.data - y.lag # Subtract todays close from yesterdays close
y.diff <- y.diff [1:length(y.diff)-1] # Make vector same length as vector y.lag
prev.y.mean <- y.lag - mean(y.lag) # Subtract yesterdays close from the mean of lagged differences
prev.y.mean <- prev.y.mean [1:length(prev.y.mean )-1] # Make vector same length as vector y.lag

# Linear Regression With Intercept
result <- lm(y.diff ~ prev.y.mean+0)
half_life <- -log(2)/coef(result)[1]
lookback = ceiling(half_life) 

###### construct trading rule - Z-score ####
y = data.frame(dt$utc,et)
ma <- function(arr, n=2){
  res = arr
  for(i in n:length(arr)){
    res[i] = mean(arr[(i-n+1):i])
  }
  res
}

masd <- function(arr, n=2){
  res = arr
  for(i in n:length(arr)){
    res[i] = sd(arr[(i-n+1):i])
  }
  res
}


MAmean = ma(y$et, n=lookback)
MAsd = masd(y$et, n = lookback)
dtt=data.frame(y,MAmean,MAsd)
mktVal=-(y$et-MAmean)/MAsd;
y = data.frame(y, dt$last.y, dt$last.x,beta_ols,MAmean,MAsd,mktVal)
#y = y[lookback:length(y$dt.utc),]

total = y %>% 
  mutate(excute = ifelse(abs(mktVal)>3.5,  mktVal, 0),
         et1 = lead(et,lookback),
         profit = excute*(et1-et) - abs(excute)*dt.last.x * 0.0025,
         cum_profit = cumsum(ifelse(is.na(profit), 0, profit)),
         time = ymd_hm(dt.utc)
  )

ggplot(data = total) +
  geom_line(aes(x=time,y=cum_profit))
summary(total$profit)

trades_num = length( total$excute[total$excute != 0 & !is.na(total$excute)] ) # number of trade
trades_percent = trades_num/nrow(total)
trades_profit = total$profit[total$excute != 0 & !is.na(total$excute)]
hist(trades_profit)


##### special example #####
max_profit = total[which.max(total$profit),]
min_profit = total[which.min(total$profit),]
na_profit = total[which(is.na(total$profit)),]  # no sd, market has no trades.








