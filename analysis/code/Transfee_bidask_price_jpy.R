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
# norice here cex price is the quoinex_Jpy 
cex_eth_prices = read_csv("/Users/Zi/crypto_project/build/inputs/raw/prices/prices_quoinex_jpy.csv")
cex_eth_prices = cex_eth_prices %>% 
  filter(utc>start_insample)


cex_eth_prices['utc'] = as.character(cex_eth_prices$utc, format = "%y-%m-%d %H:%M")


quoinex_eth_prices = read_csv("/Users/Zi/crypto_project/build/inputs/raw/prices/prices_quoinex.csv")
quoinex_eth_prices = quoinex_eth_prices %>% 
  filter(utc>start_insample) 
quoinex_eth_prices['utc'] = as.character(quoinex_eth_prices$utc, format = "%y-%m-%d %H:%M")

dt = merge(quoinex_eth_prices, cex_eth_prices,by = c("utc")) # full sample
dt_train = dt[1:70000,]
dt_out = dt[70000:length(dt),]

##### test ####
usdjpy = 112
y = dt$last.y/usdjpy # cex close, usd = 112 jpy
x = dt$last.x # liquid close
result = lm(y ~ x -1)
beta_ols =coefficients(result)[1]
et = residuals(result)
random.data = residuals(result)
summary(et)
plot.ecdf(et) # very stationary - look at normal

# jotest=ca.jo(data.frame(y,x), type="trace", K=2, ecdet="none", spec="longrun")
# summary(jotest) # cointegrate

##### check the robustness of OLS estimator and Johansen estimatpr - almost the same
# s = 1.000*y  -1.0034*x
# plot(s, type="l")
# plot(et, type="l")
# compare = data.frame(dt$utc,et,s)
# #par(mfrow=c(1,2)) # Partition or create a 2*2 matrix plot for 4 graphs 
# # for(i in 2:3)
# # {
# #   qqnorm(compare[,i], datax=TRUE, main=i)  #logical=[True, False] datax->sample quantiles is in x
# #   qqline(compare[,i], datax=T) # add a 45 degree line
# # }


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
dt$sell.y=dt$sell.y/usdjpy
dt$last.y=dt$last.y/usdjpy
dt$buy.y=dt$buy.y/usdjpy
y = data.frame(y, dt$sell.y, dt$last.y, dt$buy.y, dt$buy.x,dt$last.x,dt$sell.x,beta_ols,MAmean,MAsd,mktVal)
#y = y[lookback:length(y$dt.utc),]

total = y %>% 
  mutate(excute = ifelse(abs(mktVal)>1,  mktVal, 0),
         et.xbidyask = dt.sell.y - beta_ols*dt.sell.x , #notice buy.y > last.y > sell.y
         et.ybidxask = dt.buy.y - beta_ols*dt.buy.x, #notice sell.x > last.x > buy.x
         et.ahead = lead(et,lookback),
         et.real = ifelse(et<0, et.ybidxask, et.xbidyask), # if et<0, we should buy y sell 1.02x
         et.real_ahead = lead(et.real,lookback), # hold the asset after 17 periods
         et.real.reverse = ifelse(et>0, et.ybidxask, et.xbidyask),
         et.real_ahead_reverse = lead(et.real.reverse,lookback),
         profit_close = excute*(et.ahead - et),
         profit_close_fee = excute*(et.ahead - et)  - abs(excute)*dt.last.x * 0.0025,
         profit_bidask = excute*(et.real - et.real_ahead),
         profit_bidask_fee = excute*(et.real - et.real_ahead)  - abs(excute)*dt.last.x * 0.0025,
         cum_profit_close = cumsum(ifelse(is.na(profit_close), 0, profit_close)),
         cum_profit_fee = cumsum(ifelse(is.na(profit_close_fee), 0, profit_close_fee)),
         cum_profit_bidask_fee = cumsum(ifelse(is.na(profit_bidask_fee), 0, profit_bidask_fee)),
         cum_profit_bidask = cumsum(ifelse(is.na(profit_bidask), 0, profit_bidask)),
         time = ymd_hm(dt.utc)
  )

mm = total[,c("et",'et.ahead',"et.real",'et.real_ahead',"et.real.reverse",'et.real_ahead_reverse')]


ggplot(data = total) +
  geom_line(aes(x=time,y=cum_profit_close))


ggplot(data = total) +
  geom_line(aes(x=time,y=cum_profit_fee))


ggplot(data = total) +
  geom_line(aes(x=time,y=cum_profit_bidask))


ggplot(data = total) +
  geom_line(aes(x=time,y=cum_profit_bidask_fee))


trades_num = length( total$excute[total$excute != 0 & !is.na(total$excute)] ) # number of trade
trades_percent = trades_num/nrow(total)
trades_profit = total$profit_bidask[total$excute != 0 & !is.na(total$excute)]
hist(trades_profit)
profit_dt = as.data.frame(trades_profit)
ggplot(profit_dt, aes(x=trades_profit)) + stat_bin(binwidth = 0.3)

##### special example #####
max_profit = total[which.max(total$profit_bidask),]
min_profit = total[which.min(total$profit_bidask),]
na_profit = total[which(is.na(total$profit_bidask)),]  # no sd, market has no trades.



### save data
saveRDS(y, file = "dt.rds")




