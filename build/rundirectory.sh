#!/bin/bash 
#The script works like a roadmap, telling the operating system how to run the directory
node /Users/Zi/alfie-alt/indodax/hist.js > /Users/Zi/bitcoincoid_alt_trans.csv
mv /Users/Zi/bitcoincoid_alt_trans.csv /Users/Zi/Dropbox/alfie/Jan_alt/trans/


node /Users/Zi/alfie-alt/kucoin/hist.js > /Users/Zi/kucoin_alt_trans.csv
mv /Users/Zi/kucoin_alt_trans.csv /Users/Zi/Dropbox/alfie/Jan_alt/trans/


rsync -avz -e ssh root@206.189.169.129:/root/alfie-alt/balances /Users/Zi/Dropbox/alfie/Jan_alt --exclude='env' --exclude='*.py'

rsync -avz -e ssh root@159.89.130.68:/root/alfie-alt/prices /Users/Zi/Dropbox/alfie/Jan_alt --exclude='*.py'

rsync -avz -e ssh root@159.89.130.68:/root/alfie-alt/arb_fetcher /Users/Zi/Dropbox/alfie/Jan_alt --exclude='*.js'

cd /Users/Zi/alfie-alt/prices
python /Users/Zi/alfie-alt/prices/Pricefetcher_gdax_eth_usd.py 
mv /Users/Zi/alfie-alt/prices/prices_gdax_usd.csv /Users/Zi/Dropbox/alfie/Jan_alt/prices/

cd /Users/Zi/alfie-alt/prices
python /Users/Zi/alfie-alt/prices/Pricefetcher_gdax_eth.py 
mv /Users/Zi/alfie-alt/prices/prices_gdax_eur.csv /Users/Zi/Dropbox/alfie/Jan_alt/prices/
