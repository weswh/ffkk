#! /bin/bash
# kill -9 $(lsof -i:3001 | awk 'NR==2{print $2}')
pm2 delete aiserver