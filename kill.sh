kill -9 $(ps -aux | grep 'pretrain_g'| awk '{print $2}' |xargs)
