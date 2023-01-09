#!/bin/sh
source /etc/profile
source ~/.bash_profile

start_time=$(date +%s)

echo "******************8"
export JAVA_HOME=/home/hdfs/hadoop/jdk1.8.0_60
export HADOOP_HOME=/home/hdfs/hadoop/
export PATH=$PATH:$JAVA_HOME/bin:$HADOOP_HOME/bin
echo "******************8"
#echo $CLASSPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${JAVA_HOME}/jre/lib/amd64/server
export CLASSPATH=$($HADOOP_HOME/bin/hadoop_sz classpath --glob)


YESTEARDAY=`date -d "yesterday" +%Y%m%d`
echo ${YESTEARDAY}

count=0
ret=1
while [ $ret -ne 0 ]
do
        let count++
        source /workspace/hdfs_env.sh
        ret=$?
        if [ $count -ge 20 ];then
                break
        fi
done

#python -u /workspace/eoe_run.py
python -u /workspace/eoe_run_wnd.py

#python -u /workspace/run_newuser_dnn.py

