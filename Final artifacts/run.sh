#!/bin/bash


if [ $# != 6 ]
then
echo 'Usage: '
echo 'Please run the file in the below format'
echo './run.sh -m <location of the model dump>  -t <location of the test samples directory>  -l <location of the test labels file>'
echo 'Example: ./run.sh -m ~/Desktop/model/ -t ~/Desktop/train/ -l ~/Desktop/p1_train_pop_lab_test_label.csv'
exit 0
fi

modelLoc=$2
trainLoc=$4
testLabelLoc=$6

#echo $modelLoc
#echo $trainLoc
#echo $testLabelLoc

echo 
echo "Please wait..."

#Handle ambig_info & equivalance class

for j in "$trainLoc"/*
do
rm $j/bias/aux_info/equi.txt 2>/dev/null 
#echo  $j/bias/aux_info/eq_classes.txt
N=$(awk 'NR == 1{print}' $j/bias/aux_info/eq_classes.txt); 
v1=$(awk -v NN=$N 'NR>(NN + 3){ print $NF, $0 }' $j/bias/aux_info/eq_classes.txt | sort -nr -k1 | head -100 | sort -n -k1)
v2=$(awk '{for(i=1;i<NF;i++){printf "%d ", $i; if(i==(NF-1))printf "\n"}}' <<< "$v1")

while read line 
do
  i=1
  for indx in $line;
  do
    if [[ $i == 1 ]]
    then
      val=$indx
      i=$((i+1))
      continue
    fi

    if [[ $i != 2 ]]
    then
       read_cnt_arr["$indx"]=$val
     fi
    i=$((i + 1))
  done
done <<< "$v2"

echo "Eq_Class" > $j/bias/aux_info/equi.txt
for((i=0;i<$N;i++))
do
  if [[ -n ${read_cnt_arr["$i"]} ]]
  then
    echo  ${read_cnt_arr["$i"]} >>$j/bias/aux_info/equi.txt
  else
    echo "0" >>$j/bias/aux_info/equi.txt
  fi
done


 #echo $j/bias/quant.sf 
 paste $j/bias/quant.sf $j/bias/aux_info/ambig_info.tsv $j/bias/aux_info/equi.txt | awk -v OFS='\t' '{print $1,$3,$4,$5,$6,$7,$8}' >$j/bias/quant2.sf 
 echo -n "."
done

#call python script here.
echo 
echo
python PhenotypicPredictor.py "-m" "$modelLoc" "-t" "$trainLoc" "-l" "$testLabelLoc"
