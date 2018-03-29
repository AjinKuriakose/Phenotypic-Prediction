for i in train/*
do
 echo $i/bias/quant.sf 
 paste ./$i/bias/quant.sf ./$i/bias/aux_info/ambig_info.tsv >./$i/bias/quant1.sf
done