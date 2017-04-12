now=$(date +"%T")
echo "Current time : $now"
java -jar ../RankLib/RankLib.jar -train ../data/Fold1/train.txt -test ../data/Fold1/test.txt -ranker 1 -norm linear -epoch 20 -layer 1 -node 10 -lr 0.01 -save ../model/rn_f1_e20_l1_n10_lr-2.txt
now=$(date +"%T")
echo "Current time : $now"
java -jar ../RankLib/RankLib.jar -train ../data/Fold1/train.txt -test ../data/Fold1/test.txt -ranker 1 -norm linear -epoch 20 -layer 1 -node 10 -lr 0.001 -save ../model/rn_f1_e20_l1_n10_lr-3.txt
now=$(date +"%T")
echo "Current time : $now"
java -jar ../RankLib/RankLib.jar -train ../data/Fold1/train.txt -test ../data/Fold1/test.txt -ranker 1 -norm linear -epoch 20 -layer 1 -node 10 -lr 0.0001 -save ../model/rn_f1_e20_l1_n10_lr-4.txt
now=$(date +"%T")
echo "Current time : $now"
java -jar ../RankLib/RankLib.jar -train ../data/Fold1/train.txt -test ../data/Fold1/test.txt -ranker 1 -norm linear -epoch 20 -layer 1 -node 10 -lr 0.00001 -save ../model/rn_f1_e20_l1_n10_lr-5.txt
now=$(date +"%T")
echo "Current time : $now"
java -jar ../RankLib/RankLib.jar -train ../data/Fold1/train.txt -test ../data/Fold1/test.txt -ranker 1 -norm linear -epoch 20 -layer 1 -node 10 -lr 0.000001 -save ../model/rn_f1_e20_l1_n10_lr-6.txt
now=$(date +"%T")
