now=$(date +"%T")
echo "Current time : $now"
java -jar ../RankLib/RankLib.jar -train ../data/Fold1/train.txt -test ../data/Fold1/test.txt -ranker 1 -norm linear -epoch 20 -layer 2 -node 2 -lr 0.00001 -save ../model/rn_f1_e20_l2_n2_lr-5.txt
now=$(date +"%T")
echo "Current time : $now"
java -jar ../RankLib/RankLib.jar -train ../data/Fold1/train.txt -test ../data/Fold1/test.txt -ranker 1 -norm linear -epoch 20 -layer 2 -node 5 -lr 0.00001 -save ../model/rn_f1_e20_l2_n5_lr-5.txt
now=$(date +"%T")
echo "Current time : $now"
java -jar ../RankLib/RankLib.jar -train ../data/Fold1/train.txt -test ../data/Fold1/test.txt -ranker 1 -norm linear -epoch 20 -layer 2 -node 10 -lr 0.00001 -save ../model/rn_f1_e20_l2_n10_lr-5.txt
now=$(date +"%T")
echo "Current time : $now"
java -jar ../RankLib/RankLib.jar -train ../data/Fold1/train.txt -test ../data/Fold1/test.txt -ranker 1 -norm linear -epoch 20 -layer 2 -node 20 -lr 0.00001 -save ../model/rn_f1_e20_l2_n20_lr-5.txt
now=$(date +"%T")
echo "Current time : $now"
java -jar ../RankLib/RankLib.jar -train ../data/Fold1/train.txt -test ../data/Fold1/test.txt -ranker 1 -norm linear -epoch 20 -layer 2 -node 50 -lr 0.00001 -save ../model/rn_f1_e20_l2_n50_lr-5.txt
now=$(date +"%T")
echo "Current time : $now"
