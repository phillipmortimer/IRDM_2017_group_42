now=$(date +"%T")
echo "Current time : $now"
java -jar ../RankLib/RankLib.jar -train ../data/Fold1/train.txt -test ../data/Fold1/test.txt -ranker 1 -norm linear -epoch 1 -layer 1 -node 20 -lr 0.00001 -save ../model/rn_f1_e1_l1_n20_lr-5.txt
now=$(date +"%T")
echo "Current time : $now"
java -jar ../RankLib/RankLib.jar -train ../data/Fold1/train.txt -test ../data/Fold1/test.txt -ranker 1 -norm linear -epoch 5 -layer 1 -node 20 -lr 0.00001 -save ../model/rn_f1_e5_l1_n20_lr-5.txt
now=$(date +"%T")
echo "Current time : $now"
java -jar ../RankLib/RankLib.jar -train ../data/Fold1/train.txt -test ../data/Fold1/test.txt -ranker 1 -norm linear -epoch 10 -layer 1 -node 20 -lr 0.00001 -save ../model/rn_f1_e10_l1_n20_lr-5.txt
now=$(date +"%T")
echo "Current time : $now"
java -jar ../RankLib/RankLib.jar -train ../data/Fold1/train.txt -test ../data/Fold1/test.txt -ranker 1 -norm linear -epoch 20 -layer 1 -node 20 -lr 0.00001 -save ../model/rn_f1_e20_l1_n20_lr-5.txt
now=$(date +"%T")
echo "Current time : $now"
java -jar ../RankLib/RankLib.jar -train ../data/Fold1/train.txt -test ../data/Fold1/test.txt -ranker 1 -norm linear -epoch 50 -layer 1 -node 20 -lr 0.00001 -save ../model/rn_f1_e50_l1_n20_lr-5.txt
now=$(date +"%T")
echo "Current time : $now"
