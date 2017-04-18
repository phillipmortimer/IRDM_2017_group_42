now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold2/train.txt -test ../../data/Fold2/test.txt -ranker 1 -norm linear -epoch 9 -layer 1 -node 20 -lr 0.00001 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/rn_f2_e9_l1_n20_lr-5.txt | tee ../../model/rn_f2_e9_l1_n20_lr-5.log
now=$(date +"%T")
echo "Current time : $now"
