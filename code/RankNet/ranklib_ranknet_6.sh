now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -validate ../../data/Fold1/vali.txt -test ../../data/Fold1/test.txt -ranker 1 -norm linear -epoch 20 -layer 2 -node 5 -lr 0.000001 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/RankNet/rn_f1_e20_l2_n5_lr-6.txt | tee ../../model/RankNet/rn_f1_e20_l2_n5_lr-6.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -validate ../../data/Fold1/vali.txt -test ../../data/Fold1/test.txt -ranker 1 -norm linear -epoch 20 -layer 2 -node 10 -lr 0.000001 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/RankNet/rn_f1_e20_l2_n10_lr-6.txt | tee ../../model/RankNet/rn_f1_e20_l2_n10_lr-6.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -validate ../../data/Fold1/vali.txt -test ../../data/Fold1/test.txt -ranker 1 -norm linear -epoch 20 -layer 2 -node 20 -lr 0.000001 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/RankNet/rn_f1_e20_l2_n20_lr-6.txt | tee ../../model/RankNet/rn_f1_e20_l2_n20_lr-6.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -validate ../../data/Fold1/vali.txt -test ../../data/Fold1/test.txt -ranker 1 -norm linear -epoch 20 -layer 2 -node 40 -lr 0.000001 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/RankNet/rn_f1_e20_l2_n40_lr-6.txt | tee ../../model/RankNet/rn_f1_e20_l2_n40_lr-6.log
now=$(date +"%T")
echo "Current time : $now"