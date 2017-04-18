now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -test ../../data/Fold1/test.txt -ranker 1 -norm linear -epoch 20 -layer 2 -node 10 -lr 0.01 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/RankNet/rn_f1_e20_l2_n10_lr-2.txt | tee ../../model/RankNet/rn_f1_e20_l2_n10_lr-2.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -test ../../data/Fold1/test.txt -ranker 1 -norm linear -epoch 20 -layer 2 -node 10 -lr 0.001 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/RankNet/rn_f1_e20_l2_n10_lr-3.txt | tee ../../model/RankNet/rn_f1_e20_l2_n10_lr-3.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -test ../../data/Fold1/test.txt -ranker 1 -norm linear -epoch 20 -layer 2 -node 10 -lr 0.0001 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/RankNet/rn_f1_e20_l2_n10_lr-4.txt | tee ../../model/RankNet/rn_f1_e20_l2_n10_lr-4.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -test ../../data/Fold1/test.txt -ranker 1 -norm linear -epoch 20 -layer 2 -node 10 -lr 0.00001 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/RankNet/rn_f1_e20_l2_n10_lr-5.txt | tee ../../model/RankNet/rn_f1_e20_l2_n10_lr-5.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -test ../../data/Fold1/test.txt -ranker 1 -norm linear -epoch 20 -layer 2 -node 10 -lr 0.000001 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/RankNet/rn_f1_e20_l2_n10_lr-6.txt | tee ../../model/RankNet/rn_f1_e20_l2_n10_lr-6.log
now=$(date +"%T")
