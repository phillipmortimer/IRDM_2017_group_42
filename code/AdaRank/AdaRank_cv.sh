now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -validate ../../data/Fold1/vali.txt -test ../../data/Fold1/test.txt -ranker 3 -norm linear -round 500 -tolerance 0.002 -max 5 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/AdaRank/ar_f1_r500_t002_m5.txt | tee ../../model/AdaRank/ar_f1_r500_t002_m5.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold2/train.txt -validate ../../data/Fold2/vali.txt -test ../../data/Fold2/test.txt -ranker 3 -norm linear -round 500 -tolerance 0.002 -max 5 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/AdaRank/ar_f2_r500_t002_m5.txt | tee ../../model/AdaRank/ar_f2_r500_t002_m5.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold3/train.txt -validate ../../data/Fold3/vali.txt -test ../../data/Fold3/test.txt -ranker 3 -norm linear -round 500 -tolerance 0.002 -max 5 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/AdaRank/ar_f3_r500_t002_m5.txt | tee ../../model/AdaRank/ar_f3_r500_t002_m5.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold4/train.txt -validate ../../data/Fold4/vali.txt -test ../../data/Fold4/test.txt -ranker 3 -norm linear -round 500 -tolerance 0.002 -max 5 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/AdaRank/ar_f4_r500_t002_m5.txt | tee ../../model/AdaRank/ar_f4_r500_t002_m5.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold5/train.txt -validate ../../data/Fold5/vali.txt -test ../../data/Fold5/test.txt -ranker 3 -norm linear -round 500 -tolerance 0.002 -max 5 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/AdaRank/ar_f5_r500_t002_m5.txt | tee ../../model/AdaRank/ar_f5_r500_t002_m5.txt.log
now=$(date +"%T")
echo "Current time : $now"