now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -test ../../data/Fold1/test.txt -ranker 3 -norm linear -round 500 -tolerance 0.01 -max 5 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/AdaRank/ar_r500_t01_m5.txt | tee ../../model/AdaRank/ar_r500_t01_m5.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -test ../../data/Fold1/test.txt -ranker 3 -norm linear -round 500 -tolerance 0.005 -max 5 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/AdaRank/ar_r500_t005_m5.txt | tee ../../model/AdaRank/ar_r500_t005_m5.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -test ../../data/Fold1/test.txt -ranker 3 -norm linear -round 500 -tolerance 0.002 -max 5 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/AdaRank/ar_r500_t002_m5.txt | tee ../../model/AdaRank/ar_r500_t002_m5.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -test ../../data/Fold1/test.txt -ranker 3 -norm linear -round 500 -tolerance 0.001 -max 5 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/AdaRank/ar_r500_t001_m5.txt | tee ../../model/AdaRank/ar_r500_t001_m5.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -test ../../data/Fold1/test.txt -ranker 3 -norm linear -round 500 -tolerance 0.0005 -max 5 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/AdaRank/ar_r500_t0005_m5.txt | tee ../../model/AdaRank/ar_r500_t0005_m5.txt.log
now=$(date +"%T")
echo "Current time : $now"