now=$(date +"%T")
echo "Current time : $now"
java -jar ../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -test ../../data/Fold1/test.txt -ranker 3 -norm linear -round 500 -tolerance 0.002 -max 1 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/AdaRank/ar_r500_t002_m1.txt | tee ../../model/AdaRank/ar_r500_t002_m1.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -test ../../data/Fold1/test.txt -ranker 3 -norm linear -round 500 -tolerance 0.002 -max 2 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/AdaRank/ar_r500_t002_m2.txt | tee ../../model/AdaRank/ar_r500_t002_m2.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -test ../../data/Fold1/test.txt -ranker 3 -norm linear -round 500 -tolerance 0.002 -max 5 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/AdaRank/ar_r500_t002_m5.txt | tee ../../model/AdaRank/ar_r500_t002_m5.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -test ../../data/Fold1/test.txt -ranker 3 -norm linear -round 500 -tolerance 0.002 -max 10 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/AdaRank/ar_r500_t002_m10.txt | tee ../../model/AdaRank/ar_r500_t002_m10.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -test ../../data/Fold1/test.txt -ranker 3 -norm linear -round 500 -tolerance 0.002 -max 20 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/AdaRank/ar_r500_t002_m20.txt | tee ../../model/AdaRank/ar_r500_t002_m20.txt.log
now=$(date +"%T")
echo "Current time : $now"