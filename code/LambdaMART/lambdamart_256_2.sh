now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -validate ../../data/Fold1/vali.txt -test ../../data/Fold1/test.txt -ranker 6 -norm linear -tree 250 -tc 256 -leaf 1 -shrinkage 0.1 -estop 25 -metric2t NDCG@10 -estop 25 -metric2t NDCG@10 -save ../../model/LambdaMART/lm_tc256_lf1_sh01.txt | tee ../../model/LambdaMART/lm_tc256_lf1_sh01.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -validate ../../data/Fold1/vali.txt -test ../../data/Fold1/test.txt -ranker 6 -norm linear -tree 250 -tc 256 -leaf 10 -shrinkage 0.1 -estop 25 -metric2t NDCG@10 -estop 25 -metric2t NDCG@10 -save ../../model/LambdaMART/lm_tc256_lf10_sh01.txt | tee ../../model/LambdaMART/lm_tc256_lf10_sh01.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -validate ../../data/Fold1/vali.txt -test ../../data/Fold1/test.txt -ranker 6 -norm linear -tree 250 -tc 256 -leaf 100 -shrinkage 0.1 -estop 25 -metric2t NDCG@10 -estop 25 -metric2t NDCG@10 -save ../../model/LambdaMART/lm_tc256_lf100_sh01.txt | tee ../../model/LambdaMART/lm_tc256_lf100_sh01.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -validate ../../data/Fold1/vali.txt -test ../../data/Fold1/test.txt -ranker 6 -norm linear -tree 250 -tc 256 -leaf 1000 -shrinkage 0.1 -estop 25 -metric2t NDCG@10 -estop 25 -metric2t NDCG@10 -save ../../model/LambdaMART/lm_tc256_lf1000_sh01.txt | tee ../../model/LambdaMART/lm_tc256_lf1000_sh01.txt.log

