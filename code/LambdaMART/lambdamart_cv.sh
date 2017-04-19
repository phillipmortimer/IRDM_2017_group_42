# NDCG
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -validate ../../data/Fold1/vali.txt -test ../../data/Fold1/test.txt -ranker 6 -norm linear -tree 250 -tc 256 -leaf 100 -shrinkage 0.01 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/LambdaMART/lm_f1_tc256_lf100_sh001_ndcg.txt | tee ../../model/LambdaMART/lm_f1_tc256_lf100_sh001_ndcg.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold2/train.txt -validate ../../data/Fold2/vali.txt -test ../../data/Fold2/test.txt -ranker 6 -norm linear -tree 250 -tc 256 -leaf 100 -shrinkage 0.01 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/LambdaMART/lm_f2_tc256_lf100_sh001_ndcg.txt | tee ../../model/LambdaMART/lm_f2_tc256_lf100_sh001_ndcg.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold3/train.txt -validate ../../data/Fold3/vali.txt -test ../../data/Fold3/test.txt -ranker 6 -norm linear -tree 250 -tc 256 -leaf 100 -shrinkage 0.01 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/LambdaMART/lm_f3_tc256_lf100_sh001_ndcg.txt | tee ../../model/LambdaMART/lm_f3_tc256_lf100_sh001_ndcg.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold4/train.txt -validate ../../data/Fold4/vali.txt -test ../../data/Fold4/test.txt -ranker 6 -norm linear -tree 250 -tc 256 -leaf 100 -shrinkage 0.01 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/LambdaMART/lm_f4_tc256_lf100_sh001_ndcg.txt | tee ../../model/LambdaMART/lm_f4_tc256_lf100_sh001_ndcg.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold5/train.txt -validate ../../data/Fold5/vali.txt -test ../../data/Fold5/test.txt -ranker 6 -norm linear -tree 250 -tc 256 -leaf 100 -shrinkage 0.01 -metric2t NDCG@10 -metric2T NDCG@10 -save ../../model/LambdaMART/lm_f5_tc256_lf100_sh001_ndcg.txt | tee ../../model/LambdaMART/lm_f5_tc256_lf100_sh001_ndcg.txt.log
now=$(date +"%T")
echo "Current time : $now"

# ERR
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold1/train.txt -validate ../../data/Fold1/vali.txt -test ../../data/Fold1/test.txt -ranker 6 -norm linear -tree 250 -tc 256 -leaf 100 -shrinkage 0.01 -metric2t ERR@10 -metric2T ERR@10 -save ../../model/LambdaMART/lm_f1_tc256_lf100_sh001_err.txt | tee ../../model/LambdaMART/lm_f1_tc256_lf100_sh001_err.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold2/train.txt -validate ../../data/Fold2/vali.txt -test ../../data/Fold2/test.txt -ranker 6 -norm linear -tree 250 -tc 256 -leaf 100 -shrinkage 0.01 -metric2t ERR@10 -metric2T ERR@10 -save ../../model/LambdaMART/lm_f2_tc256_lf100_sh001_err.txt | tee ../../model/LambdaMART/lm_f2_tc256_lf100_sh001_err.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold3/train.txt -validate ../../data/Fold3/vali.txt -test ../../data/Fold3/test.txt -ranker 6 -norm linear -tree 250 -tc 256 -leaf 100 -shrinkage 0.01 -metric2t ERR@10 -metric2T ERR@10 -save ../../model/LambdaMART/lm_f3_tc256_lf100_sh001_err.txt | tee ../../model/LambdaMART/lm_f3_tc256_lf100_sh001_err.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold4/train.txt -validate ../../data/Fold4/vali.txt -test ../../data/Fold4/test.txt -ranker 6 -norm linear -tree 250 -tc 256 -leaf 100 -shrinkage 0.01 -metric2t ERR@10 -metric2T ERR@10 -save ../../model/LambdaMART/lm_f4_tc256_lf100_sh001_err.txt | tee ../../model/LambdaMART/lm_f4_tc256_lf100_sh001_err.txt.log
now=$(date +"%T")
echo "Current time : $now"
java -jar ../../RankLib/RankLib.jar -train ../../data/Fold5/train.txt -validate ../../data/Fold5/vali.txt -test ../../data/Fold5/test.txt -ranker 6 -norm linear -tree 250 -tc 256 -leaf 100 -shrinkage 0.01 -metric2t ERR@10 -metric2T ERR@10 -save ../../model/LambdaMART/lm_f5_tc256_lf100_sh001_err.txt | tee ../../model/LambdaMART/lm_f5_tc256_lf100_sh001_err.txt.log
now=$(date +"%T")
echo "Current time : $now"

