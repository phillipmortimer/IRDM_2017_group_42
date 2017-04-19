# Fold 1
java -jar ../../RankLib/RankLib.jar -load ../../model/AdaRank/ar_f1_r500_t002_m5.txt -test ../../data/Fold1/test.txt -norm linear -metric2t NDCG@10
java -jar ../../RankLib/RankLib.jar -load ../../model/AdaRank/ar_f1_r500_t002_m5.txt -test ../../data/Fold1/test.txt -norm linear -metric2t ERR@10

# Fold 2
java -jar ../../RankLib/RankLib.jar -load ../../model/AdaRank/ar_f2_r500_t002_m5.txt -test ../../data/Fold2/test.txt -norm linear -metric2t NDCG@10
java -jar ../../RankLib/RankLib.jar -load ../../model/AdaRank/ar_f2_r500_t002_m5.txt -test ../../data/Fold2/test.txt -norm linear -metric2t ERR@10

# Fold 3
java -jar ../../RankLib/RankLib.jar -load ../../model/AdaRank/ar_f3_r500_t002_m5.txt -test ../../data/Fold3/test.txt -norm linear -metric2t NDCG@10
java -jar ../../RankLib/RankLib.jar -load ../../model/AdaRank/ar_f3_r500_t002_m5.txt -test ../../data/Fold3/test.txt -norm linear -metric2t ERR@10

# Fold 4
java -jar ../../RankLib/RankLib.jar -load ../../model/AdaRank/ar_f4_r500_t002_m5.txt -test ../../data/Fold4/test.txt -norm linear -metric2t NDCG@10
java -jar ../../RankLib/RankLib.jar -load ../../model/AdaRank/ar_f4_r500_t002_m5.txt -test ../../data/Fold4/test.txt -norm linear -metric2t ERR@10

# Fold 5
java -jar ../../RankLib/RankLib.jar -load ../../model/AdaRank/ar_f5_r500_t002_m5.txt -test ../../data/Fold5/test.txt -norm linear -metric2t NDCG@10
java -jar ../../RankLib/RankLib.jar -load ../../model/AdaRank/ar_f5_r500_t002_m5.txt -test ../../data/Fold5/test.txt -norm linear -metric2t ERR@10