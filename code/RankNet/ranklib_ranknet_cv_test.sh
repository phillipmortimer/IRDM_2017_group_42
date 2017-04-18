# Fold 1
java -jar ../../RankLib/RankLib.jar -load ../../model/RankNet/rn_f1_e9_l1_n20_lr-5.txt -test ../../data/Fold1/test.txt -norm linear -metric2t NDCG@10
java -jar ../../RankLib/RankLib.jar -load ../../model/RankNet/rn_f1_e9_l1_n20_lr-5.txt -test ../../data/Fold1/test.txt -norm linear -metric2t ERR@10

# Fold 2
java -jar ../../RankLib/RankLib.jar -load ../../model/RankNet/rn_f2_e9_l1_n20_lr-5.txt -test ../../data/Fold2/test.txt -norm linear -metric2t NDCG@10
java -jar ../../RankLib/RankLib.jar -load ../../model/RankNet/rn_f2_e9_l1_n20_lr-5.txt -test ../../data/Fold2/test.txt -norm linear -metric2t ERR@10

# Fold 3
java -jar ../../RankLib/RankLib.jar -load ../../model/RankNet/rn_f3_e9_l1_n20_lr-5.txt -test ../../data/Fold3/test.txt -norm linear -metric2t NDCG@10
java -jar ../../RankLib/RankLib.jar -load ../../model/RankNet/rn_f3_e9_l1_n20_lr-5.txt -test ../../data/Fold3/test.txt -norm linear -metric2t ERR@10

# Fold 4
java -jar ../../RankLib/RankLib.jar -load ../../model/RankNet/rn_f4_e9_l1_n20_lr-5.txt -test ../../data/Fold4/test.txt -norm linear -metric2t NDCG@10
java -jar ../../RankLib/RankLib.jar -load ../../model/RankNet/rn_f4_e9_l1_n20_lr-5.txt -test ../../data/Fold4/test.txt -norm linear -metric2t ERR@10

# Fold 5
java -jar ../../RankLib/RankLib.jar -load ../../model/RankNet/rn_f5_e9_l1_n20_lr-5.txt -test ../../data/Fold5/test.txt -norm linear -metric2t NDCG@10
java -jar ../../RankLib/RankLib.jar -load ../../model/RankNet/rn_f5_e9_l1_n20_lr-5.txt -test ../../data/Fold5/test.txt -norm linear -metric2t ERR@10