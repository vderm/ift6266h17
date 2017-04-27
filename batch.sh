#!/usr/bin/env bash
# First batch
# python dcgan_doc2vec_embed.py --drawImageEveryNEpochs 15 --sEncd 512 --sVector 128 --epochs 30 --endTrainExample 5120
# python dcgan_doc2vec_embed.py --drawImageEveryNEpochs 15 --sEncd 512 --sVector 128 --epochs 120 --endTrainExample 5120
# python dcgan_doc2vec_embed.py --drawImageEveryNEpochs 15 --sEncd 512 --sVector 128 --epochs 30 --endTrainExample 5120 --addExtraGLayer True
# python dcgan_doc2vec_embed.py --drawImageEveryNEpochs 15 --sEncd 512 --sVector 128 --epochs 120 --endTrainExample 5120 --addExtraGLayer True
# python dcgan_doc2vec_embed.py --drawImageEveryNEpochs 15 --sEncd 512 --sVector 512 --epochs 30 --endTrainExample 5120
# python dcgan_doc2vec_embed.py --drawImageEveryNEpochs 15 --sEncd 512 --sVector 512 --epochs 120 --endTrainExample 5120
# python dcgan_doc2vec_embed.py --drawImageEveryNEpochs 15 --sEncd 512 --sVector 512 --epochs 30 --endTrainExample 5120 --addExtraGLayer True
# python dcgan_doc2vec_embed.py --drawImageEveryNEpochs 15 --sEncd 512 --sVector 512 --epochs 120 --endTrainExample 5120 --addExtraGLayer True
# python dcgan_doc2vec_embed.py --drawImageEveryNEpochs 15 --sEncd 2048 --sVector 128 --epochs 30 --endTrainExample 5120
# python dcgan_doc2vec_embed.py --drawImageEveryNEpochs 15 --sEncd 2048 --sVector 128 --epochs 120 --endTrainExample 5120
# python dcgan_doc2vec_embed.py --drawImageEveryNEpochs 15 --sEncd 2048 --sVector 128 --epochs 30 --endTrainExample 5120 --addExtraGLayer True
# python dcgan_doc2vec_embed.py --drawImageEveryNEpochs 15 --sEncd 2048 --sVector 128 --epochs 120 --endTrainExample 5120 --addExtraGLayer True
# python dcgan_doc2vec_embed.py --drawImageEveryNEpochs 15 --sEncd 2048 --sVector 512 --epochs 30 --endTrainExample 5120
# python dcgan_doc2vec_embed.py --drawImageEveryNEpochs 15 --sEncd 2048 --sVector 512 --epochs 120 --endTrainExample 5120
# python dcgan_doc2vec_embed.py --drawImageEveryNEpochs 15 --sEncd 2048 --sVector 512 --epochs 30 --endTrainExample 5120 --addExtraGLayer True
# python dcgan_doc2vec_embed.py --drawImageEveryNEpochs 15 --sEncd 2048 --sVector 512 --epochs 120 --endTrainExample 5120 --addExtraGLayer True

# Second batch
python dcgan_doc2vec_embed.py --drawImageEveryNEpochs 15 --sEncd 2048 --sVector 128 --epochs 120 --trainDoc2Vec True --addExtraGLayer True
python dcgan_doc2vec_embed.py --drawImageEveryNEpochs 15 --sEncd 2048 --sVector 128 --epochs 120 --genLosssW 0.1 --addExtraGLayer True
