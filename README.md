# DCPHA
A test implementation for the paper "Deep Consistency-Preserving Hash Auto-encoders for Neuroimage Cross-Modal Retrieval" (Under reviewing)

# Environment:

python        3.6

pytorch       1.8.1+cu101

torchvision   0.9.1+cu101

numpy         1.18.5

# Steps

1. Download pre-trained models from [there](https://pan.baidu.com/s/1h8NNO9GKD1R1EDtC_Ty2Kw) password: mxss

2. Download partial test samples from [there](https://pan.baidu.com/s/1SbiEetuQUcZz5pvHwIn0MQ) password: data

3. Remove the pre-trained model (such as DCPHA_ADNI2_16bits) into corresponding dir. e.g.: /DCPHA/pretrain_models/DCPHA_ADNI2_16bits.pth

4. Remove the test set (such as ADNI2) into corresponding dir. e.g.: /DCPHA/data/ADNI2/test_MRI.mat

6. Testing PET or MRI retrieval performance from MRI or PET in proposed DCPHA

7. python valid_model.py
