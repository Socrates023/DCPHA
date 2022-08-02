# DCPHA
A test implementation for the paper "Deep Consistency-Preserving Hash Auto-encoders for Neuroimage Cross-Modal Retrieval" (Under reviewing)

# Environment:
python        3.6
pytorch       1.8.1+cu101
torchvision   0.9.1+cu101
numpy         1.18.5

# Steps

1. Download pre-trained models from [there]() password: .

2. Download partial test samples from [there]() password: .

3. Remove the pre-trained model (such as DCPHA_ADNI2_16bits) into corresponding dir. e.g.: /DCPHA/pretrain_models/DCPHA_ADNI2_16bits.pth

4. Remove the test set (such as ADNI2) into corresponding dir. e.g.: /DCPHA/data/ADNI2/test_MRI.mat

6. Testing PET or MRI retrieval performance from MRI or PET in proposed DCPHA

7. python valid_model.py
