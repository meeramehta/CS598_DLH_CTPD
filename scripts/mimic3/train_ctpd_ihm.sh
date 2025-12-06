python train_mimic3.py --task ihm --model_name ctpd --devices 1 --accelerator cpu \
   --pooling_type attention \
   --use_prototype \
   --use_multiscale --lamb1 0.1 --lamb2 0.5 --lamb3 0.5
