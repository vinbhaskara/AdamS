# Runs for LR models on MNIST. 
# Batch Size 128, No Dropout

# Original Adam
python main.py -a mnistlr -d mnist --drop 0 --opt adam --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-lr-nodropout-seed2
python main.py -a mnistlr -d mnist --drop 0 --opt adam --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-lr-nodropout
python main.py -a mnistlr -d mnist --drop 0 --opt adam --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-lr-nodropout-seed1

# AdamUCB - 0.01

python main.py -a mnistlr -d mnist --drop 0 --opt adamucb --eta1 1.0 --eta2 0.01 --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-lr-nodropout-seed2
python main.py -a mnistlr -d mnist --drop 0 --opt adamucb --eta1 1.0 --eta2 0.01 --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-lr-nodropout
python main.py -a mnistlr -d mnist --drop 0 --opt adamucb --eta1 1.0 --eta2 0.01 --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-lr-nodropout-seed1


# AdamCB - 0.001

python main.py -a mnistlr -d mnist --drop 0 --opt adamcb --eta1 1.0 --eta2 0.001 --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-lr-nodropout-seed2
python main.py -a mnistlr -d mnist --drop 0 --opt adamcb --eta1 1.0 --eta2 0.001 --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-lr-nodropout
python main.py -a mnistlr -d mnist --drop 0 --opt adamcb --eta1 1.0 --eta2 0.001 --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-lr-nodropout-seed1



# AdamS - 0.0001
python main.py -a mnistlr -d mnist --drop 0 --opt adams --eta1 1.0 --eta2 0.0001 --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-lr-nodropout-seed2
python main.py -a mnistlr -d mnist --drop 0 --opt adams --eta1 1.0 --eta2 0.0001 --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-lr-nodropout
python main.py -a mnistlr -d mnist --drop 0 --opt adams --eta1 1.0 --eta2 0.0001 --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-lr-nodropout-seed1

