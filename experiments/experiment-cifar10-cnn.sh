# Runs for CNN models on CIFAR-10 with Batch Size {128, 16} and {With, Without} Dropout.  

# Batch Size 128, No Dropout
# Original Adam
python main.py -a cifar10cnn --drop 0 --opt adam --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout-seed2
python main.py -a cifar10cnn --drop 0 --opt adam --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout
python main.py -a cifar10cnn --drop 0 --opt adam --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout-seed1

# AdamUCB - 0.01

python main.py -a cifar10cnn --drop 0 --opt adamucb --eta1 1.0 --eta2 0.01 --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout-seed2
python main.py -a cifar10cnn --drop 0 --opt adamucb --eta1 1.0 --eta2 0.01 --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout
python main.py -a cifar10cnn --drop 0 --opt adamucb --eta1 1.0 --eta2 0.01 --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout-seed1


# AdamCB - 5e-05

python main.py -a cifar10cnn --drop 0 --opt adamcb --eta1 1.0 --eta2 5e-05 --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout-seed2
python main.py -a cifar10cnn --drop 0 --opt adamcb --eta1 1.0 --eta2 5e-05 --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout
python main.py -a cifar10cnn --drop 0 --opt adamcb --eta1 1.0 --eta2 5e-05 --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout-seed1

# AdamS - 0.0001
python main.py -a cifar10cnn --drop 0 --opt adams --eta1 1.0 --eta2 0.0001 --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout-seed2
python main.py -a cifar10cnn --drop 0 --opt adams --eta1 1.0 --eta2 0.0001 --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout
python main.py -a cifar10cnn --drop 0 --opt adams --eta1 1.0 --eta2 0.0001 --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout-seed1


# Batch Size 16, No Dropout
# Original Adam
python main.py -a cifar10cnn --drop 0 --opt adam --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout-seed2
python main.py -a cifar10cnn --drop 0 --opt adam --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout
python main.py -a cifar10cnn --drop 0 --opt adam --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout-seed1

# AdamUCB - 0.3

python main.py -a cifar10cnn --drop 0 --opt adamucb --eta1 1.0 --eta2 0.3 --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout-seed2
python main.py -a cifar10cnn --drop 0 --opt adamucb --eta1 1.0 --eta2 0.3 --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout
python main.py -a cifar10cnn --drop 0 --opt adamucb --eta1 1.0 --eta2 0.3 --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout-seed1


# AdamCB - 1e-05

python main.py -a cifar10cnn --drop 0 --opt adamcb --eta1 1.0 --eta2 1e-05 --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout-seed2
python main.py -a cifar10cnn --drop 0 --opt adamcb --eta1 1.0 --eta2 1e-05 --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout
python main.py -a cifar10cnn --drop 0 --opt adamcb --eta1 1.0 --eta2 1e-05 --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout-seed1


# AdamS - 0.005

python main.py -a cifar10cnn --drop 0 --opt adams --eta1 1.0 --eta2 0.005 --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout-seed2
python main.py -a cifar10cnn --drop 0 --opt adams --eta1 1.0 --eta2 0.005 --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout
python main.py -a cifar10cnn --drop 0 --opt adams --eta1 1.0 --eta2 0.005 --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-nodropout-seed1


# Batch Size 128, With Dropout

# Original Adam
python main.py -a cifar10cnn --drop 1 --opt adam --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-withdropout-seed2
python main.py -a cifar10cnn --drop 1 --opt adam --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-withdropout
python main.py -a cifar10cnn --drop 1 --opt adam --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-withdropout-seed1

# AdamUCB - 0.05

python main.py -a cifar10cnn --drop 1 --opt adamucb --eta1 1.0 --eta2 0.05 --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-withdropout-seed2
python main.py -a cifar10cnn --drop 1 --opt adamucb --eta1 1.0 --eta2 0.05 --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-withdropout
python main.py -a cifar10cnn --drop 1 --opt adamucb --eta1 1.0 --eta2 0.05 --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-withdropout-seed1


# AdamCB - 0.0001
python main.py -a cifar10cnn --drop 1 --opt adamcb --eta1 1.0 --eta2 0.0001 --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-withdropout-seed2
python main.py -a cifar10cnn --drop 1 --opt adamcb --eta1 1.0 --eta2 0.0001 --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-withdropout
python main.py -a cifar10cnn --drop 1 --opt adamcb --eta1 1.0 --eta2 0.0001 --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-withdropout-seed1


# AdamS - 0.0001
python main.py -a cifar10cnn --drop 1 --opt adams --eta1 1.0 --eta2 0.0001 --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-withdropout-seed2
python main.py -a cifar10cnn --drop 1 --opt adams --eta1 1.0 --eta2 0.0001 --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-withdropout
python main.py -a cifar10cnn --drop 1 --opt adams --eta1 1.0 --eta2 0.0001 --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/cifar10/cifar10-cnn-withdropout-seed1
