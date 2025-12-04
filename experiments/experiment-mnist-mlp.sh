# Runs for MLP models on MNIST with Batch Size {128, 16} and {With, Without} Dropout.  

# Batch Size 128, No Dropout
# Adam
python main.py -a mnistmlp -d mnist --drop 0 --opt adam --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout-seed2
python main.py -a mnistmlp -d mnist --drop 0 --opt adam --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout
python main.py -a mnistmlp -d mnist --drop 0 --opt adam --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout-seed1

# AdamUCB

python main.py -a mnistmlp -d mnist --drop 0 --opt adamucb --eta1 1.0 --eta2 0.1 --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout-seed2
python main.py -a mnistmlp -d mnist --drop 0 --opt adamucb --eta1 1.0 --eta2 0.1 --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout
python main.py -a mnistmlp -d mnist --drop 0 --opt adamucb --eta1 1.0 --eta2 0.1 --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout-seed1

# AdamCB

python main.py -a mnistmlp -d mnist --drop 0 --opt adamcb --eta1 1.0 --eta2 0.001 --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout-seed2
python main.py -a mnistmlp -d mnist --drop 0 --opt adamcb --eta1 1.0 --eta2 0.001 --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout
python main.py -a mnistmlp -d mnist --drop 0 --opt adamcb --eta1 1.0 --eta2 0.001 --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout-seed1


# AdamS

python main.py -a mnistmlp -d mnist --drop 0 --opt adams --eta1 1.0 --eta2 0.005 --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout-seed2
python main.py -a mnistmlp -d mnist --drop 0 --opt adams --eta1 1.0 --eta2 0.005 --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout
python main.py -a mnistmlp -d mnist --drop 0 --opt adams --eta1 1.0 --eta2 0.005 --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout-seed1



# Batch Size 16, No Dropout

# Adam
python main.py -a mnistmlp -d mnist --drop 0 --opt adam --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout-seed2
python main.py -a mnistmlp -d mnist --drop 0 --opt adam --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout
python main.py -a mnistmlp -d mnist --drop 0 --opt adam --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout-seed1

# AdamUCB


python main.py -a mnistmlp -d mnist --drop 0 --opt adamucb --eta1 1.0 --eta2 0.3 --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout-seed2
python main.py -a mnistmlp -d mnist --drop 0 --opt adamucb --eta1 1.0 --eta2 0.3 --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout
python main.py -a mnistmlp -d mnist --drop 0 --opt adamucb --eta1 1.0 --eta2 0.3 --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout-seed1


# AdamCB
python main.py -a mnistmlp -d mnist --drop 0 --opt adamcb --eta1 1.0 --eta2 0.0001 --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout-seed2
python main.py -a mnistmlp -d mnist --drop 0 --opt adamcb --eta1 1.0 --eta2 0.0001 --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout
python main.py -a mnistmlp -d mnist --drop 0 --opt adamcb --eta1 1.0 --eta2 0.0001 --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout-seed1

# AdamS


python main.py -a mnistmlp -d mnist --drop 0 --opt adams --eta1 1.0 --eta2 0.05 --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout-seed2
python main.py -a mnistmlp -d mnist --drop 0 --opt adams --eta1 1.0 --eta2 0.05 --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout
python main.py -a mnistmlp -d mnist --drop 0 --opt adams --eta1 1.0 --eta2 0.05 --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 16 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-nodropout-seed1




# Batch 128 Size, With Dropout
# Adam
python main.py -a mnistmlp -d mnist --drop 1 --opt adam --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-withdropout-seed2
python main.py -a mnistmlp -d mnist --drop 1 --opt adam --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-withdropout
python main.py -a mnistmlp -d mnist --drop 1 --opt adam --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-withdropout-seed1

# AdamUCB

python main.py -a mnistmlp -d mnist --drop 1 --opt adamucb --eta1 1.0 --eta2 0.1 --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-withdropout-seed2
python main.py -a mnistmlp -d mnist --drop 1 --opt adamucb --eta1 1.0 --eta2 0.1 --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-withdropout
python main.py -a mnistmlp -d mnist --drop 1 --opt adamucb --eta1 1.0 --eta2 0.1 --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-withdropout-seed1


# AdamCB
python main.py -a mnistmlp -d mnist --drop 1 --opt adamcb --eta1 1.0 --eta2 0.0005 --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-withdropout-seed2
python main.py -a mnistmlp -d mnist --drop 1 --opt adamcb --eta1 1.0 --eta2 0.0005 --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-withdropout
python main.py -a mnistmlp -d mnist --drop 1 --opt adamcb --eta1 1.0 --eta2 0.0005 --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-withdropout-seed1

# AdamS

python main.py -a mnistmlp -d mnist --drop 1 --opt adams --eta1 1.0 --eta2 0.005 --manualSeed 06121947 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-withdropout-seed2
python main.py -a mnistmlp -d mnist --drop 1 --opt adams --eta1 1.0 --eta2 0.005 --manualSeed 24122001 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-withdropout
python main.py -a mnistmlp -d mnist --drop 1 --opt adams --eta1 1.0 --eta2 0.005 --manualSeed 28081994 --valevery 0 --learning-rate 0.001 --train-batch 128 --test-batch 128 --epochs 45 --updates -1 --gamma 0.0 --wd 1e-4 --checkpoint checkpoints/mnist/mnist-mlp-withdropout-seed1
