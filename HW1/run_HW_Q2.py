import subprocess
import time

ti = time.monotonic()
for n_layers in [1,2,3]:
    # set default values 
    lr = 0.01
    hs = 100
    drop = 0.3
    act = "relu"


    for lr in [0.001, 0.01, 0.1]:
        subprocess.run(f"python hw1-q2.py mlp -batch_size 16 -epochs 20 -learning_rate {lr} -hidden_sizes {hs} -dropout {drop} -activation {act} -layers {n_layers}".split(" "))

    # set default values 
    lr = 0.01
    hs = 100
    drop = 0.3
    act = "relu"

    for hs in [100,200]:
        subprocess.run(f"python hw1-q2.py mlp -batch_size 16 -epochs 20 -learning_rate {lr} -hidden_sizes {hs} -dropout {drop} -activation {act} -layers {n_layers}".split(" "))




    # set default values 
    lr = 0.01
    hs = 100
    drop = 0.3
    act = "relu"

    for drop in [0.3,0.5]:
        subprocess.run(f"python hw1-q2.py mlp -batch_size 16 -epochs 20 -learning_rate {lr} -hidden_sizes {hs} -dropout {drop} -activation {act} -layers {n_layers}".split(" "))




    # set default values 
    lr = 0.01
    hs = 100
    drop = 0.3
    act = "relu"

    for act in ["relu", "tanh"]:
        subprocess.run(f"python hw1-q2.py mlp -batch_size 16 -epochs 20 -learning_rate {lr} -hidden_sizes {hs} -dropout {drop} -activation {act} -layers {n_layers}".split(" "))

tf = time.monotonic()
delta = tf-ti
print("WHOLE TRAINING TOOK: ", delta)