
from bayes_opt import BayesianOptimization
import os
import sys
import datetime



data = sys.argv[1]
ITERATIONS = sys.argv[2]
print("sparsity scale set to " + data)
print("Iterations set to " + ITERATIONS)


SPARSITY_SCALE = float(data)
current_layer_name = 'tomato'
prune_levels = {}

layer_names = []
iteration = 0

def write_final_schedule(starting_epoch):
    schedule = open('/home/omer/schedules/final.yaml','w')
    schedule.write("version: 1\n")
    schedule.write("pruners:\n")
    i = 0
    for name in layer_names:
        schedule.write("  layer"+str(i)+"_pruner:\n")
        schedule.write("    class: 'AutomatedGradualPruner'\n")
        #schedule.write("    initial_sparsity : " + str(prune_levels[name])+"\n")
        schedule.write("    initial_sparsity : 0.01\n")
        schedule.write("    final_sparsity : " + str(float(prune_levels[name])+0.0001)+"\n")
        schedule.write("    weights: ['" + name +"']\n")
        i = i + 1

    schedule.write("lr_schedulers:\n")
    schedule.write("  pruning_lr:\n")
    schedule.write("    class: StepLR\n")
    schedule.write("    step_size: 40\n")
    schedule.write("    gamma: 0.2\n")

    schedule.write("policies:\n")
    i = 0
    for name in layer_names:
        schedule.write("  - pruner:\n")
        schedule.write("      instance_name : 'layer"+str(i)+"_pruner'\n")
        schedule.write("    starting_epoch: "+str(starting_epoch)+"\n")
        schedule.write("    ending_epoch: "+str(starting_epoch+20)+"\n")
        schedule.write("    frequency: 1\n")
        i = i + 1

    schedule.write("  - lr_scheduler:\n")
    schedule.write("      instance_name: pruning_lr\n")
    schedule.write("    starting_epoch: 0\n")
    schedule.write("    ending_epoch: 500\n")
    schedule.write("    frequency: 1\n")

    schedule.close()


def write_schedule(layer_name,sparsity,epoch):
    schedule = open('/home/omer/schedules/temp.yaml','w')
    schedule.write("version: 1\n")
    schedule.write("pruners:\n")
    schedule.write("  layer_pruner:\n")
    schedule.write("    class: 'AutomatedGradualPruner'\n")
    schedule.write("    initial_sparsity : " + str(sparsity)+"\n")
    schedule.write("    final_sparsity : " + str(0.9999)+"\n")
    schedule.write("    weights: ['" + layer_name +"']\n")
    schedule.write("policies:\n")
    schedule.write("  - pruner:\n")
    schedule.write("      instance_name : 'layer_pruner'\n")
    schedule.write("    starting_epoch: "+str(epoch)+"\n")
    schedule.write("    ending_epoch: "+str(epoch+10)+"\n")
    schedule.write("    frequency: 1\n")

    schedule.close()


def read_top1():
    top1file = open('top1.txt','r')
    top1 = top1file.read()
    top1file.close()
    return float(top1)


def read_sparsity():
    sfile = open('sparsity.txt','r')
    sparsity = sfile.read()
    sfile.close()
    return float(sparsity)


def read_done():
    dfile = open('done.txt','r')
    done = dfile.read()
    dfile.close()
    if(int(done)==1):
        dfile = open('done.txt', 'w')
        dfile.write('0')
        dfile.close()
        return 1
    else:
        print('\nError executing distiller\n')
        return 0





def run_distiller():
    nbatches = 1
    mini_batch = 256
    training_dataset_size = 45000
    train_size = (mini_batch/training_dataset_size) * nbatches
    os.system('python3 compress_classifier.py -a simplenet_cifar ../../../data.cifar10 -p 30 -j=4 --lr=0.0000001 --out-dir logs/agp75/ --compress=/home/omer/schedules/temp.yaml --resume logs/baseline/2019.04.01-203750/best.pth.tar --epochs 1 --effective-train-size ' + str(train_size)+' >/dev/null 2>&1')
    #os.system('python3 compress_classifier.py -a simplenet_cifar ../../../data.cifar10 -p 30 -j=4 --lr=0.0000001 --out-dir logs/agp75/ --compress=/home/omer/schedules/temp.yaml --resume logs/baseline/2019.04.01-203750/best.pth.tar --epochs 1 --effective-train-size ' + str( train_size))
    top1 = read_top1()
    sparsity = read_sparsity()
    if((read_done()==1)):
        return top1,sparsity
    else:
        return run_distiller()


def black_box_function(prune_level):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    global current_layer_name
    global iteration
    iteration = iteration + 1

    write_schedule(current_layer_name,prune_level,97)
    top1, sparsity = run_distiller()

    print('Top1 :'+ '{:0.2f}'.format(top1) + '    Sparsity: '+'{:0.2f}'.format(sparsity))



    target = top1 + (sparsity/100)*SPARSITY_SCALE
    progressf.write('{0}, {1:0.3f}, {2:0.3f}, {3:0.3f}, {4:0.3f}\n'.format(iteration,
                                                                           target,
                                                                           prune_level,
                                                                           top1,
                                                                           sparsity))
    return target




dfile = open('done.txt', 'w')
dfile.write('0')
dfile.close()

now = datetime.datetime.now()
progressf = open('progress.txt',"a")
progressf.write('\n\n')
progressf.write(str(now))
progressf.write("\nsparsity scale set to " + data)
progressf.write('\n\n')
progressf.write(ITERATIONS)



layer_names.append('module.conv1.weight')
layer_names.append('module.conv2.weight')
layer_names.append('module.fc1.weight')
layer_names.append('module.fc2.weight')
layer_names.append('module.fc3.weight')

for name in layer_names:
    # Bounded region of parameter space
    pbounds = {'prune_level': (0.01, 0.99)}

    current_layer_name = name
    #progressf.write(name)
    #progressf.write('\n')

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.probe(
        params={"prune_level": 0.1},
        lazy=True,
    )

    optimizer.probe(
        params={"prune_level": 0.3},
        lazy=True,
    )

    optimizer.probe(
        params={"prune_level": 0.5},
        lazy=True,
    )

    optimizer.probe(
        params={"prune_level": 0.7},
        lazy=True,
    )

    optimizer.probe(
        params={"prune_level": 0.9},
        lazy=True,
    )

    iteration = 0
    optimizer.maximize(
        init_points=0,
        n_iter=int(ITERATIONS),
    )

    prune_levels[name] = str(optimizer.max['params']['prune_level'])



print(prune_levels)

write_final_schedule(97)

os.system('python3 compress_classifier.py -a simplenet_cifar ../../../data.cifar10 -p 30 -j=4 --lr=0.002 --out-dir logs/agp75/ --compress=/home/omer/schedules/final.yaml --resume logs/baseline/2019.04.01-203750/best.pth.tar --epochs 80 ')


final_top1 = read_top1()
final_sparsity = read_sparsity()

progressf.write('\n Final Sparsity: {0:0.2f}    Final Top1: {1:0.3f}\n'.format(final_sparsity,final_top1))

progressf.close()


print('done')
#
# print(optimizer.max)

# layer_names = []
# layer_names.append('module.conv1.weight')
# layer_names.append('module.conv2.weight')
# layer_names.append('module.fc1.weight')
# layer_names.append('module.fc2.weight')
# layer_names.append('module.fc3.weight')
#
# for name in layer_names:
#     write_schedule(name,0.5,97)

#os.system('python3 compress_classifier.py -a simplenet_cifar ../../../data.cifar10 -p 30 -j=4 --lr=0.000001 --out-dir logs/agp75/ --compress=/home/omer/schedules/agp75.yaml --resume logs/baseline/2019.04.01-203750/best.pth.tar --epochs 1 >/dev/null 2>&1')
# top1 = read_top1()
# sparsity = read_sparsity()
# print('Top 1 accuracy ' + str(top1))
# print('Sparsity ' + str(sparsity))
