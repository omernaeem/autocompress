from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import os
import sys
import datetime
import time

data = sys.argv[1]
ITERATIONS = sys.argv[2]
print("sparsity scale set to " + data)
print("Iterations set to " + ITERATIONS)

DATASET_PATH = "../../../data.cifar10"
DATASET_SIZE = 45000
DATASET_VALID_SIZE = 5000
EFFECTIVE_TEST_SIZE = 1
EFFECTIVE_TEST_SIZE_PRUNE = 0.001

QUANTIZE_ACTIVATIONS = 1

SUPPRESS_OUTPUT = ' >/dev/null 2>&1'
#SUPPRESS_OUTPUT = ' '
# DATASET_SIZE = 4500
# DATASET_PATH = "../../../data.cifar10"
SPARSITY_SCALE = float(data)
STARTING_EPOCH = 179
PRUNE_EPOCHS = 3
GRADUAL_PRUNE_EPOCHS = PRUNE_EPOCHS - 1
QUANT_EPOCHS = 40
layer_biases = [16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 10]

current_layer_name = 'tomato'
network_size = 0
prune_levels = {}
bitwidths = {}

layer_names = []
layer_sizes = []
layer_sparsities = []
layer_zeros_storage_reqs = []
layer_c = [0] * 22
iteration = 0




def write_final_schedule_quant(epoch):
    schedule = open('schedules/final_quant.yaml', 'w')
    schedule.write("version: 1\n")
    schedule.write("quantizers:\n")
    schedule.write("  linear_quantizer:\n")
    schedule.write("    class: 'QuantAwareTrainRangeLinearQuantizer'\n")
    schedule.write("    bits_activations : 32\n")
    schedule.write("    bits_weights : 32\n")
    schedule.write("    mode: 'ASYMMETRIC_UNSIGNED'\n")
    schedule.write("    ema_decay: 0.999\n")
    schedule.write("    per_channel_wts: True\n")
    schedule.write("    bits_overrides:\n")
    for name in layer_names:
        layer_name = name
        layer_name = layer_name.replace('module.', '')
        layer_name = layer_name.replace('.weight', '')
        schedule.write("        " + layer_name + ":\n")
        schedule.write("             wts: " + str(round(float(bitwidths[name]))) + " \n")
        if QUANTIZE_ACTIVATIONS == 1:
            if str(layer_name).__contains__('conv'):
                activation_name = layer_name.replace('conv', 'relu')
                schedule.write("        " + activation_name + ":\n")
                schedule.write("             acts: " + str(round(float(bitwidths[name]))) + " \n")

    schedule.write("policies:\n")
    schedule.write("  - quantizer:\n")
    schedule.write("      instance_name : 'linear_quantizer'\n")
    schedule.write("    starting_epoch: " + str(epoch) + "\n")
    schedule.write("    ending_epoch: " + str(epoch + 500) + "\n")
    schedule.write("    frequency: 1\n")
    schedule.close()


def write_final_schedule(starting_epoch):
    schedule = open('schedules/final.yaml', 'w')
    schedule.write("version: 1\n")
    schedule.write("pruners:\n")
    i = 0
    for name in layer_names:
        schedule.write("  layer" + str(i) + "_pruner:\n")
        schedule.write("    class: 'AutomatedGradualPruner'\n")
        # schedule.write("    initial_sparsity : " + str(prune_levels[name])+"\n")
        schedule.write("    initial_sparsity : 0.01\n")
        schedule.write("    final_sparsity : " + str(float(prune_levels[name]) + 0.0001) + "\n")
        schedule.write("    weights: ['" + name + "']\n")
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
        schedule.write("      instance_name : 'layer" + str(i) + "_pruner'\n")
        schedule.write("    starting_epoch: " + str(starting_epoch) + "\n")
        schedule.write("    ending_epoch: " + str(starting_epoch + GRADUAL_PRUNE_EPOCHS) + "\n")
        schedule.write("    frequency: 1\n")
        i = i + 1

    schedule.write("  - lr_scheduler:\n")
    schedule.write("      instance_name: pruning_lr\n")
    schedule.write("    starting_epoch: 0\n")
    schedule.write("    ending_epoch: 500\n")
    schedule.write("    frequency: 1\n")

    schedule.close()


def write_schedule(layer_name, sparsity, epoch):
    schedule = open('schedules/temp.yaml', 'w')
    schedule.write("version: 1\n")
    schedule.write("pruners:\n")
    schedule.write("  layer_pruner:\n")
    schedule.write("    class: 'AutomatedGradualPruner'\n")
    schedule.write("    initial_sparsity : " + str(sparsity) + "\n")
    schedule.write("    final_sparsity : " + str(0.9999) + "\n")
    schedule.write("    weights: ['" + layer_name + "']\n")
    schedule.write("policies:\n")
    schedule.write("  - pruner:\n")
    schedule.write("      instance_name : 'layer_pruner'\n")
    schedule.write("    starting_epoch: " + str(epoch) + "\n")
    schedule.write("    ending_epoch: " + str(epoch + 1000) + "\n")
    schedule.write("    frequency: 1\n")

    schedule.close()


def write_schedule_quant(layer_name, bitwidth, epoch):
    #layer_name = layer_name[layer_name.find(".") + 1:layer_name.find(".w")]  # from . to .weight
    layer_name = layer_name.replace('module.','')
    layer_name = layer_name.replace('.weight','')
    schedule = open('schedules/temp.yaml', 'w')
    schedule.write("version: 1\n")
    schedule.write("quantizers:\n")
    schedule.write("  linear_quantizer:\n")
    schedule.write("    class: 'QuantAwareTrainRangeLinearQuantizer'\n")
    schedule.write("    bits_activations : 32\n")
    schedule.write("    bits_weights : 32\n")
    schedule.write("    mode: 'ASYMMETRIC_UNSIGNED'\n")
    schedule.write("    ema_decay: 0.999\n")
    schedule.write("    per_channel_wts: True\n")
    schedule.write("    bits_overrides:\n")
    schedule.write("        " + layer_name + ":\n")
    schedule.write("             wts: " + str(bitwidth) + " \n")
    if QUANTIZE_ACTIVATIONS == 1:
        if str(layer_name).__contains__('conv'):
            activation_name = layer_name.replace('conv', 'relu')
            schedule.write("        " + activation_name + ":\n")
            schedule.write("             acts: " + str(bitwidth) + " \n")
    schedule.write("policies:\n")
    schedule.write("  - quantizer:\n")
    schedule.write("      instance_name : 'linear_quantizer'\n")
    schedule.write("    starting_epoch: " + str(epoch) + "\n")
    schedule.write("    ending_epoch: " + str(epoch + 10) + "\n")
    schedule.write("    frequency: 1\n")
    schedule.close()


def read_checkpoint_path():
    f = open('checkpoint_path.txt', 'r')
    checkpoint_path = f.read()
    f.close()
    return checkpoint_path


def read_top1():
    top1file = open('top1.txt', 'r')
    top1 = top1file.read()
    top1file.close()
    return float(top1)


def read_sparsity():
    sfile = open('sparsity.txt', 'r')
    sparsity = sfile.read()
    sfile.close()
    return float(sparsity)


def read_done():
    dfile = open('done.txt', 'r')
    done = dfile.read()
    dfile.close()
    if (int(done) == 1):
        dfile = open('done.txt', 'w')
        dfile.write('0')
        dfile.close()
        return 1
    else:
        print('\nError executing distiller\n')
        return 0


def read_model_details():
    f = open('model_details.txt', 'r')
    global network_size
    network_size = 0
    for line in f:
        data = line.split(',')
        layer_names.append(data[0])
        layer_sizes.append(data[1])
        layer_sparsities.append(data[2])
        network_size = network_size + int(data[1])
        layer_zeros_storage_reqs.append(0)

    f.close()


def read_sparsities():
    f = open('model_details.txt', 'r')
    i = 0
    for line in f:
        data = line.split(',')
        layer_sparsities[i] = (data[2])
        layer_zeros_storage_reqs[i] = [float(data[3]),
                                       float(data[4]),
                                       float(data[5]),
                                       float(data[6]),
                                       float(data[7]),
                                       float(data[8]),
                                       float(data[9]),
                                       float(data[10])]
        i = i + 1

    f.close()


def run_distiller_summary():
    os.system(
        "python3 compress_classifier.py -a resnet20_cifar " + DATASET_PATH + " -p 30 -j=4 --lr=0.0000001 --out-dir logs/agp75/  --epochs 1 --resume ../ssl/checkpoints/checkpoint_trained_dense.pth.tar --summary sparsity")


def run_distiller_quant():
    nbatches = 1
    mini_batch = 256
    training_dataset_size = DATASET_SIZE
    train_size = (mini_batch / training_dataset_size) * nbatches
    valid_size = (mini_batch / DATASET_VALID_SIZE) * nbatches
    checkpoint_path = read_checkpoint_path()
    os.system(
        "python3 compress_classifier.py -a resnet20_cifar " + DATASET_PATH + " -p 30 -j=4 --lr=0.0000001 --out-dir logs/agp75/ --compress=schedules/temp.yaml --resume " + checkpoint_path + "/checkpoint.pth.tar --epochs 1 --effective-train-size " + str(
            train_size) + " --effective-valid-size " + str(valid_size) + " --effective-test-size " + str(
            EFFECTIVE_TEST_SIZE) + " --gpus 0 " + SUPPRESS_OUTPUT)
    top1 = read_top1()
    sparsity = read_sparsity()
    if ((read_done() == 1)):
        return top1, sparsity
    else:
        time.sleep(15)
        return run_distiller_quant()


def run_distiller():
    nbatches = 1
    mini_batch = 256
    training_dataset_size = DATASET_SIZE
    train_size = (mini_batch / training_dataset_size) * nbatches
    valid_size = (mini_batch / DATASET_VALID_SIZE) * nbatches
    os.system("rm -rf logs/agp75")
    os.system(
        "python3 compress_classifier.py -a resnet20_cifar " + DATASET_PATH + " -p 30 -j=4 --lr=0.0000001 --out-dir logs/agp75/ --compress=schedules/temp.yaml --resume ../ssl/checkpoints/checkpoint_trained_dense.pth.tar --epochs 1 --effective-train-size " + str(
            train_size) + " --effective-valid-size " + str(valid_size) + " --effective-test-size " + str(
            EFFECTIVE_TEST_SIZE_PRUNE) + SUPPRESS_OUTPUT)
    top1 = read_top1()
    sparsity = read_sparsity()
    if ((read_done() == 1)):
        return top1, sparsity
    else:
        time.sleep(15)
        return run_distiller()


def run_distiller_final():
    os.system(
        "python3 compress_classifier.py -a resnet20_cifar "+ DATASET_PATH + " -p 30 -j=4 --lr=0.001 --out-dir logs/agp75/ --compress=schedules/final.yaml --resume ../ssl/checkpoints/checkpoint_trained_dense.pth.tar --epochs " + str(
            PRUNE_EPOCHS))


def run_distiller_final_quant():
    checkpoint_path = read_checkpoint_path()
    os.system(
        "python3 compress_classifier.py -a resnet20_cifar " + DATASET_PATH + " -p 30 -j=4 --lr=0.0008 --out-dir logs/agp75/ --compress=schedules/final_quant.yaml --resume " + checkpoint_path + "/checkpoint.pth.tar --epochs " + str(
            QUANT_EPOCHS) + " --gpus 0 --validation-split 0")


def find_storage_req(li, b):
    """Returns bits required to store in sparse representation as in Deep Compression
    """
    req=0
    best_req = 999999999999999
    c = 1
    nnz = (1 - (float(layer_sparsities[li])/100))*float(layer_sizes[li])
    for zs in layer_zeros_storage_reqs[li]:
        req = (zs + nnz) * (c + b)
        if (req < best_req):
            best_req = req
            layer_c[li] = c
        c = c + 1

    return best_req + (layer_biases[li]*32)


def black_box_function(prune_level, bitwidth):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    global current_layer_name
    global iteration
    iteration = iteration + 1

    bitwidthi = round(bitwidth)

    write_schedule(current_layer_name, prune_level, STARTING_EPOCH)
    top1_prune, total_sparsity = run_distiller()

    read_sparsities()
    li = layer_names.index(current_layer_name)
    storage_req = find_storage_req(li,bitwidthi)

    write_schedule_quant(current_layer_name, bitwidthi, STARTING_EPOCH)
    top1, total_sparsity = run_distiller_quant()


    sparsity = float(layer_sparsities[li])

    total_network_size_bits = (network_size*32) + sum(layer_biases)*32
    uncomp_layer_size_bits = (float(layer_sizes[li])*32) + layer_biases[li]*32

    ci = (uncomp_layer_size_bits - storage_req)/total_network_size_bits
    top1_error = 100 - top1

    target = -1*(top1_error - (SPARSITY_SCALE*ci))

    print('Top1_prune :' + '{:0.2f}'.format(top1_prune) + '    Sparsity: ' + '{:0.2f}'.format(
        sparsity) + '    Top1: ' + '{:0.2f}'.format(top1))

    c = layer_c[li]
    compression = uncomp_layer_size_bits/storage_req
    # bit_size_reduction = 32.0 / float(bitwidthi)
    # compression_due_to_sparsity = 1.0 / (1.0 - (sparsity / 100))
    # compression = bit_size_reduction * compression_due_to_sparsity
    # layer_ratio = float(layer_sizes[li]) / network_size
    # target = top1 + compression * layer_ratio * SPARSITY_SCALE
    progressf.write('{0}, {1:0.3f}, {2:0.3f}, {3:0.3f}, {4:0.3f}, {5:0.3f}, {6:0.3f}, {7}, {8}, {9}, {10}\n'.format(iteration,
                                                                                                                    target,
                                                                                                                    prune_level,
                                                                                                                    top1_prune,
                                                                                                                    top1,
                                                                                                                    sparsity,
                                                                                                                    bitwidth,
                                                                                                                    bitwidthi,
                                                                                                                    storage_req,
                                                                                                                    c,
                                                                                                                    compression))

    print('{0}, {1:0.3f}, {2:0.3f}, {3:0.3f}, {4:0.3f}, {5:0.3f}, {6:0.3f}, {7}, {8}, {9}, {10}'.format(iteration,
                                                                                                          target,
                                                                                                          prune_level,
                                                                                                          top1_prune,
                                                                                                          top1,
                                                                                                          sparsity,
                                                                                                          bitwidth,
                                                                                                          bitwidthi,
                                                                                                          storage_req,
                                                                                                          c,
                                                                                                          compression))
    return target


dfile = open('done.txt', 'w')
dfile.write('0')
dfile.close()

now = datetime.datetime.now()
progressf = open('progress.txt', "a")
progressf.write('\n\n')
progressf.write(str(now))
progressf.write("\nCompression scale set to " + data)
progressf.write('\n\n')
progressf.write(ITERATIONS)

run_distiller_summary()
read_model_details()

# layer_names.append('module.conv1.weight')
# layer_names.append('module.conv2.weight')
# layer_names.append('module.fc1.weight')
# layer_names.append('module.fc2.weight')
# layer_names.append('module.fc3.weight')

lidx_bo = -1

for name in layer_names:
    # Bounded region of parameter space

    iteration = 0
    pbounds = {'prune_level': (0.01, 0.99), 'bitwidth': (2, 8)}

    # lidx_bo = lidx_bo + 1
    # if lidx_bo < 21:
    #     continue

    current_layer_name = name
    print('Searching Theta for ' + name)
    # progressf.write(name)
    # progressf.write('\n')

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1,
    )

    # optimizer = BayesianOptimization(
    #     f=None,
    #     pbounds=pbounds,
    #     random_state=1
    # )

    optimizer.set_gp_params(alpha=1e-3)

    #utility = UtilityFunction(kind="ucb", kappa=8, xi=0.0)
    #utility = UtilityFunction(kind="ei", kappa=8, xi=1e-2)

    # params = {"prune_level": 0.21, "bitwidth": 8}
    # target = black_box_function(**params)
    # optimizer.register(params=params, target=target)

    # params = {"prune_level": 0.2, "bitwidth": 4}
    # target = black_box_function(**params)
    # optimizer.register(params=params, target=target)
    #
    # params = {"prune_level": 0.2, "bitwidth": 8}
    # target = black_box_function(**params)
    # optimizer.register(params=params, target=target)
    #
    # params = {"prune_level": 0.9, "bitwidth": 4}
    # target = black_box_function(**params)
    # optimizer.register(params=params, target=target)
    #
    # params = {"prune_level": 0.9, "bitwidth": 8}
    # target = black_box_function(**params)
    # optimizer.register(params=params, target=target)


    # optimizer.probe(
    #     params={"prune_level": 0.2, "bitwidth": 2},
    #     lazy=True,
    # )
    #
    # optimizer.probe(
    #     params={"prune_level": 0.2, "bitwidth": 4},
    #     lazy=True,
    # )
    #
    # optimizer.probe(
    #     params={"prune_level": 0.2, "bitwidth": 8},
    #     lazy=True,
    # )
    #
    # optimizer.probe(
    #     params={"prune_level": 0.9, "bitwidth": 4},
    #     lazy=True,
    # )
    #
    # optimizer.probe(
    #     params={"prune_level": 0.9, "bitwidth": 8},
    #     lazy=True,
    # )



    # Kappa : 1 means prefer exploitation, 10 means prefer exploration

    optimizer.maximize(
        init_points=0,
        n_iter=int(ITERATIONS),
        acq='ei',
        kappa=8,
        alpha=1e-3
    )

    # points = []
    #
    # for i in range(int(ITERATIONS)):
    #     next_point = optimizer.suggest(utility)
    #     point_to_register = next_point.copy()
    #     bitwidthi = round(next_point['bitwidth'])
    #     point_to_register['bitwidth'] = bitwidthi
    #
    #     if points.__contains__(point_to_register):
    #         print("-------------------------------------------REPEAT-----------------------------")
    #
    #     points.append(point_to_register)
    #     target = black_box_function(**next_point)
    #     optimizer.register(params=next_point, target=target)




    # for i in range(int(ITERATIONS)):
    #     next_point = optimizer.suggest(utility)
    #     point_to_register = next_point.copy()
    #     bitwidthi = round(next_point['bitwidth'])
    #     point_to_register['bitwidth'] = bitwidthi
    #     if not points.__contains__(point_to_register):
    #         points.append(point_to_register)
    #         target = black_box_function(**next_point)
    #         optimizer.register(params=point_to_register, target=target)
    #     else:
    #         if bitwidthi < 8:
    #             print('Increasing bw from suggested val')
    #             point_to_register['bitwidth'] = bitwidthi + 1
    #         else:
    #             print('Decreasing bw from suggested val')
    #             point_to_register['bitwidth'] = bitwidthi - 1
    #
    #         if not points.__contains__(point_to_register):
    #             points.append(point_to_register)
    #             target = black_box_function(**next_point)
    #             optimizer.register(params=point_to_register, target=target)
    #         else:
    #             print('Still suggesting same')
    #             continue




        # print(target, next_point)
        # print('\n')

    prune_levels[name] = str(optimizer.max['params']['prune_level'])
    bitwidths[name] = str(optimizer.max['params']['bitwidth'])

# prune_levels = {'module.conv1.weight': '0.2560', 'module.layer1.0.conv1.weight': '0.1960', 'module.layer1.0.conv2.weight': '0.1530', 'module.layer1.1.conv1.weight': '0.0100', 'module.layer1.1.conv2.weight': '0.0100', 'module.layer1.2.conv1.weight': '0.0140', 'module.layer1.2.conv2.weight': '0.3080', 'module.layer2.0.conv1.weight': '0.1620', 'module.layer2.0.conv2.weight': '0.3020', 'module.layer2.0.downsample.0.weight': '0.1020', 'module.layer2.1.conv1.weight': '0.3380', 'module.layer2.1.conv2.weight': '0.2860', 'module.layer2.2.conv1.weight': '0.0100', 'module.layer2.2.conv2.weight': '0.0100', 'module.layer3.0.conv1.weight': '0.3240', 'module.layer3.0.conv2.weight': '0.2190', 'module.layer3.0.downsample.0.weight': '0.6240', 'module.layer3.1.conv1.weight': '0.0890', 'module.layer3.1.conv2.weight': '0.2630', 'module.layer3.2.conv1.weight': '0.3130', 'module.layer3.2.conv2.weight': '0.5010', 'module.fc.weight': '0.0100'}
#
# bitwidths = {'module.conv1.weight': '8', 'module.layer1.0.conv1.weight': '4', 'module.layer1.0.conv2.weight': '6', 'module.layer1.1.conv1.weight': '6', 'module.layer1.1.conv2.weight': '8', 'module.layer1.2.conv1.weight': '8', 'module.layer1.2.conv2.weight': '7', 'module.layer2.0.conv1.weight': '7', 'module.layer2.0.conv2.weight': '8', 'module.layer2.0.downsample.0.weight': '7', 'module.layer2.1.conv1.weight': '5', 'module.layer2.1.conv2.weight': '3', 'module.layer2.2.conv1.weight': '3', 'module.layer2.2.conv2.weight': '3', 'module.layer3.0.conv1.weight': '7', 'module.layer3.0.conv2.weight': '3', 'module.layer3.0.downsample.0.weight': '7', 'module.layer3.1.conv1.weight': '4', 'module.layer3.1.conv2.weight': '4', 'module.layer3.2.conv1.weight': '4', 'module.layer3.2.conv2.weight': '2', 'module.fc.weight': '7'}
print(prune_levels)
print(bitwidths)

write_final_schedule(STARTING_EPOCH)

run_distiller_final()

read_sparsities()

write_final_schedule_quant(0)
run_distiller_final_quant()

li=0
total_compressed_storage_req = 0
for name in layer_names:
    bw = round(float(bitwidths[layer_names[li]]))
    s = find_storage_req(li, bw)
    total_compressed_storage_req = total_compressed_storage_req + s
    layer_compression = (float(layer_sizes[li])*32)/s
    print('Layer ' + name + ' Compression: ' + str(layer_compression) + 'x')
    li = li + 1

compression_ratio = (network_size*32)/total_compressed_storage_req
print('Total Compressed Size: ' + str(total_compressed_storage_req))
print('Total Compression: ' + str(compression_ratio) + 'x')

#
#
final_top1 = read_top1()
final_sparsity = read_sparsity()
#
# li = 0
# compressed_network_size = 0
# for size in layer_sizes:
#     bw = round(float(bitwidths[layer_names[li]]))
#     lsize = int(size) * (1 - float(prune_levels[layer_names[li]])) * bw
#     compressed_network_size = compressed_network_size + lsize
#     li = li + 1
#
# compression_ratio = (network_size * 32) / compressed_network_size
#
# print(compression_ratio)
#

progressf.write(
     '\n Final Sparsity: {0:0.2f}    Final Top1: {1:0.3f}    Compression: {2:0.3f}\n'.format(final_sparsity, final_top1,
                                                                                             compression_ratio))

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

# os.system('python3 compress_classifier.py -a simplenet_cifar ../../../data.cifar10 -p 30 -j=4 --lr=0.000001 --out-dir logs/agp75/ --compress=/home/omer/schedules/agp75.yaml --resume logs/baseline/2019.04.01-203750/best.pth.tar --epochs 1 >/dev/null 2>&1')
# top1 = read_top1()
# sparsity = read_sparsity()
# print('Top 1 accuracy ' + str(top1))
# print('Sparsity ' + str(sparsity))
