from bayes_opt import BayesianOptimization

import datetime

iteration =0
def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    global iteration
    iteration = iteration + 1
    f.write('{},{},{}\n'.format(iteration, x, y))
    return -x ** 2 - (y - 1) ** 2 + 1




# network_size = 61770
#
# layer_sizes = {'module.conv1.weight': '450', 'module.conv2.weight': '2400', 'module.fc1.weight': '48000', 'module.fc2.weight': '10080', 'module.fc3.weight': '840'}
# bitwidths = {'module.conv1.weight': '7.996480315683638', 'module.conv2.weight': '7.533544976590081', 'module.fc1.weight': '3.8332277799932792', 'module.fc2.weight': '4.0', 'module.fc3.weight': '2.5188836116667774'}
# prune_levels = {'module.conv1.weight': '0.11865013577945516', 'module.conv2.weight': '0.21292680817479515', 'module.fc1.weight': '0.34062441135411103', 'module.fc2.weight': '0.2', 'module.fc3.weight': '0.08804933799585231'}
#
#
# li = 0
# compressed_network_size = 0
# for name,size in layer_sizes.items():
#     bw = round(float(bitwidths[name]))
#     lsize = float(size) * (1 - float(prune_levels[name])) * bw
#     compressed_network_size = compressed_network_size + lsize
#     li = li + 1
#
# compression_ratio = (network_size * 32) / compressed_network_size
#
# print(compression_ratio)
#

layer_sizes = [23232,307200,663552,884736,589824,37748736,16777216,4096000]
bitwidths = [8,8,8,4,3,2,3,8]
prune_levels = [0.19035651321979571,0.2001,0.19727572128502277,0.17016746741625835,0.10846661065893867,0.9901,0.5058178960162529,0.21519]

network_size = 61090496
li = 0
compressed_network_size = 0
for size in layer_sizes:
    bw = bitwidths[li]
    lsize = size * (1 - prune_levels[li]) * bw
    compressed_network_size = compressed_network_size + lsize
    li = li + 1

compression_ratio = (network_size * 32) / compressed_network_size

print(compression_ratio)




# now = datetime.datetime.now()
# f = open('progress.txt',"a")
# f.write('\n\n')
# f.write(str(now))
# f.write('\n\n')
#
# # Bounded region of parameter space
# pbounds = {'x': (2, 4), 'y': (-3, 3)}
#
# iteration = 0
# optimizer = BayesianOptimization(
#     f=black_box_function,
#     pbounds=pbounds,
#     random_state=1,
# )
#
# optimizer.maximize(
#     init_points=2,
#     n_iter=3,
# )
#
# f.close()