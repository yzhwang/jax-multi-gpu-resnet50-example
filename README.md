# jax-multi-gpu-resnet50-example

This repo shows how to use jax for multi-node multi-GPU training. The example is adapted from the resnet50 example in dm-haiku (https://github.com/deepmind/dm-haiku/tree/main/examples/imagenet). It only requires each node knows the IP of the rank 0 node, very similar to PyTorch's DDP.

When two containers on the same cluster are running, one can run the following script in each container to launch a multi-node multi-GPU training job:

`
python train.py --server_ip=$ROOT_IP --server_port=$PORT --num_hosts=$NUM_HOSTS --host_idx=$HOST_IDX
`

