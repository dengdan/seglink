import vgg

net_dict = {
    "vgg": vgg
}

def get_basenet(name, inputs):
    net = net_dict[name];
    return net.basenet(inputs);
