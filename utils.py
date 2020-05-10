def replace(network, original, replacement, name):
    def replace_(m, name, original=original, replacement=replacement):
        for attr_str in dir(m):
            target_attr = getattr(m, attr_str)
            if type(target_attr) == original:
                setattr(m, attr_str, replacement())
    for m in network.modules():
        replace_(m, name)
