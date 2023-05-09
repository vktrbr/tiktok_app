import resource


def set_max_memory():
    resource.setrlimit(resource.RLIMIT_AS, (2 ** 39, 20 * 2 ** 39))
