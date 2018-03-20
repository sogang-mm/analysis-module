try:    # python 2.X
    import Queue as Queue
except: # python 3.X
    import queue as Queue


class ModuleManager:
    module_queue = Queue.Queue()

    def __init__(self, modules, num_of_modules):
        for i in range(num_of_modules):
            self.module_queue.put(modules())

    def get_result_by_path(self, image_path):
        now_module = self.module_queue.get()
        result = now_module.inference_by_path(image_path)
        print (result)
        self.module_queue.put(now_module)
        return result


# TODO
from dummy.example import Dummy
analyzer_module = ModuleManager(Dummy, 4)