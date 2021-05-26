import queue
import threading
import numpy as np

# 多线程计算
class ThreadedIterator:
    """An iterator object that computes its elements in a parallel thread to be ready to be consumed.
    The iterator should *not* return None"""

    def __init__(self, original_iterator, max_queue_size: int = 2):
        self.__queue = queue.Queue(maxsize=max_queue_size)
        self.__thread = threading.Thread(target=lambda: self.worker(original_iterator))
        self.__thread.start()

    def worker(self, original_iterator):
        for element in original_iterator:
            assert element is not None, 'By convention, iterator elements much not be None'
            self.__queue.put(element, block=True)
        self.__queue.put(None, block=True)

    def __iter__(self):
        next_element = self.__queue.get(block=True)
        while next_element is not None:
            yield next_element
            next_element = self.__queue.get(block=True)
        self.__thread.join()


# glorot初始化
def glorot_init(shape):
    initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32)


# 将graph转换成[num_edge_types, max_n_vertices, max_n_vertices]邻接矩阵
def graph_to_adj_mat(graph, max_n_vertices, num_edge_types, tie_fwd_bkwd=False):
    # print(max_n_vertices,num_edge_types)
    bwd_edge_offset = 0 if tie_fwd_bkwd else (num_edge_types // 2)
    amat = np.zeros((num_edge_types, max_n_vertices, max_n_vertices))
    # 序号的起始是1，因此都记得要减1
    for src, e, dest in graph:
        # 如果graph是通过补0扩展得到的，那么补充的部分应该忽略
        if (e == 0 and src == 0 and dest == 0):
            continue
        amat[e - 1, dest - 1, src - 1] = 1
        amat[e - 1 + bwd_edge_offset, src - 1, dest - 1] = 1
    return amat