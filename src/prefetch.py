import queue
import threading

def prefetch_decorator(next_fn, buffer_size=4):
    """
    预取装饰器：用队列缓存提前加载的数据
    next_fn: 原函数（每次返回一个字符串，可能有网络阻塞）
    buffer_size: 预取缓冲区大小
    """
    class PrefetchWrapper:
        def __init__(self):
            self.queue = queue.Queue(maxsize=buffer_size)
            self.stop_event = threading.Event()
            self.thread = threading.Thread(target=self._prefetch_worker, daemon=True)
            self.thread.start()  # 启动后台线程

        def _prefetch_worker(self):
            """后台线程：持续预取数据到队列"""
            while not self.stop_event.is_set():
                try:
                    data = next_fn()
                    self.queue.put(data)
                except StopIteration:
                    self.queue.put(None)  # 结束标志
                    break
                except Exception as e:
                    self.queue.put(e)     # 传递异常

        def __iter__(self):
            return self

        def __next__(self):
            data = self.queue.get()
            if isinstance(data, Exception):
                raise data
            if data is None:
                self.stop_event.set()
                raise StopIteration
            return data

    return PrefetchWrapper()


# 测试
if __name__ == "__main__":
    import time
    def your_next_function():
        print("start fetch sleep 1")
        time.sleep(1)
        return "data"

    prefetched_next = prefetch_decorator(your_next_function, buffer_size=4)
    time.sleep(2)
    print("start")
    for data in prefetched_next:
        print(f"get data: {data}")
        time.sleep(2)
