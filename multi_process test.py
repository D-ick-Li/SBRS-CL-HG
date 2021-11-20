from multiprocessing.dummy import Pool
import itertools


class Funcs(object):
    def __init__(self, a):
        self.a = a

    def func1(self, param_tuple):
        x = param_tuple[0]
        y = param_tuple[1]
        z = param_tuple[2]
        print("子进程在执行函数1")
        return x + y + self.a, [x, y, z]

    def func2(self, param_tuple):
        x = param_tuple[0]
        y = param_tuple[1]
        print("子进程在执行函数2")
        return x*y


class MultiRun(object):
    def __init__(self, lenth):
        self.lenth = lenth

    def func3(self):
        funcs = Funcs(1)
        B = []
        C = []
        pool = Pool(4)
        # result1, result2 = [], []
        for i in range(5):
            params = list(itertools.product(range(self.lenth), [2], [i]))
            params2 = list(itertools.product(range(self.lenth), range(2)))
            b = pool.map(funcs.func1, params)
            c = pool.map(funcs.func2, params2)

            # b = pool.map(funcs.func1, params)
            # c = pool.map(funcs.func2, params2)
            B.append(b)
            C.append(c)
        pool.close()
        pool.join()
        """
        for b in B:
            print(b.get())
        for c in C:
            print(c.get())
        """
        return B, C

    def call_back_test(x):
        print('the result is {}'.format(x))

if __name__ == '__main__':
    """
    pool = Pool(4)
    b = pool.map(func1, [1,2,3,4,5,6,7,8,9])
    pool.close()
    pool.join()
    print(type(b))
    """
    multi_run = MultiRun(10)
    B, C = multi_run.func3()

    print(type(B))
