import numpy as np
import time
from matplotlib import pyplot as plt
import sys


class Logger(object):
    """
    输出控制台
    """

    def __init__(self, fileN="result1.txt"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def read(data_file, i):
    """
    读取txt文件
    :param data_file:输入测试样例文件路径
    :param i:样例i
    :return: W：背包最大容量(int)；N：物品数量(int)；w：每件物品的重量(list);v：每件物品的价值(list)
    """
    W = 0
    N = 0
    w = []
    v = []
    with open(data_file, 'r') as f:
        string = f.readlines()
        for j in range(len(string)):
            if string[j] == '(' + str(i) + ')' + ' \n':
                W = int(string[j + 2].split(' ')[0])
                N = int(string[j + 2].split(' ')[1])
                for k in range(1, N + 1):
                    w.append(int(string[j + k + 2].split(' ')[0]))
                    v.append(int(string[j + k + 2].split(' ')[1]))
    return W, N, w, v


def GA(W, N, w, v, save_fig_path):
    """
    遗传算法解决0-1背包问题主函数
    :param W: 背包最大承重
    :param N: 物品总数
    :param w: 每件物品的重量
    :param v: 每件物品的价值
    :param save_fig_path: 样例i的收敛曲线存储路径
    :return: max_value:求解的放入背包的物品最大价值(int)；best_solu：放入背包的物品序号(list)
    """
    # -----------------遗传算法-----------
    max_iteration = 1000
    solutions = [0] * max_iteration  # 输出每轮迭代装入背包的物品价值

    length = N  # 染色体长度
    number = 100  # 种群规模
    retain_rate = 0.2  # 精英保持率
    rand_select_rate = 0.5  # 随机选择率，非精英的，有0.5的概率保持
    crossover_rate = 0.9  # 交叉率
    mutation_rate = 0.3  # 变异率

    # 初始化种群
    init_rate = 0.2  # 种群初始化染色体为1的比例
    population = np.zeros([length, number])
    for i in range(length):
        for j in range(number):
            if np.random.random() < init_rate:
                population[i][j] = 1

    # 计算适应度
    def fitness(chromosome):
        # weight, price = 0, 0
        # for i in range(len(chromosome)):
        #     if chromosome[i] == 1:
        #         weight += w[i]
        #         price += v[i]
        weight = np.matmul(chromosome, w)
        price = np.matmul(chromosome, v)
        return price if weight <= W else 0

    # 选择
    def selection(population):
        # print('==selection==')
        sort_population = np.array([[] for _ in range(len(population) + 1)])
        for i in range(len(population[0])):
            x1 = population[:, i]
            x2 = fitness(x1)
            x = np.r_[x1, x2]
            sort_population = np.c_[sort_population, x]

        sort_population = sort_population.T[np.lexsort(sort_population)].T  # 联合排序，从小到大排列

        # 选出适应性强的个体，精英选择
        retain_length = sort_population.shape[1] * retain_rate
        parents = np.array([[] for _ in range(len(population) + 1)])
        for i in range(int(retain_length)):
            y1 = sort_population[:, -(i + 1)]
            parents = np.c_[parents, y1]

        rest = len(sort_population[0]) - retain_length  # 非精英个数
        for i in range(int(rest)):
            if np.random.random() < rand_select_rate:
                y2 = sort_population[:, i]
                parents = np.c_[parents, y2]
        parents = np.delete(parents, -1, axis=0)
        parents = np.array(parents, dtype=np.int16)
        return parents

    # 交叉
    def crossover(parents):
        # print("==crossover==")
        children = np.array([[] for _ in range(length)])
        while children.shape[1] < number:
            father = np.random.randint(0, parents.shape[1])
            mother = np.random.randint(0, parents.shape[1])
            if father == mother:    continue
            father = parents[:, father]
            mother = parents[:, mother]
            child1 = father[:]
            child2 = mother[:]
            if np.random.random() < crossover_rate:
                crossP = np.random.randint(0, length)  # 随机选取染色体上的交叉点
                for i in range(length):
                    if i < crossP:
                        child1[i] = father[i]
                        child2[i] = mother[i]
                    else:
                        child1[i] = mother[i]
                        child2[i] = father[i]
            children = np.c_[children, child1, child2]
        children = np.array(children, dtype=np.int16)
        return children

    # 变异
    def mutation(children):
        # print("==mutation==")
        for i in range(len(children[0])):
            if np.random.random() < mutation_rate:
                j = np.random.randint(0, length)
                children[:, i][j] ^= 1
        children = np.array(children, dtype=np.int16)
        return children

    # 开始迭代
    iteration = 0
    result = dict()
    while iteration < max_iteration:
        parents = selection(population)  # 自然选择
        children = crossover(parents)  # 交叉
        mutation_children = mutation(children)  # 变异
        population = mutation_children  # 新的种群

        fitn = list()
        for i in range(len(population[0])):
            fit = fitness(population[:, i])
            fitn.append(fit)
        solutions[iteration] = max(fitn)
        result[max(fitn)] = population[:, fitn.index(max(fitn))]

        # if iteration % 100 == 0:
        #     print('==============')
        #     print('iteration:', iteration)
        #     print('population:', population.shape)
        #     print('solutions:', max(solutions))
        #     print('result:', result[max(fitn)])

        iteration += 1

    # -----------------绘制每轮迭代的解---------------

    # 画出收敛曲线
    plt.cla()
    plt.plot(solutions)
    plt.title('The curve of the optimal function value of each generation with the number of iterations',
              color='#123456')
    plt.xlabel('the number of iterations')
    plt.ylabel('the optimal function value of each generation')
    plt.savefig(save_fig_path)
    # plt.show()

    max_value = max(solutions)
    best_solu = result[max_value]
    return max_value, best_solu


if __name__ == '__main__':
    data_file = "txt1.txt"
    sys.stdout = Logger("result1.txt")
    for i in range(1, 6):
        W, N, w, v = read(data_file, i)
        start = time.time()
        print("样例%d:" % i)
        print("背包最大承重:", W)
        print("物品件数：", N)
        print("每件物品的重量：", w)
        print("每件物品的价值：", v)

        save_fig_path = "result" + str(i) + ".png"
        print("收敛曲线存储文件名：", save_fig_path)
        max_value, best_solu = GA(W, N, w, v, save_fig_path)
        print("装入背包的最大价值：", max_value)
        print("装入背包最大价值时的最优物品组合：", best_solu)
        end = time.time()
        print("测试用时：%f" % (end - start))


"""
133
334
2614
2558
2617
"""