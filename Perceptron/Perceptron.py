import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt

tf.disable_v2_behavior()


# BP反向传播更新权重参数
def BP(x, h, d, y, w1, w2, b1, b2, eta):
    delt_2 = (d - y) * y * (1 - y)  # 6.18
    delt_w2 = eta * tf.matmul(tf.transpose(delt_2), h)  # 6.19
    w2 = w2 + delt_w2  # 6.20
    b2 = b2 + eta * delt_2

    sigmoid_ = h * (1 - h)  # sigmoid函数的导数
    delt_1 = tf.matmul(delt_2, w2) * sigmoid_  # 6.25
    delt_w1 = eta * tf.matmul(tf.transpose(delt_1), x)  # 6.27
    w1 = w1 + delt_w1  # 6.28
    b1 = b1 + eta * delt_1

    # 返回更新后的权重和偏置
    return w1, w2, b1, b2


# 计算误差   公式6.6
def evaluate(session, d, y):
    sub = tf.subtract(y, d)  # 相减
    power = tf.multiply(sub, sub)  # 平方
    E = session.run(tf.reduce_sum(power))  # 求和
    E /= 2  # 除以2
    return E


# 训练
def train(iteration):
    # 导入数据
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # 输入层到隐藏层的权重
    w1 = tf.Variable(tf.random_uniform([32, 784], minval=0, maxval=0.1), name='w1')
    # 隐藏层到输出层的权重
    w2 = tf.Variable(tf.random_uniform([10, 32], minval=0, maxval=0.1), name='w2')
    # 输入层到隐藏层的偏置（阈值）
    b1 = tf.Variable(tf.random_uniform([1, 32], minval=0, maxval=0.1, ), name='b1')
    # 隐藏层到输出层的偏置（阈值）
    b2 = tf.Variable(tf.random_uniform([1, 10], minval=0, maxval=0.1, ), name='b2')
    # 输入
    x = tf.placeholder('float', [None, 784])
    # 期望（标签）
    d = tf.placeholder('float', [None, 10])

    eta = 0.1  # 增益因子
    e = 0.1  # 误差阈值

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    result_show = list()
    for i in range(iteration):
        # print('iteration:', i)
        # 取数据
        batch_x, batch_d = mnist.train.next_batch()
        x, d = batch_x, batch_d
        # 隐藏层
        h = tf.nn.sigmoid(tf.matmul(w1, tf.transpose(x)) - tf.transpose(b1))
        h = tf.transpose(h)
        # 输出层
        y = tf.nn.sigmoid(tf.matmul(w2, tf.transpose(h)) - tf.transpose(b2))
        y = tf.transpose(y)
        # 计算误差
        # E = evaluate(session, d, y)
        # result_show.append(1 - E / 4)
        if i % 10 == 0 or i == iteration - 1:
            E = evaluate(session, d, y)
            result_show.append(1 - E / 5)
            print(i, E)
        if E <= e:
            break
        # 反向传播
        w1, w2, b1, b2 = BP(x, h, d, y, w1, w2, b1, b2, eta)

    print('accuracy:', result_show)
    plt.plot(result_show)
    plt.title('iteration — accuracy')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.savefig('MNIST_data/result.png')
    plt.show()


# 训练次数
train(100)
