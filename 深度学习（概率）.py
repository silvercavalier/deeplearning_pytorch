import torch
from torch.distributions import multinomial
from d2l import torch as d2l
fair_probs = torch.ones([6]) / 6 # 创建长度为6的均匀概率分布
multinomial.Multinomial(1,fair_probs).sample() # sample()输出每个类别被抽到的次数（返回一个张量）
""" multinomial方法：是一个按权重抽签的函数，根据给定的概率权重随机抽取元素的下标（索引）
    如此处，相当于掷一次公平的六面骰子，将随机返回0到5中的某一个数字
    关键参数：num_samples（抽取数量）
    replacement（是否放回），默认为False
    返回torch.Longtensor类型的索引
"""
# 多次抽取
counts = multinomial.Multinomial(10,fair_probs).sample((500,))
"""向sample函数传参：(a,b,...)
   返回a * b * ...组独立结果
   形成一个形状为(a,b,......,shape(fair_probs))的张量
"""
cum_counts = counts.cumsum(dim=0) # 累计求和：把当前位置以及之前所有元素累加，作为对应输出位置的数值
"""[1,2,3,4].cumsum = [1,3,6,10]"""

