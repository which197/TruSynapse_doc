子网执行体
==========

子网执行体概述
----------

TODO: 添加子网执行体介绍

TruSynapse 将神经网络以 NFU（Neural Functional Unit）为粒度进行切分、存储布局和调度决策。每个子网对应一个适配单个 NFU 的计算单元，包含其计算图、权重和必要的元数据，能独立进行部署与执行。

关键行为与特性：

- 切分与存储：模型的计算与参数按 NFU 粒度划分，便于实现内存局部性和带宽优化。
- 调度：专用子网调度器负责将子网映射到具体 NFU 上，调度策略类似操作系统将进程调度到 CPU 核心，包含负载均衡和资源约束考虑。
- 分布式执行：通过上层资源管理组件可实现跨节点部署，从而支持并行计算与弹性扩展。

上述设计使得子网在单节点和分布式环境中均能高效调度与执行，并利于资源隔离与扩展性规划。

子网执行体结构
--------------

NFU子网执行体结构中的神经网络描述信息包含：

#. NFU编号
#. 子网在神经网络全局中的位置，比如是否是输入层/输出层/隐藏层
#. NFU内各GNC之间的连接关系
#. 子网的输入层、输出层神经元列表
#. 前驱NFU，以及前驱NFU输出层逻辑神经元号与本子网输入层物理神经元的对应关系
#. 后继NFU


子网执行体数据结构SNNData
------------------------

.. code-block:: python
    :linenos:

    class SNNData(Structure):
    """SNN计算所需的数据结构，与C库中的定义对应"""
        _fields_ = [
            ("params", c_uint64 * 27),                      # 27个配置参数
            ("connection_data", POINTER(c_uint64)),         # 连接数据
            ("connection_len", c_size_t),                   # 连接数据长度
            ("inputneuronlist_data", POINTER(c_uint32)),    # 输入神经元列表
            ("inputneuronlist_len", c_size_t),              # 输入神经元列表长度
            ("inputspike_data", POINTER(c_uint32)),         # 输入脉冲数据
            ("inputspike_len", c_size_t),                   # 输入脉冲数据长度
            ("neuronbase_data", POINTER(c_uint32)),         # 神经元数据
            ("neuronbase_len", c_size_t),                   # 神经元数据长度
            ("neuron_data", POINTER(c_uint32)),             # 神经元配置数据
            ("neuron_data_len", c_size_t),                  # 神经元配置数据长度
            ("output_data", POINTER(c_uint32)),             # 输出结果数据
            ("output_len", c_size_t)                        # 输出结果长度
        ]

**字段说明**：

- **params**: 27个64位配置参数，对应snnparam.txt文件内容，定义硬件工作模式和内存布局
- **connection_data**: 指向64位连接权重数据的指针，描述神经元之间的连接强度
- **inputneuronlist_data**: 指向32位输入神经元列表的指针，指定哪些神经元接收外部输入
- **inputspike_data**: 指向32位输入脉冲时序数据的指针，定义输入脉冲的时间序列
- **neuronbase_data**: 指向32位神经元基础参数的指针，包含神经元的初始状态和基本属性
- **neuron_data**: 指向32位神经元配置数据的指针，定义神经元的类型和特性参数
- **output_data**: 输出结果数据指针，由驱动库在计算完成后填充
- **各`_len`字段**: 对应数据数组的元素数量


