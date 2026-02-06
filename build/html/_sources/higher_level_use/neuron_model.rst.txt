神经元模型
==========

7.1 汇编代码
------------
本节指出神经元模型在底层由 BCE 扩展指令集实现。详细指令请参见《BCE 体系结构之指令集》。

7.2 事件语言
------------
事件语言用于描述神经元模型的事件驱动处理逻辑。下列示例给出一个典型的 LIF（leaky integrate-and-fire）神经元模型及其事件处理流程，并在注释中说明了关键语义。

示例：

.. code-block::

    start
    # 全局变量声明
    int global_count = 0

    # 自定义神经元模型
    neuron lif:
        float v = 0.0
        float threshold = 1.0
        float reset = 0.0
        bool spike = false

    # 事件声明（使用二进制/十六进制或自定义编码）
    set event_spike                       00
    set event_timestep                    01
    set event_timestep_with_spike         10
    set event_decay                       11

    # 等待事件到来：编译器将 wait_event 映射为分支指令以跳转到 main
    wait_event
    # 示例映射（说明）：wait_event -> bre a1 a1  （a1 存放 main 的地址）

    # 主事件循环：根据事件类型跳转到对应处理块
    main:
        if event_type == event_spike:
            goto event_spike
        if event_type == event_timestep:
            goto event_timestep
        if event_type == event_timestep_with_spike:
            goto event_timestep_with_spike
        if event_type == event_decay:
            goto event_decay
        # 忽略其它类型的事件（假设调用方设置正确的事件类型）

    # 处理 event_spike：只做积分，不立即发放脉冲；结束时更新神经元状态
    event_spike:
        # 将输入权重累加到膜电位（示例表述）
        lif.v = potential + weight
    end_event
    # end_event 语义：在硬件上触发状态寄存器更新，然后返回 main 循环等待下一个事件

    # 处理 event_timestep：判断是否产生脉冲（spike）；若产生则发放并重置状态
    event_timestep:
        lif.v = potential
        if (lif.v >= lif.threshold):
            lif.spike = true
            # 发放脉冲事件（由运行时/硬件处理）
            emit_spike
            # 重置神经元状态
            lif.v = lif.reset
            lif.threshold += 0.1 * lif.threshold
        else:
            lif.spike = false
    end_event

    # 处理膜电位衰减事件（decay）：仅做衰减，最后更新状态
    event_decay:
        lif.v = 0.9 * potential
    end_event

    # 处理同时有输入和判断发放的事件
    event_timestep_with_spike:
        lif.v = potential + weight
        if (lif.v >= lif.threshold):
            lif.spike = true
            emit_spike
            lif.v = lif.reset
            lif.threshold += 0.1 * lif.threshold
        else:
            lif.spike = false
    end_event

    end

说明：

- 各事件处理块以 end_event 结束，编译器负责将其映射为必要的寄存器/指令序列以更新硬件状态并返回主循环。
- emit_spike 语义由运行时或硬件实现，表示向外部发放脉冲。
- 示例中的变量（potential、weight 等）为外部输入或上文定义的上下文变量，应根据实际模型补充声明。
