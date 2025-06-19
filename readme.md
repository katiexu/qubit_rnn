qubit_RNN: 一个类似RNN结构的pennylane量子电路。

single_qubit_circuit: 类RNN cell   
输入 params, input_val, hidden_state  
输出 hidden_state

circuit： 循环层+神经网络层  
循环层： n个cell  
神经网络层： (n, 4) (4, 1)  
输入：params, input_val  
输出：out

结果  
![Figure_1.png](Figure_1.png)