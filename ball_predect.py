import numpy as np
import matplotlib.pyplot as plt

# 参数定义
g = 9.81  # 重力加速度 (m/s^2)
K_D = 0.1  # 阻尼系数
delta_t = 0.01  # 时间步长 (s)
num_steps = 1000  # 仿真步数

# 初始条件
b0 = np.array([0.0, 0.0])  # 初始位置 [x, y] (m)
v0 = np.array([10.0, 10.0])  # 初始速度 [vx, vy] (m/s)

# 轨迹存储
positions = [b0]
velocities = [v0]

# 数值计算函数
def compute_acceleration(position, velocity):
    norm_b = np.linalg.norm(position)
    return -g - K_D * norm_b * velocity

# 数值仿真
b_k = b0
v_k = v0

for _ in range(num_steps):
    a_k = compute_acceleration(b_k, v_k)  # 计算加速度
    v_k1 = v_k + delta_t * a_k  # 更新速度
    b_k1 = b_k + delta_t * v_k + 0.5 * delta_t**2 * a_k  # 更新位置

    # 更新当前状态
    positions.append(b_k1)
    velocities.append(v_k1)
    b_k, v_k = b_k1, v_k1

# 转换为numpy数组便于绘图
positions = np.array(positions)

# 绘制轨迹
plt.figure(figsize=(8, 6))
plt.plot(positions[:, 0], positions[:, 1], label="Trajectory")
plt.xlabel("x position (m)")
plt.ylabel("y position (m)")
plt.title("Object Trajectory Prediction")
plt.legend()
plt.grid()
plt.show()

