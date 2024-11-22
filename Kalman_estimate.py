import numpy as np
import threading
import time

# 参数定义
g = 9.81  # 重力加速度 (m/s^2)
K_D = 0.1  # 阻尼系数
delta_t = 0.002  # 数值计算时间步长 (s)
measurement_noise = 0.5  # 测量噪声协方差
process_noise = 0.1  # 过程噪声协方差
initial_state = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])  # 初始状态 [x, y, z, vx, vy, vz]

# 卡尔曼滤波变量初始化
state = initial_state  # 状态向量 [x, y, z, vx, vy, vz]
P = np.eye(6)  # 状态协方差矩阵 (初始误差)
Q = np.eye(6) * process_noise  # 过程噪声协方差矩阵
R = np.eye(3) * measurement_noise  # 测量噪声协方差矩阵
H = np.zeros((3, 6))  # 测量矩阵
H[:, :3] = np.eye(3)  # 测量位置 (仅观察位置)

# 状态转移矩阵 (离散化动力学方程)
F = np.eye(6)
F[:3, 3:] = delta_t * np.eye(3)  # 位置和速度的耦合项

# 过程噪声模型中的力
G = np.zeros((6, 3))
G[3:, :] = delta_t * np.eye(3)

# 共享变量，用于多线程打印结果
estimated_position = None
estimated_velocity = None

# 模拟真实的加速度
def compute_true_acceleration(position, velocity):
    norm_b = np.linalg.norm(position)
    acceleration = np.zeros(3)
    acceleration[:2] = -K_D * norm_b * velocity[:2]  # x, y方向阻尼力
    acceleration[2] = -g - K_D * norm_b * velocity[2]  # z方向阻尼力和重力
    return acceleration

# 卡尔曼滤波器
def kalman_filter(measurement):
    global state, P, estimated_position, estimated_velocity

    # 1. 预测阶段
    # 预测状态
    acceleration = compute_true_acceleration(state[:3], state[3:])  # 获取真实加速度
    u = np.array([0, 0, g])  # 控制量（用于考虑恒定的外力模型，例如重力）
    state = F @ state + G @ (u + acceleration)  # 状态预测
    P = F @ P @ F.T + Q  # 更新状态协方差矩阵

    # 2. 更新阶段
    # 计算卡尔曼增益
    y = measurement - H @ state  # 观测与预测的残差
    S = H @ P @ H.T + R  # 残差协方差
    K = P @ H.T @ np.linalg.inv(S)  # 卡尔曼增益

    # 更新状态和协方差矩阵
    state = state + K @ y
    P = (np.eye(6) - K @ H) @ P

    # 更新估计结果
    estimated_position = state[:3]
    estimated_velocity = state[3:]

# 模拟测量值 (添加噪声)
def generate_measurement(true_position):
    noise = np.random.normal(0, measurement_noise, size=3)
    return true_position + noise

# 预测线程 (每20ms运行一次)
def prediction_thread():
    true_position = initial_state[:3]
    true_velocity = initial_state[3:]

    while True:
        start_time = time.time()

        # 数值积分计算真实轨迹
        acceleration = compute_true_acceleration(true_position, true_velocity)
        true_velocity = true_velocity + delta_t * acceleration
        true_position = true_position + delta_t * true_velocity + 0.5 * delta_t**2 * acceleration

        # 生成带噪声的测量值
        measurement = generate_measurement(true_position)

        # 卡尔曼滤波更新
        kalman_filter(measurement)

        # 打印估计结果
        print(f"Estimated Position: {estimated_position}")
        print(f"Estimated Velocity: {estimated_velocity}")

        # 控制线程频率为50Hz
        elapsed_time = time.time() - start_time
        time.sleep(max(0, 1 / 50 - elapsed_time))  # 每20ms运行一次

# 启动线程
thread = threading.Thread(target=prediction_thread, daemon=True)
thread.start()

# 主线程等待一段时间，观察输出
time.sleep(5)  # 运行5秒观察结果

