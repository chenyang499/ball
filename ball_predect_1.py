import numpy as np
import threading
import time

# 参数定义
g = 9.81  # 重力加速度 (m/s^2)
K_D = 0.1  # 阻尼系数
delta_t = 0.002  # 数值计算时间步长 (s)
num_steps = int(1 / delta_t)  # 1秒的预测所需步数

# 初始条件
initial_position = np.array([0.0, 0.0])  # 初始位置 [x, y] (m)
initial_velocity = np.array([10.0, 10.0])  # 初始速度 [vx, vy] (m/s)

# 预测结果共享变量
predicted_position = None
predicted_velocity = None

# 数值计算函数
def compute_acceleration(position, velocity):
    norm_b = np.linalg.norm(position)
    return -g - K_D * norm_b * velocity

# 预测轨迹计算函数
def predict_trajectory(initial_position, initial_velocity):
    global predicted_position, predicted_velocity

    # 初始化当前状态
    position = initial_position
    velocity = initial_velocity

    # 数值积分计算
    for _ in range(num_steps):
        acceleration = compute_acceleration(position, velocity)  # 计算加速度
        velocity = velocity + delta_t * acceleration  # 更新速度
        position = position + delta_t * velocity + 0.5 * delta_t**2 * acceleration  # 更新位置

    # 保存预测结果
    predicted_position = position
    predicted_velocity = velocity

# 每20ms (50Hz) 的计算线程
def prediction_thread():
    while True:
        start_time = time.time()

        # 调用预测计算函数
        predict_trajectory(initial_position, initial_velocity)

        # 打印预测结果
        print(f"Predicted Position: {predicted_position}")
        print(f"Predicted Velocity: {predicted_velocity}")

        # 控制线程运行频率为50Hz
        elapsed_time = time.time() - start_time
        time.sleep(max(0, 1 / 50 - elapsed_time))  # 每20ms运行一次

# 启动线程
thread = threading.Thread(target=prediction_thread, daemon=True)
thread.start()

# 主线程等待一段时间，观察输出
time.sleep(5)  # 运行5秒观察预测结果

