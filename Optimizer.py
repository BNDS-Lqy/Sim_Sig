# ========== 环境配置 ==========
import os
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import csv
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False

# ============================ 核心参数 =============================
N_0 = 10 ** ((-174 - 30) / 10)  # 噪声功率谱密度 (W/Hz)
L_0 = 70  # 参考路径损耗 (dB)
GRID_SIZE = 200  # 网格精度
MAX_DISTANCE = 200  # 最大模拟距离 (m)
MIN_DISTANCE = 1  # 最小计算距离 (m)
COLORBAR_MIN = 0  # 速率下限
# 比例公平算法参数
ALPHA = 0.15  # 指数移动平均权重
EPS = 1e-6  # 避免除零
# 评价函数参数（Score计算核心）
W_S = 0.6  # 均速权重
W_G = 0.4  # 公平权重
S_MAX = 500.0  # 均速上限(Mbps)
K = 1e5  # 缩放系数
MIN_SPEED = 10.0  # 最低网速阈值(Mbps)
AVG_SPEED = 30.0  # 平均网速阈值(Mbps)
MACRO_BS_COST = 50.0  # 宏基站单价(万元)
MICRO_BS_COST = 10.0  # 微基站单价(万元)
MAX_COST = 500.0  # 最大总成本(万元)

#查询函数用区：load _get_blind Gaoptimizer vital from_csv

# 基站参数配置
bs_type_config = {
    '宏基站': {
        'P_t_range': (100, 400),
        'n_value': 3,  # 路径损耗指数
        'B_range': (100, 1000),
        'B_default': 200,  # 默认发射功率
        'P_t_default': 200  # 默认带宽
    },
    '微基站': {
        'P_t_range': (5, 20),
        'n_value': 4,
        'B_range': (100, 200),
        'B_default': 100,
        'P_t_default': 10
    }
}
freq_config = {3.5: 3.5e9, 28: 28e9}  # 频率配置
DEFAULT_FREQ = 3.5  # 默认频率(GHz)
# ============================ 贪心算法核心参数（解决堆叠/覆盖问题）============================
INIT_MACRO_NUM = 2  # 初始宏基站数量
MAX_MACRO_NUM = 10  # 最大宏基站数量
MAX_MICRO_NUM = 50  # 最大微基站数量
# 基站最小距离约束 (m) - 放宽约束，允许更灵活部署
MIN_MACRO_DIST = 150  # 从200降低到150
MIN_MICRO_DIST = 30  # 从50降低到30
MIN_MACRO_MICRO_DIST = 80  # 从100降低到80
# 区域网格化参数
GRID_CELL_SIZE = 100  # 网格大小(m)，用于密度/速率评估
# 动态基站类型选择参数
LARGE_AREA_THRESHOLD = 40000  # 大面积盲区阈值(m²)，超过则补宏基站
# 边际效益终止参数
MIN_SPEED_IMPROVE = 1.0  # 最低速率最小提升值
STAGNANT_ITER = 2  # 连续迭代无提升则终止
# ============================ GA新增/位置调整参数 =============================
MAX_MACRO_ADD = 5  # 从3提升到5（允许更多宏基站新增）
MAX_MICRO_ADD = 30  # 从20提升到30（允许更多微基站新增）
MACRO_MUTATE_RATIO = 0.6  # 从0.3提升到0.6（宏基站位置变异率翻倍）
MICRO_POS_STEP = 0.6  # 从0.3提升到0.6（微基站移动幅度翻倍）
MACRO_POS_STEP = 0.5  # 从0.2提升到0.5（宏基站移动幅度翻倍）
GA_POP_SIZE = 50  # 从50提升到100（种群多样性提升）
GA_MAX_ITER = 60  # 从50提升到200（迭代次数翻倍，充分探索）


# ============================ 核心计算函数 =============================
def calculate_base_total_capacity(base_stations):
    """计算系统理论总容量（参考距离1m，路径损耗L0）"""
    theoretical_capacity = 0.0
    for bs in base_stations:
        bs_P_t = bs['P_t']
        bs_B = bs['B'] * 1e6  # MHz转Hz
        bs_n = bs['n']
        # 参考距离：MIN_DISTANCE=1m
        L_d = L_0
        P_t_dBm = 10 * np.log10(bs_P_t * 1000)
        P_r_dBm = P_t_dBm - L_d
        P_r = 10 ** (P_r_dBm / 10) / 1000  # dBm转W
        noise = N_0 * bs_B
        SNR = P_r / noise if noise != 0 else 0
        # 香农公式：转Mbps（除以1e6）
        bs_theoretical_capacity = bs_B * np.log2(1 + SNR) / 1e6
        theoretical_capacity += bs_theoretical_capacity
    return max(theoretical_capacity, COLORBAR_MIN)


def calculate_user_instant_speed(user_x, user_y, base_stations):
    """计算单个用户的瞬时速率"""
    instant_speed = 0.0
    if not base_stations:
        return instant_speed
    for bs in base_stations:
        bs_x, bs_y = bs['x'], bs['y']
        bs_P_t, bs_n = bs['P_t'], bs['n']
        bs_B = bs['B'] * 1e6  # MHz转Hz
        # 计算距离
        distance = np.sqrt((user_x - bs_x) ** 2 + (user_y - bs_y) ** 2)
        distance = max(distance, MIN_DISTANCE)
        # 路径损耗计算
        L_d = L_0 + 10 * bs_n * np.log10(distance)
        P_t_dBm = 10 * np.log10(bs_P_t * 1000)
        P_r_dBm = P_t_dBm - L_d
        P_r = 10 ** (P_r_dBm / 10) / 1000
        noise = N_0 * bs_B
        SNR = P_r / noise if noise != 0 else 0
        speed = bs_B * np.log2(1 + SNR) / 1e6  # 转Mbps
        instant_speed += speed
    return max(instant_speed, COLORBAR_MIN)


def update_all_users_pf_speed(users, base_stations):
    """比例公平调度算法"""
    if not base_stations or len(users) == 0:
        return users
    # 计算系统理论总容量
    base_total_capacity = calculate_base_total_capacity(base_stations)
    if base_total_capacity < EPS:
        for user in users:
            user['pf_speed'] = 0
        return users
    # 更新瞬时速率和平均速率
    user_priorities = []
    for user in users:
        instant_speed = calculate_user_instant_speed(user['x'], user['y'], base_stations)
        user['instant_speed'] = instant_speed
        avg_speed = user['avg_speed']
        # 指数移动平均更新
        if avg_speed < EPS:
            user['avg_speed'] = instant_speed
        else:
            user['avg_speed'] = ALPHA * instant_speed + (1 - ALPHA) * avg_speed
        # 计算PF优先级
        priority = instant_speed / max(user['avg_speed'], EPS)
        user_priorities.append(priority)
    # 按优先级分配速率（约束：不超过瞬时速率）
    total_priority = sum(user_priorities)
    if total_priority < EPS:
        equal_share = base_total_capacity / len(users)
        for user in users:
            user['pf_speed'] = max(min(equal_share, user['instant_speed']), COLORBAR_MIN)
    else:
        for i, user in enumerate(users):
            pf_speed = (user_priorities[i] / total_priority) * base_total_capacity
            user['pf_speed'] = max(min(pf_speed, user['instant_speed']), COLORBAR_MIN)
    return users


def calculate_evaluation_score(users, base_stations):
    """计算评价分数（核心优化目标）"""
    if len(users) == 0 or not base_stations:
        return 0.0
    # 更新用户PF速率
    users = update_all_users_pf_speed(users, base_stations)
    pf_speeds = [user['pf_speed'] for user in users]
    # 检查必要条件（不满足则评分为0）
    if min(pf_speeds) < MIN_SPEED or np.mean(pf_speeds) < AVG_SPEED:
        return 0.0
    # 计算总成本
    macro_count = sum(1 for bs in base_stations if bs['type_name'] == '宏基站')
    micro_count = sum(1 for bs in base_stations if bs['type_name'] == '微基站')
    total_cost = macro_count * MACRO_BS_COST + micro_count * MICRO_BS_COST
    if total_cost > MAX_COST or total_cost < EPS:
        return 0.0
    # 计算核心指标
    R_avg = np.mean(pf_speeds)
    R_min = min(pf_speeds)
    R_max = max(pf_speeds)
    G = (R_max - R_min) / R_avg  # 阻尼系数
    # 计算评价分数（核心公式）
    S_term = W_S * (R_avg / S_MAX)
    G_term = W_G * (1 - G) if G <= 1 else 0.0
    score = (S_term + G_term) * K / total_cost
    return score


def evaluate_deployment(users, base_stations):
    """完整部署评估（返回详细指标）"""
    users = update_all_users_pf_speed(users, base_stations)
    pf_speeds = [user['pf_speed'] for user in users]
    min_r = min(pf_speeds) if pf_speeds else 0.0
    avg_r = np.mean(pf_speeds) if pf_speeds else 0.0
    max_r = max(pf_speeds) if pf_speeds else 0.0
    # 计算总成本
    macro_count = sum(1 for bs in base_stations if bs['type_name'] == '宏基站')
    micro_count = sum(1 for bs in base_stations if bs['type_name'] == '微基站')
    total_cost = macro_count * MACRO_BS_COST + micro_count * MICRO_BS_COST
    # 检查必要条件
    meet_min = min_r >= MIN_SPEED
    meet_avg = avg_r >= AVG_SPEED
    meet_cost = total_cost <= MAX_COST
    meet_necessary = meet_min and meet_avg and meet_cost
    # 计算评价分数（核心）
    score = calculate_evaluation_score(users, base_stations) if meet_necessary else 0.0
    return {
        'score': score, 'min_r': min_r, 'avg_r': avg_r, 'max_r': max_r,
        'total_cost': total_cost, 'macro_count': macro_count, 'micro_count': micro_count,
        'meet_necessary': meet_necessary, 'pf_speeds': pf_speeds
    }


# ============================ 数据生成函数 =============================
def generate_users(num_users=100, area_range=(0, MAX_DISTANCE)):
    """生成用户数据"""
    np.random.seed(42)  # 固定随机种子
    users = []
    x_coords = np.random.uniform(area_range[0], area_range[1], num_users)
    y_coords = np.random.uniform(area_range[0], area_range[1], num_users)
    for i in range(num_users):
        users.append({
            'id': i, 'x': x_coords[i], 'y': y_coords[i],
            'instant_speed': 0.0, 'avg_speed': 0.0, 'pf_speed': 0.0
        })
    print(f"✅ 生成{num_users}个用户，分布在{area_range[0]}~{area_range[1]}m区域")
    return users


def load_users_from_csv(csv_path):
    """无表头CSV的解析逻辑（列索引：0=区域编号，1=实际X，2=实际Y）"""
    users = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # 跳过空行/注释行
            parts = line.split(',')
            if len(parts) < 3:  # 至少需要区域编号、实际X、实际Y三列
                print(f"⚠️ 跳过第{line_idx+1}行：列数不足（仅{len(parts)}列）")
                continue
            try:
                # 按列索引读取：1=实际X坐标，2=实际Y坐标
                x = float(parts[1])
                y = float(parts[2])
                x = np.clip(x, 0, MAX_DISTANCE)
                y = np.clip(y, 0, MAX_DISTANCE)
                x += np.random.uniform(-0.1, 0.1)
                y += np.random.uniform(-0.1, 0.1)
                # 完整字典（无...）
                users.append({
                    'id': line_idx,
                    'x': x,
                    'y': y,
                    'instant_speed': 0.0,
                    'avg_speed': 0.0,
                    'pf_speed': 0.0
                })
            except ValueError:
                print(f"⚠️ 跳过第{line_idx+1}行：坐标不是有效数字")
                continue
    if len(users) == 0:
        raise ValueError(f"❌ 从{csv_path}加载不到有效用户数据！")
    print(f"✅ 加载{len(users)}个用户，坐标范围：X[{min(u['x'] for u in users):.2f}, {max(u['x'] for u in users):.2f}]m，Y[{min(u['y'] for u in users):.2f}, {max(u['y'] for u in users):.2f}]m")
    return users


def create_base_station(bs_type, x, y, active=1):
    """创建基站（新增active参数）"""
    config = bs_type_config[bs_type]
    return {
        'type_name': bs_type,
        'x': np.clip(x, 0, MAX_DISTANCE),
        'y': np.clip(y, 0, MAX_DISTANCE),
        'P_t': config['P_t_default'],
        'f': freq_config[DEFAULT_FREQ],
        'B': config['B_default'],
        'n': config['n_value'],
        'id': f"{bs_type}_{np.random.randint(1000, 9999)}",  # 随机ID
        'active': active  # 激活位
    }


# ============================ 基站距离约束校验函数 =============================
def check_bs_distance_constraint(base_stations):
    """校验基站间距离约束，过滤不满足的基站（避免堆叠）"""
    valid_bs = []
    for new_bs in base_stations:
        is_valid = True
        for exist_bs in valid_bs:
            dist = np.sqrt((new_bs['x'] - exist_bs['x']) ** 2 + (new_bs['y'] - exist_bs['y']) ** 2)
            # 宏基站间约束
            if new_bs['type_name'] == '宏基站' and exist_bs['type_name'] == '宏基站':
                if dist < MIN_MACRO_DIST:
                    is_valid = False
                    break
            # 微基站间约束
            elif new_bs['type_name'] == '微基站' and exist_bs['type_name'] == '微基站':
                if dist < MIN_MICRO_DIST:
                    is_valid = False
                    break
            # 宏微基站间约束
            else:
                if dist < MIN_MACRO_MICRO_DIST:
                    is_valid = False
                    break
        if is_valid:
            valid_bs.append(new_bs)
    return valid_bs


def adjust_bs_position_to_constraint(bs, existing_bs):
    """调整基站位置以满足距离约束（修复空列表时的变量未定义问题）"""
    if not existing_bs:  # 无现有基站时直接返回原位置
        return bs
    new_x, new_y = bs['x'], bs['y']
    min_dist = float('inf')
    closest_bs = None
    # 找到最近的基站
    for exist_bs in existing_bs:
        dist = np.sqrt((new_x - exist_bs['x']) ** 2 + (new_y - exist_bs['y']) ** 2)
        if dist < min_dist:
            min_dist = dist
            closest_bs = exist_bs
    # 计算需要的最小距离
    if bs['type_name'] == '宏基站' and closest_bs['type_name'] == '宏基站':
        required_dist = MIN_MACRO_DIST
    elif bs['type_name'] == '微基站' and closest_bs['type_name'] == '微基站':
        required_dist = MIN_MICRO_DIST
    else:
        required_dist = MIN_MACRO_MICRO_DIST
    # 若距离不足，向随机方向偏移
    if min_dist < required_dist and min_dist > 0:
        offset_angle = np.random.uniform(0, 2 * np.pi)
        offset_dist = required_dist - min_dist + 10  # 额外偏移10m
        new_x = new_x + offset_dist * np.cos(offset_angle)
        new_y = new_y + offset_dist * np.sin(offset_angle)
        # 边界约束
        new_x = np.clip(new_x, 0, MAX_DISTANCE)
        new_y = np.clip(new_y, 0, MAX_DISTANCE)
    bs['x'] = new_x
    bs['y'] = new_y
    return bs


# ============================ 贪心算法核心辅助函数 =============================
def calculate_user_density_grid(users):
    """计算用户密度网格，返回密度热图和热点坐标"""
    # 生成网格
    grid_x = np.arange(0, MAX_DISTANCE + GRID_CELL_SIZE, GRID_CELL_SIZE)
    grid_y = np.arange(0, MAX_DISTANCE + GRID_CELL_SIZE, GRID_CELL_SIZE)
    density_grid = np.zeros((len(grid_y) - 1, len(grid_x) - 1))
    user_coords = np.array([[u['x'], u['y']] for u in users])
    # 统计每个网格的用户数
    for i in range(len(grid_y) - 1):
        for j in range(len(grid_x) - 1):
            x_min, x_max = grid_x[j], grid_x[j + 1]
            y_min, y_max = grid_y[i], grid_y[i + 1]
            # 筛选网格内的用户
            in_grid = np.logical_and(
                np.logical_and(user_coords[:, 0] >= x_min, user_coords[:, 0] < x_max),
                np.logical_and(user_coords[:, 1] >= y_min, user_coords[:, 1] < y_max)
            )
            density_grid[i, j] = np.sum(in_grid)
    # 找到密度最高的网格中心
    max_density_idx = np.unravel_index(np.argmax(density_grid), density_grid.shape)
    hot_x = (grid_x[max_density_idx[1]] + grid_x[max_density_idx[1] + 1]) / 2
    hot_y = (grid_y[max_density_idx[0]] + grid_y[max_density_idx[0] + 1]) / 2
    return density_grid, (hot_x, hot_y), grid_x, grid_y


def check_bs_distance(new_x, new_y, new_type, base_stations):
    """检查新基站与现有基站的距离是否满足约束"""
    if not base_stations:
        return True  # 无基站时直接通过
    for bs in base_stations:
        dist = np.sqrt((bs['x'] - new_x) ** 2 + (bs['y'] - new_y) ** 2)
        # 宏基站间约束
        if new_type == '宏基站' and bs['type_name'] == '宏基站':
            if dist < MIN_MACRO_DIST:
                return False
        # 微基站间约束
        elif new_type == '微基站' and bs['type_name'] == '微基站':
            if dist < MIN_MICRO_DIST:
                return False
        # 宏微基站间约束
        else:
            if dist < MIN_MACRO_MICRO_DIST:
                return False
    return True


def find_optimal_bs_position(target_area, base_stations, bs_type):
    """在目标区域内找满足距离约束的最优位置（避开已有基站）"""
    area_x, area_y, area_size = target_area  # (中心x, 中心y, 区域面积)
    max_attempts = 20  # 最大尝试次数
    attempt = 0
    while attempt < max_attempts:
        # 在目标区域内随机偏移
        offset = np.sqrt(area_size) / 4  # 偏移范围与区域大小正相关
        new_x = area_x + np.random.uniform(-offset, offset)
        new_y = area_y + np.random.uniform(-offset, offset)
        # 边界约束
        new_x = np.clip(new_x, 0, MAX_DISTANCE)
        new_y = np.clip(new_y, 0, MAX_DISTANCE)
        # 检查距离约束
        if check_bs_distance(new_x, new_y, bs_type, base_stations):
            return (new_x, new_y)
        attempt += 1
    # 多次尝试失败后，直接用区域中心（强制满足距离，仅作兜底）
    return (area_x, area_y)


def identify_low_speed_areas(users, base_stations):
    """识别连续低速率盲区，返回：[(中心x, 中心y, 区域面积, 平均速率), ...]"""
    # 1. 更新用户PF速率
    users = update_all_users_pf_speed(users, base_stations)
    user_coords = np.array([[u['x'], u['y']] for u in users])
    user_speeds = np.array([u['pf_speed'] for u in users])
    # 2. 网格化评估速率
    grid_x = np.arange(0, MAX_DISTANCE + GRID_CELL_SIZE, GRID_CELL_SIZE)
    grid_y = np.arange(0, MAX_DISTANCE + GRID_CELL_SIZE, GRID_CELL_SIZE)
    low_speed_grids = []  # 存储低速率网格的中心和速率
    for i in range(len(grid_y) - 1):
        for j in range(len(grid_x) - 1):
            x_min, x_max = grid_x[j], grid_x[j + 1]
            y_min, y_max = grid_y[i], grid_y[i + 1]
            # 筛选网格内的用户
            in_grid = np.logical_and(
                np.logical_and(user_coords[:, 0] >= x_min, user_coords[:, 0] < x_max),
                np.logical_and(user_coords[:, 1] >= y_min, user_coords[:, 1] < y_max)
            )
            if np.sum(in_grid) == 0:
                continue  # 无用户的网格跳过
            # 计算网格平均速率
            grid_avg_speed = np.mean(user_speeds[in_grid])
            if grid_avg_speed < AVG_SPEED:  # 低于最低速率阈值则标记为盲区
                grid_center_x = (x_min + x_max) / 2
                grid_center_y = (y_min + y_max) / 2
                grid_area = GRID_CELL_SIZE ** 2
                low_speed_grids.append((grid_center_x, grid_center_y, grid_area, grid_avg_speed))
    if not low_speed_grids:
        return []
    # 3. 聚类连续低速率网格（合并为大区域）
    low_speed_coords = np.array([[g[0], g[1]] for g in low_speed_grids])
    dbscan = DBSCAN(eps=GRID_CELL_SIZE * 1.5, min_samples=2)  # 邻域半径=1.5个网格
    clusters = dbscan.fit_predict(low_speed_coords)
    # 4. 计算每个聚类区域的中心、面积和平均速率
    low_speed_areas = []
    for cluster_id in np.unique(clusters):
        if cluster_id == -1:
            continue  # 孤立点跳过
        # 筛选该聚类的网格
        cluster_mask = clusters == cluster_id
        cluster_grids = np.array(low_speed_grids)[cluster_mask]
        # 计算区域中心
        area_x = np.mean([g[0] for g in cluster_grids])
        area_y = np.mean([g[1] for g in cluster_grids])
        # 计算区域面积（网格数×单网格面积）
        area_size = len(cluster_grids) * GRID_CELL_SIZE ** 2
        # 计算区域平均速率
        area_avg_speed = np.mean([g[3] for g in cluster_grids])
        low_speed_areas.append((area_x, area_y, area_size, area_avg_speed))
    # 按平均速率升序排序（优先优化速率最低的区域）
    low_speed_areas.sort(key=lambda x: x[3])
    return low_speed_areas


# ============================ 重新设计的贪心算法主函数 =============================
def greedy_deploy_base_stations(users, init_macro_num=INIT_MACRO_NUM):
    """
    重新设计的贪心算法：
    1. 密度驱动初始宏基站部署（覆盖人多区域）
    2. 距离约束避免基站堆叠
    3. 分区域动态选择基站类型（大面积补宏，小面积补微）
    4. 分散式微基站补充（盲区内均匀分布）
    5. 边际效益终止迭代（避免无效补充）
    """
    base_stations = []
    user_coords = np.array([[user['x'], user['y']] for user in users])
    speed_improve_history = []  # 记录最低速率提升值，用于边际效益判断
    # ========== 步骤1：用户密度驱动的初始宏基站部署 ==========
    print("📌 基于用户密度部署初始宏基站...")
    density_grid, first_hotspot, _, _ = calculate_user_density_grid(users)
    # 部署第一个宏基站（密度最高的热点）
    first_macro_x, first_macro_y = first_hotspot
    first_macro = create_base_station('宏基站', first_macro_x, first_macro_y)
    first_macro['id'] = "宏基站_密度热点_1"
    base_stations.append(first_macro)
    # 部署剩余初始宏基站（在次热点区域，满足距离约束）
    if init_macro_num > 1:
        for i in range(1, init_macro_num):
            # 找次热点区域（避开已有宏基站）
            attempt = 0
            max_attempts = 20
            while attempt < max_attempts:
                # 随机选一个非热点但有用户的位置
                random_user_idx = np.random.choice(len(users))
                candidate_x = users[random_user_idx]['x']
                candidate_y = users[random_user_idx]['y']
                if check_bs_distance(candidate_x, candidate_y, '宏基站', base_stations):
                    macro = create_base_station('宏基站', candidate_x, candidate_y)
                    macro['id'] = f"宏基站_密度热点_{i + 1}"
                    base_stations.append(macro)
                    break
                attempt += 1
    print(f"✅ 初始宏基站部署完成：共{len(base_stations)}个，覆盖用户密度热点区域")
    # ========== 步骤2：迭代补充基站 ==========
    iter_count = 0
    stagnant_count = 0  # 连续无提升次数
    macro_count = init_macro_num
    micro_count = 0
    while True:
        # 评估当前部署
        current_eval = evaluate_deployment(users, base_stations)
        current_min_speed = current_eval['min_r']
        current_avg_speed = current_eval['avg_r']
        current_cost = current_eval['total_cost']
        print(f"\n📌 贪心迭代{iter_count}：")
        print(
            f"   最低速率={current_min_speed:.2f}Mbps | 平均速率={current_avg_speed:.2f}Mbps | 成本={current_cost:.2f}万")
        print(f"   宏基站={macro_count} | 微基站={micro_count} | Score={current_eval['score']:.2f}")
        # 终止条件1：满足所有必要条件
        if current_eval['meet_necessary'] and current_eval['score'] > EPS:
            print(f"🎉 迭代{iter_count}次：满足所有速率+预算条件，停止部署")
            break
        # 终止条件2：预算用尽或基站数量达上限
        if current_cost + MICRO_BS_COST > MAX_COST:
            print("💰 预算用尽，无法继续补充基站")
            break
        if macro_count >= MAX_MACRO_NUM and micro_count >= MAX_MICRO_NUM:
            print("🔢 宏/微基站数量达上限，停止部署")
            break
        # 终止条件3：边际效益不足（连续2次提升<1Mbps）
        if len(speed_improve_history) >= 2:
            last_improve = speed_improve_history[-1]
            second_last_improve = speed_improve_history[-2]
            if last_improve < MIN_SPEED_IMPROVE and second_last_improve < MIN_SPEED_IMPROVE:
                stagnant_count += 1
                if stagnant_count >= STAGNANT_ITER:
                    print(f"📉 连续{STAGNANT_ITER}次速率提升<{MIN_SPEED_IMPROVE}Mbps，终止迭代")
                    break
            else:
                stagnant_count = 0
        # ========== 识别低速率盲区 ==========
        low_speed_areas = identify_low_speed_areas(users, base_stations)
        if not low_speed_areas:
            print("✅ 无低速率盲区，停止部署")
            break
        # 优先优化速率最低的区域
        target_area = low_speed_areas[0]
        area_x, area_y, area_size, area_avg_speed = target_area
        print(
            f"🎯 目标优化区域：中心({area_x:.2f},{area_y:.2f}) | 面积={area_size:.0f}m² | 平均速率={area_avg_speed:.2f}Mbps")
        # ========== 动态选择基站类型 ==========
        if area_size >= LARGE_AREA_THRESHOLD and macro_count < MAX_MACRO_NUM and (
                current_cost + MACRO_BS_COST) <= MAX_COST:
            # 大面积盲区：补充宏基站
            bs_type = '宏基站'
            macro_count += 1
            cost_increase = MACRO_BS_COST
        else:
            # 小面积盲区：补充微基站
            bs_type = '微基站'
            micro_count += 1
            cost_increase = MICRO_BS_COST
        # ========== 找满足距离约束的部署位置 ==========
        new_x, new_y = find_optimal_bs_position((area_x, area_y, area_size), base_stations, bs_type)
        # 创建新基站
        new_bs = create_base_station(bs_type, new_x, new_y)
        new_bs['id'] = f"{bs_type}_补充_{macro_count if bs_type == '宏基站' else micro_count}"
        base_stations.append(new_bs)
        # ========== 计算速率提升值（用于边际效益判断） ==========
        new_eval = evaluate_deployment(users, base_stations)
        speed_improve = new_eval['min_r'] - current_min_speed
        speed_improve_history.append(max(speed_improve, 0))  # 避免负提升
        print(f"✅ 补充{bs_type}：位置({new_x:.2f},{new_y:.2f}) | 速率提升={speed_improve:.2f}Mbps")
        iter_count += 1
    # 最终评估
    final_eval = evaluate_deployment(users, base_stations)
    print("\n" + "=" * 50)
    print(f"✅ 贪心算法部署完成：")
    print(f"   宏基站={final_eval['macro_count']} | 微基站={final_eval['micro_count']}")
    print(f"   最低速率={final_eval['min_r']:.2f}Mbps | 平均速率={final_eval['avg_r']:.2f}Mbps")
    print(f"   总成本={final_eval['total_cost']:.2f}万 | 最终Score={final_eval['score']:.2f}")
    print("=" * 50)
    return base_stations


# ============================ 支持宏基站新增/位置调整的GA算法 =============================
class GAOptimizer:
    """
    以Score为核心的GA算法（支持宏基站新增+宏/微基站位置自由调整）：
    1. 新增宏基站池+激活位，支持动态新增宏基站；
    2. 提高宏基站位置变异率，放开位置调整限制；
    3. 新增位置约束校验/调整，避免基站堆叠；
    4. 微基站位置变异步长增大，调整更灵活；
    5. 所有调整以最大化Score为核心目标。
    """

    def __init__(self, users, init_base_stations, pop_size=GA_POP_SIZE, max_iter=GA_MAX_ITER):
        self.users = users
        self.init_bs = init_base_stations
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.cross_rate = 0.7
        self.mutate_rate = 0.2
        # 分离初始宏/微基站（核心覆盖）
        self.base_macro = [bs for bs in init_base_stations if bs['type_name'] == '宏基站']
        self.base_micro = [bs for bs in init_base_stations if bs['type_name'] == '微基站']
        self.base_macro_num = len(self.base_macro)
        self.base_micro_num = len(self.base_micro)
        # 新增基站池配置
        self.max_macro_add = MAX_MACRO_ADD
        self.max_micro_add = MAX_MICRO_ADD
        self.total_macro_slot = self.base_macro_num + self.max_macro_add
        self.total_micro_slot = self.base_micro_num + self.max_micro_add
        # Score强化参数（核心）
        self.score_boost = 2.0  # 从1.2提升到2.0，强化优质解优先级
        self.penalty_invalid = 1e4  # 从-1e3降低到-1e4，严厉惩罚无效解
        # 记录优化过程
        self.score_history = []
        self.macro_num_history = []
        self.micro_num_history = []
        self.best_score = 0
        self.best_bs = self.init_bs

    def _get_blind_areas(self, base_stations):
        """获取速率盲区的中心坐标（用于定向变异）+ 限制在模拟范围"""
        users = update_all_users_pf_speed(self.users, base_stations)
        blind_users = [u for u in users if u['pf_speed'] < MIN_SPEED]
        if not blind_users:
            return [(np.random.uniform(0, MAX_DISTANCE), np.random.uniform(0, MAX_DISTANCE))]
        blind_x = [u['x'] for u in blind_users]
        blind_y = [u['y'] for u in blind_users]
        # 核心修正：裁剪盲区中心坐标到模拟范围
        blind_center_x = np.clip(np.mean(blind_x), 0, MAX_DISTANCE)
        blind_center_y = np.clip(np.mean(blind_y), 0, MAX_DISTANCE)
        blind_coords = np.array([[blind_center_x, blind_center_y]])
        if len(blind_coords) <= 3:
            return [(blind_center_x, blind_center_y)]
        kmeans = KMeans(n_clusters=min(3, len(blind_coords)), random_state=42)
        kmeans.fit(blind_coords)
        # 裁剪聚类后的中心坐标
        centers = []
        for center in kmeans.cluster_centers_:
            cx = np.clip(center[0], 0, MAX_DISTANCE)
            cy = np.clip(center[1], 0, MAX_DISTANCE)
            centers.append((cx, cy))
        return centers

    def _encode(self, base_stations):
        """
        编码函数（支持宏基站新增）：
        - 宏基站：基础宏基站 + 新增宏基站池（每个6维：x,y,P_t,B,active,n）
        - 微基站：基础微基站 + 新增微基站池（每个6维：x,y,P_t,B,active,n）
        所有参数归一化到[0,1]
        """
        # 分离并补全宏基站（基础+新增池）
        macro_bs = [bs for bs in base_stations if bs['type_name'] == '宏基站']
        while len(macro_bs) < self.total_macro_slot:
            # 新增宏基站池：初始位置随机，激活位0
            rand_x = np.random.uniform(0, MAX_DISTANCE)
            rand_y = np.random.uniform(0, MAX_DISTANCE)
            dummy_macro = create_base_station('宏基站', rand_x, rand_y, active=0)
            macro_bs.append(dummy_macro)
        macro_bs = macro_bs[:self.total_macro_slot]
        # 分离并补全微基站（基础+新增池）
        micro_bs = [bs for bs in base_stations if bs['type_name'] == '微基站']
        while len(micro_bs) < self.total_micro_slot:
            # 新增微基站池：初始位置随机，激活位0
            rand_x = np.random.uniform(0, MAX_DISTANCE)
            rand_y = np.random.uniform(0, MAX_DISTANCE)
            dummy_micro = create_base_station('微基站', rand_x, rand_y, active=0)
            micro_bs.append(dummy_micro)
        micro_bs = micro_bs[:self.total_micro_slot]
        # 编码宏基站
        macro_code = []
        macro_config = bs_type_config['宏基站']
        p_min, p_max = macro_config['P_t_range']
        b_min, b_max = macro_config['B_range']
        for bs in macro_bs:
            macro_code.append(bs['x'] / MAX_DISTANCE)
            macro_code.append(bs['y'] / MAX_DISTANCE)
            macro_code.append((bs['P_t'] - p_min) / (p_max - p_min))
            macro_code.append((bs['B'] - b_min) / (b_max - b_min))
            macro_code.append(bs['active'])  # 激活位
            macro_code.append(0)  # 路径损耗指数（固定，占位）
        # 编码微基站
        micro_code = []
        micro_config = bs_type_config['微基站']
        p_min, p_max = micro_config['P_t_range']
        b_min, b_max = micro_config['B_range']
        for bs in micro_bs:
            micro_code.append(bs['x'] / MAX_DISTANCE)
            micro_code.append(bs['y'] / MAX_DISTANCE)
            micro_code.append((bs['P_t'] - p_min) / (p_max - p_min))
            micro_code.append((bs['B'] - b_min) / (b_max - b_min))
            micro_code.append(bs['active'])  # 激活位
            micro_code.append(0)  # 路径损耗指数（固定，占位）
        return np.array(macro_code + micro_code, dtype=np.float32)

    def _decode(self, chrom):
        """解码函数（支持宏基站新增+位置调整）+ 强化坐标约束 + 修复n_value未定义"""
        base_stations = []
        # 计算编码维度
        macro_dim = self.total_macro_slot * 6
        micro_dim = self.total_micro_slot * 6
        # 解码宏基站
        macro_code = chrom[:macro_dim]
        macro_config = bs_type_config['宏基站']
        p_min, p_max = macro_config['P_t_range']
        b_min, b_max = macro_config['B_range']
        n_value = macro_config['n_value']  # 核心修复：提前定义宏基站的n_value
        for i in range(self.total_macro_slot):
            idx = i * 6
            x = macro_code[idx] * MAX_DISTANCE
            y = macro_code[idx + 1] * MAX_DISTANCE
            # 核心修正：解码后再次裁剪
            x = np.clip(x, 0, MAX_DISTANCE)
            y = np.clip(y, 0, MAX_DISTANCE)
            P_t = macro_code[idx + 2] * (p_max - p_min) + p_min
            B = macro_code[idx + 3] * (b_max - b_min) + b_min
            active = 1 if macro_code[idx + 4] > 0.5 else 0  # 激活位阈值
            if active:
                # 创建宏基站并调整位置以满足约束
                macro_bs = create_base_station('宏基站', x, y, active)
                macro_bs['P_t'] = P_t
                macro_bs['B'] = B
                macro_bs['n'] = n_value  # 现在n_value已定义，不会报错
                # 位置约束调整
                if len(base_stations) > 0:
                    macro_bs = adjust_bs_position_to_constraint(macro_bs, base_stations)
                # 最终裁剪（双重保险）
                macro_bs['x'] = np.clip(macro_bs['x'], 0, MAX_DISTANCE)
                macro_bs['y'] = np.clip(macro_bs['y'], 0, MAX_DISTANCE)
                base_stations.append(macro_bs)
        # 解码微基站（同步检查n_value定义）
        micro_code = chrom[macro_dim:macro_dim + micro_dim]
        micro_config = bs_type_config['微基站']
        p_min, p_max = micro_config['P_t_range']
        b_min, b_max = micro_config['B_range']
        n_value = micro_config['n_value']  # 同步修复：定义微基站的n_value
        for i in range(self.total_micro_slot):
            idx = i * 6
            x = micro_code[idx] * MAX_DISTANCE
            y = micro_code[idx + 1] * MAX_DISTANCE
            # 核心修正：解码后裁剪
            x = np.clip(x, 0, MAX_DISTANCE)
            y = np.clip(y, 0, MAX_DISTANCE)
            P_t = micro_code[idx + 2] * (p_max - p_min) + p_min
            B = micro_code[idx + 3] * (b_max - b_min) + b_min
            active = 1 if micro_code[idx + 4] > 0.5 else 0  # 激活位阈值
            if active:
                # 创建微基站并调整位置以满足约束
                micro_bs = create_base_station('微基站', x, y, active)
                micro_bs['P_t'] = P_t
                micro_bs['B'] = B
                micro_bs['n'] = n_value  # 微基站n_value已定义
                # 位置约束调整
                if len(base_stations) > 0:
                    micro_bs = adjust_bs_position_to_constraint(micro_bs, base_stations)
                # 最终裁剪（双重保险）
                micro_bs['x'] = np.clip(micro_bs['x'], 0, MAX_DISTANCE)
                micro_bs['y'] = np.clip(micro_bs['y'], 0, MAX_DISTANCE)
                base_stations.append(micro_bs)
        # 最终校验距离约束，过滤无效基站
        base_stations = check_bs_distance_constraint(base_stations)
        return base_stations

    def _fitness(self, chrom):
        """适应度函数（严格以Score为唯一优化标准，增强选择压力）"""
        base_stations = self._decode(chrom)
        eval_res = evaluate_deployment(self.users, base_stations)
        score = eval_res['score']
        # 辅助奖励/惩罚（不稀释Score核心）
        if score > EPS:
            fitness = score * self.score_boost
        else:
            fitness = -self.penalty_invalid
        return max(fitness, 0)

    def _tournament_selection(self, pop, fitness, k=3):
        """锦标赛选择"""
        selected = []
        for _ in range(self.pop_size):
            candidates = np.random.choice(len(pop), k, replace=False)
            best_candidate = candidates[np.argmax(fitness[candidates])]
            selected.append(pop[best_candidate])
        return np.array(selected)

    def _layered_crossover(self, parent1, parent2):
        """分层交叉（宏/微基站分开交叉）"""
        if np.random.random() >= self.cross_rate:
            return parent1, parent2
        macro_dim = self.total_macro_slot * 6
        child1 = parent1.copy()
        child2 = parent2.copy()
        # 宏基站段交叉（单点）
        if macro_dim > 1:
            cross_idx = np.random.randint(1, macro_dim - 1)
            child1[:cross_idx] = parent2[:cross_idx]
            child2[:cross_idx] = parent1[:cross_idx]
        # 微基站段交叉（多点）
        micro_code = parent1[macro_dim:]
        micro_len = len(micro_code)
        if micro_len > 0:
            cross_points = np.random.choice(micro_len, size=max(1, micro_len // 10), replace=False)
            for idx in cross_points:
                child1[macro_dim + idx] = parent2[macro_dim + idx]
                child2[macro_dim + idx] = parent1[macro_dim + idx]
        return child1, child2

    def _directed_mutation(self, chrom):
        """定向变异（支持宏/微基站新增/删除/大幅移动，以Score为导向）"""
        new_chrom = chrom.copy()
        macro_dim = self.total_macro_slot * 6
        blind_centers = self._get_blind_areas(self._decode(chrom))
        blind_x, blind_y = blind_centers[0]
        blind_x_norm = blind_x / MAX_DISTANCE
        blind_y_norm = blind_y / MAX_DISTANCE
        # 1. 宏基站变异（位置+激活+参数）
        for i in range(self.total_macro_slot):
            idx = i * 6
            # 位置变异：大幅移动（步长0.5）+ 随机扰动（幅度0.1）
            if np.random.random() < self.mutate_rate * MACRO_MUTATE_RATIO:
                # 向盲区偏移
                new_chrom[idx] += (blind_x_norm - new_chrom[idx]) * MACRO_POS_STEP
                new_chrom[idx] += np.random.normal(0, 0.1)  # 扰动幅度翻倍
            if np.random.random() < self.mutate_rate * MACRO_MUTATE_RATIO:
                new_chrom[idx + 1] += (blind_y_norm - new_chrom[idx + 1]) * MACRO_POS_STEP
                new_chrom[idx + 1] += np.random.normal(0, 0.1)  # 随机扰动
            # 参数变异（功率/带宽）
            if np.random.random() < self.mutate_rate:
                new_chrom[idx + 2] += np.random.normal(0, 0.1)
                new_chrom[idx + 3] += np.random.normal(0, 0.1)
            # 激活位变异：支持新增（0→1）和删除（1→0），概率提升到0.8
            if np.random.random() < self.mutate_rate * 0.8:
                new_chrom[idx + 4] = 1.0 if new_chrom[idx + 4] < 0.5 else 0.0
            # 边界约束
            for j in range(5):
                new_chrom[idx + j] = np.clip(new_chrom[idx + j], 0, 1)
        # 2. 微基站变异（位置+激活+参数，更大步长）
        for i in range(self.total_micro_slot):
            idx = macro_dim + i * 6
            # 位置变异：大幅移动（步长0.6）+ 随机扰动
            if np.random.random() < self.mutate_rate:
                new_chrom[idx] += (blind_x_norm - new_chrom[idx]) * MICRO_POS_STEP
                new_chrom[idx] += np.random.normal(0, 0.1)
            if np.random.random() < self.mutate_rate:
                new_chrom[idx + 1] += (blind_y_norm - new_chrom[idx + 1]) * MICRO_POS_STEP
                new_chrom[idx + 1] += np.random.normal(0, 0.1)
            # 参数变异
            if np.random.random() < self.mutate_rate:
                new_chrom[idx + 2] += np.random.normal(0, 0.1)
                new_chrom[idx + 3] += np.random.normal(0, 0.1)
            # 激活位变异：支持新增/删除，概率提升到0.9
            if np.random.random() < self.mutate_rate * 0.9:
                new_chrom[idx + 4] = 1.0 if new_chrom[idx + 4] < 0.5 else 0.0
            # 边界约束
            for j in range(5):
                new_chrom[idx + j] = np.clip(new_chrom[idx + j], 0, 1)
        return new_chrom

    def optimize(self):
        """GA主流程（加入精英保留，强化Score优化）"""
        print(f"\n📌 GA初始化（支持宏基站新增+删除+大幅移动）：")
        print(f"   基础宏基站{self.base_macro_num}个 | 可新增宏基站{self.max_macro_add}个")
        print(f"   基础微基站{self.base_micro_num}个 | 可新增微基站{self.max_micro_add}个")
        print(f"   种群规模={self.pop_size} | 迭代次数={self.max_iter} | 宏基站位置变异率={MACRO_MUTATE_RATIO}")
        # 初始化种群
        init_chrom = self._encode(self.init_bs)
        pop = []
        for _ in range(self.pop_size):
            chrom = init_chrom + np.random.normal(0, 0.05, len(init_chrom))
            chrom = np.clip(chrom, 0, 1)
            pop.append(chrom)
        pop = np.array(pop)
        # 迭代优化
        for iter in range(self.max_iter):
            fitness = np.array([self._fitness(chrom) for chrom in pop])
            current_best_idx = np.argmax(fitness)
            current_best_chrom = pop[current_best_idx].copy()
            current_best_bs = self._decode(current_best_chrom)
            current_eval = evaluate_deployment(self.users, current_best_bs)
            current_score = current_eval['score']
            current_macro_num = current_eval['macro_count']
            current_micro_num = current_eval['micro_count']
            # 更新全局最优
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_bs = current_best_bs
            # 记录过程
            self.score_history.append(current_score)
            self.macro_num_history.append(current_macro_num)
            self.micro_num_history.append(current_micro_num)
            # 进化操作
            pop = self._tournament_selection(pop, fitness)
            new_pop = []
            for i in range(0, self.pop_size, 2):
                parent1 = pop[i]
                parent2 = pop[i + 1] if i + 1 < self.pop_size else pop[i]
                child1, child2 = self._layered_crossover(parent1, parent2)
                new_pop.append(self._directed_mutation(child1))
                new_pop.append(self._directed_mutation(child2))
            # 关键修正：精英保留——保留当前代最优个体，避免优质解丢失
            new_pop[0] = current_best_chrom  # 第1个位置固定为当前最优
            pop = np.array(new_pop[:self.pop_size])
            # 打印进度（每20次迭代打印一次，减少输出）
            if (iter + 1) % 20 == 0:
                print(
                    f"🔧 GA迭代{iter + 1}/{self.max_iter} | 最优Score={current_score:.2f} | 宏基站={current_macro_num}个 | 微基站={current_micro_num}个")
        # 最终评估
        final_eval = evaluate_deployment(self.users, self.best_bs)
        print("\n" + "=" * 50)
        print(f"✅ GA优化完成（支持宏基站新增+删除+移动）：")
        print(
            f"   贪心宏基站={self.base_macro_num}个 → GA宏基站={final_eval['macro_count']}个（变化{final_eval['macro_count'] - self.base_macro_num}个）")
        print(
            f"   贪心微基站={self.base_micro_num}个 → GA微基站={final_eval['micro_count']}个（变化{final_eval['micro_count'] - self.base_micro_num}个）")
        print(
            f"   最低速率={final_eval['min_r']:.2f}Mbps | 平均速率={final_eval['avg_r']:.2f}Mbps | 成本={final_eval['total_cost']:.2f}万")
        print(
            f"   贪心Score={calculate_evaluation_score(self.users, self.init_bs):.2f} → GA Score={self.best_score:.2f}（提升{self.best_score / max(calculate_evaluation_score(self.users, self.init_bs), EPS):.2f}倍）")
        print("=" * 50)
        return self.best_bs


# ============================ 可视化模块（完整修复版）============================
def visualize_results(users, greedy_bs, optimal_bs, ga_optimizer=None):
    """可视化结果：修复箭头绘制、图例重复等bug，强化宏基站变化展示 + 统一坐标范围"""
    greedy_eval = evaluate_deployment(users, greedy_bs)
    optimal_eval = evaluate_deployment(users, optimal_bs)
    greedy_r = greedy_eval['pf_speeds']
    optimal_r = optimal_eval['pf_speeds']
    greedy_score = greedy_eval['score']
    optimal_score = optimal_eval['score']
    greedy_macro = greedy_eval['macro_count']
    optimal_macro = optimal_eval['macro_count']
    greedy_micro = greedy_eval['micro_count']
    optimal_micro = optimal_eval['micro_count']
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(
        f'5G基站排布优化结果（GA迭代{GA_MAX_ITER}次）| 贪心Score={greedy_score:.2f} → GA Score={optimal_score:.2f}',
        fontsize=18, fontweight='bold', y=0.98)
    # 子图1：贪心部署 - 用户/基站分布（强制坐标范围）
    ax1 = axes[0, 0]
    user_x = [u['x'] for u in users]
    user_y = [u['y'] for u in users]
    scatter1 = ax1.scatter(user_x, user_y, c=greedy_r, cmap='viridis', s=60, alpha=0.8, label='用户')
    # 绘制贪心基站
    macro_x_g = [bs['x'] for bs in greedy_bs if bs['type_name'] == '宏基站']
    macro_y_g = [bs['y'] for bs in greedy_bs if bs['type_name'] == '宏基站']
    micro_x_g = [bs['x'] for bs in greedy_bs if bs['type_name'] == '微基站']
    micro_y_g = [bs['y'] for bs in greedy_bs if bs['type_name'] == '微基站']
    ax1.scatter(macro_x_g, macro_y_g, c='crimson', s=250, marker='^', edgecolors='black', linewidth=2,
                label=f'宏基站（{greedy_macro}个）')
    ax1.scatter(micro_x_g, micro_y_g, c='orange', s=180, marker='s', edgecolors='black', linewidth=2,
                label=f'微基站（{greedy_micro}个）')
    ax1.set_xlabel('X坐标 (m)', fontsize=12)
    ax1.set_ylabel('Y坐标 (m)', fontsize=12)
    ax1.set_title(f'贪心算法部署结果', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    plt.colorbar(scatter1, ax=ax1, label='用户PF速率 (Mbps)', shrink=0.8)
    ax1.grid(alpha=0.3, linestyle='--')
    # 核心修正：强制坐标范围为模拟范围 [0, MAX_DISTANCE]
    ax1.set_xlim(0, MAX_DISTANCE)
    ax1.set_ylim(0, MAX_DISTANCE)

    # 子图2：GA优化部署 - 速率热力图+基站位置变化（统一坐标范围）
    ax2 = axes[0, 1]
    # 生成速率热力图（基于模拟范围，而非用户坐标）
    grid_size = 40
    x_grid = np.linspace(0, MAX_DISTANCE, grid_size)
    y_grid = np.linspace(0, MAX_DISTANCE, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    dummy_user = {'x': 0, 'y': 0, 'instant_speed': 0, 'avg_speed': 0, 'pf_speed': 0}
    for i in range(grid_size):
        for j in range(grid_size):
            dummy_user['x'] = X[i, j]
            dummy_user['y'] = Y[i, j]
            Z[i, j] = calculate_user_instant_speed(dummy_user['x'], dummy_user['y'], optimal_bs)
    # 绘制热力图（降低透明度，避免覆盖用户/基站）
    contour = ax2.contourf(X, Y, Z, cmap='plasma', levels=25, antialiased=True, alpha=0.6)
    # 绘制用户点（GA优化后的速率）
    ax2.scatter(user_x, user_y, c=optimal_r, cmap='viridis', s=60, alpha=0.8, label='用户')
    # 绘制GA优化后的基站
    macro_x_o = [bs['x'] for bs in optimal_bs if bs['type_name'] == '宏基站']
    macro_y_o = [bs['y'] for bs in optimal_bs if bs['type_name'] == '宏基站']
    micro_x_o = [bs['x'] for bs in optimal_bs if bs['type_name'] == '微基站']
    micro_y_o = [bs['y'] for bs in optimal_bs if bs['type_name'] == '微基站']
    ax2.scatter(macro_x_o, macro_y_o, c='crimson', s=250, marker='^', edgecolors='black', linewidth=2,
                label=f'宏基站（{optimal_macro}个）')
    ax2.scatter(micro_x_o, micro_y_o, c='orange', s=180, marker='s', edgecolors='black', linewidth=2,
                label=f'微基站（{optimal_micro}个）')
    # 绘制宏基站位置变化箭头（修复图例重复）
    min_macro = min(len(macro_x_g), len(macro_x_o))
    arrow_label = '宏基站移动'
    for i in range(min_macro):
        ax2.arrow(macro_x_g[i], macro_y_g[i], macro_x_o[i] - macro_x_g[i], macro_y_o[i] - macro_y_g[i],
                  head_width=5, head_length=8, fc='lime', ec='darkgreen', linewidth=2, alpha=0.8,
                  label=arrow_label if i == 0 else "")
        arrow_label = ""  # 仅第一个箭头显示图例
    ax2.set_xlabel('X坐标 (m)', fontsize=12)
    ax2.set_ylabel('Y坐标 (m)', fontsize=12)
    ax2.set_title(f'GA优化部署结果（速率热力图）', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    plt.colorbar(contour, ax=ax2, label='速率热力 (Mbps)', shrink=0.8)
    ax2.grid(alpha=0.3, linestyle='--')
    # 核心修正：强制坐标范围为模拟范围
    ax2.set_xlim(0, MAX_DISTANCE)
    ax2.set_ylim(0, MAX_DISTANCE)

    # 子图3：GA迭代Score进化曲线（无修改）
    ax3 = axes[1, 0]
    if ga_optimizer and len(ga_optimizer.score_history) > 0:
        iterations = range(len(ga_optimizer.score_history))
        ax3.plot(iterations, ga_optimizer.score_history, c='navy', linewidth=2.5, marker='.', markersize=4,
                 label='Score进化')
        ax3.axhline(y=greedy_score, c='red', linestyle='--', linewidth=2, label=f'贪心Score({greedy_score:.2f})')
        ax3.axhline(y=optimal_score, c='green', linestyle='-.' , linewidth=2, label=f'GA最优Score({optimal_score:.2f})')
        ax3.set_xlabel('GA迭代次数', fontsize=12)
        ax3.set_ylabel('评价Score', fontsize=12)
        ax3.set_title('GA优化Score进化曲线', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(alpha=0.3, linestyle='--')
    else:
        ax3.text(0.5, 0.5, '无GA迭代数据', ha='center', va='center', fontsize=14, transform=ax3.transAxes)
        ax3.set_xlabel('GA迭代次数', fontsize=12)
        ax3.set_ylabel('评价Score', fontsize=12)
        ax3.set_title('GA优化Score进化曲线', fontsize=14, fontweight='bold')

    # 子图4：贪心vs GA速率分布箱线图（无修改）
    ax4 = axes[1, 1]
    data = [greedy_r, optimal_r]
    labels = [f'贪心算法\n(均值{np.mean(greedy_r):.2f}Mbps)', f'GA优化\n(均值{np.mean(optimal_r):.2f}Mbps)']
    bp = ax4.boxplot(data, labels=labels, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', edgecolor='navy', linewidth=2),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='navy', linewidth=2),
                     capprops=dict(color='navy', linewidth=2))
    ax4.axhline(y=MIN_SPEED, c='orange', linestyle='--', linewidth=2, label=f'最低速率阈值({MIN_SPEED}Mbps)')
    ax4.axhline(y=AVG_SPEED, c='purple', linestyle='--', linewidth=2, label=f'平均速率阈值({AVG_SPEED}Mbps)')
    ax4.set_ylabel('用户PF速率 (Mbps)', fontsize=12)
    ax4.set_title('用户速率分布对比（箱线图）', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3, linestyle='--', axis='y')

    # 保存图片
    plt.tight_layout()
    plt.savefig('5G基站优化结果.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================ 主函数执行逻辑 =============================
def main():
    """主函数：生成用户→贪心部署→GA优化→可视化结果"""
    print("=" * 60)
    print("🚀 5G基站智能部署优化系统启动")
    print("=" * 60)
    # 1. 生成用户数据（可替换为load_users_from_csv("user_data.csv")）
    users = load_users_from_csv(r"C:\Users\Lenovo\Desktop\多区域独立生点结果_20251124_204323.csv")
    # 2. 贪心算法部署基站
    greedy_bs = greedy_deploy_base_stations(users, init_macro_num=INIT_MACRO_NUM)
    # 3. GA算法优化基站部署
    ga_optimizer = GAOptimizer(users, greedy_bs, pop_size=GA_POP_SIZE, max_iter=GA_MAX_ITER)
    optimal_bs = ga_optimizer.optimize()
    # 4. 可视化结果
    visualize_results(users, greedy_bs, optimal_bs, ga_optimizer)
    # 5. 输出最终结论
    final_greedy = evaluate_deployment(users, greedy_bs)
    final_ga = evaluate_deployment(users, optimal_bs)
    print("\n" + "=" * 60)
    print("📊 最终优化结论")
    print("=" * 60)
    print(f"贪心算法：Score={final_greedy['score']:.2f} | 成本={final_greedy['total_cost']:.2f}万 | 平均速率={final_greedy['avg_r']:.2f}Mbps")
    print(f"GA算法：Score={final_ga['score']:.2f} | 成本={final_ga['total_cost']:.2f}万 | 平均速率={final_ga['avg_r']:.2f}Mbps")
    print(f"Score提升：{(final_ga['score'] - final_greedy['score']) / max(final_greedy['score'], EPS) * 100:.2f}%")
    print("=" * 60)

    # 6. 输出基站详细信息到result.txt
    def output_base_stations_to_file(greedy_bs, optimal_bs, filename="result.txt"):
        """将基站详细信息输出到文本文件"""
        with open(filename, "w", encoding="utf-8") as f:  # 使用写入模式打开文件 [[1]]
            # 贪心算法部署的基站
            f.write("# 贪心算法部署的基站\n")
            f.write("类型,X坐标,Y坐标,ID,功率(dBm),带宽(MHz)\n")
            for bs in greedy_bs:
                # 使用f-string格式化输出 [[2]]
                f.write(
                    f"{bs['type_name']},{bs['x']:.2f},{bs['y']:.2f},{bs['id']},{10 * np.log10(bs['P_t'] * 1000):.2f},{bs['B']:.2f}\n")

            f.write("\n# GA优化后的基站\n")
            f.write("类型,X坐标,Y坐标,ID,功率(dBm),带宽(MHz)\n")
            for bs in optimal_bs:
                f.write(
                    f"{bs['type_name']},{bs['x']:.2f},{bs['y']:.2f},{bs['id']},{10 * np.log10(bs['P_t'] * 1000):.2f},{bs['B']:.2f}\n")

        print(f"\n✅ 基站详细信息已输出到 {filename}")

    # 调用输出函数
    output_base_stations_to_file(greedy_bs, optimal_bs)


if __name__ == "__main__":
    main()
