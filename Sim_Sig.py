import numpy as np
import matplotlib
matplotlib.use('TkAgg') # 指定后端
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, RadioButtons, Button
from matplotlib.colors import LinearSegmentedColormap
import win32api # 检测CapsLock状态（Windows系统）

# ========== 字体与渲染优化 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 0.01
matplotlib.rcParams['figure.dpi'] = 120
matplotlib.rcParams['savefig.dpi'] = 200
matplotlib.rcParams['text.antialiased'] = True
matplotlib.rcParams['lines.antialiased'] = True

# ===================== 核心参数 =====================
N_0 = 10 ** ((-174 - 30) / 10) # 噪声功率谱密度 (W/Hz)
L_0 = 70 # 参考路径损耗 (dB)
GRID_SIZE = 200 # 网格精度
MAX_DISTANCE = 2000 # 最大模拟距离 (m)
MIN_DISTANCE = 1 # 最小计算距离 (m)，也作为计算理论容量的参考距离
COLORBAR_MIN = 0 # 速率下限

# 比例公平算法参数
ALPHA = 0.15 # 指数移动平均权重
EPS = 1e-6 # 避免除零

# ===================== 全局变量 =====================
users = [] # 用户列表：{'x','y','instant_speed','avg_speed','pf_speed'}
base_stations = [] # 基站参数列表
current_bs_type = '宏基站' # 默认基站类型
fig = None
ax_plot = None
cbar_ax = None
radio_type = None
slider_P_t_add = None
slider_f_add = None
slider_B_add = None
slider_P_t_edit = None
slider_f_edit = None
slider_B_edit = None
selected_bs_idx = -1
selected_user_idx = -1
CLICK_THRESHOLD = 100
USER_CLICK_THRESHOLD = 50

# ===================== 基站参数映射 =====================
bs_type_config = {
    '宏基站': {
        'P_t_range': (100, 400),
        'n_value': 3, # 路径损耗指数
        'B_range': (100, 1000),
        'B_options': [100, 200, 400, 800, 1000]
    },
    '微基站': {
        'P_t_range': (5, 20),
        'n_value': 4,
        'B_range': (100, 200),
        'B_options': [100, 200]
    }
}

# 频段配置
freq_config = {3.5: 3.5e9, 28: 28e9}
FREQ_OPTIONS = [3.5, 28]
P_T_STEP = 1
F_STEP = 0.1
B_STEP = 1

# ===================== 辅助函数 =====================
def is_capslock_on():
    return win32api.GetKeyState(0x14) & 1

def get_closest_option(value, options):
    return min(options, key=lambda x: abs(x - value))

def round_by_step(value, step):
    return round(value / step) * step

# 高对比度配色方案
def create_high_contrast_cmap():
    colors = [
        (0.00, 0.0, 1.0), (0.10, 0.2, 1.0), (0.20, 0.4, 0.9), (0.30, 0.0, 0.8),
        (0.40, 0.5, 0.8), (0.50, 0.8, 0.8), (0.60, 1.0, 0.5), (0.70, 1.0, 0.2),
        (0.80, 1.0, 0.0), (0.90, 0.8, 0.0), (0.95, 0.6, 0.0), (1.00, 0.4, 0.0),
    ]
    return LinearSegmentedColormap.from_list('HighContrastSpeed', colors, N=500)

DETAILED_CMAP = create_high_contrast_cmap()

# ===================== 核心计算函数 =====================
def calculate_base_total_capacity():
    """计算系统理论总容量 (在路径损耗为L0时的距离计算)"""
    theoretical_capacity = 0.0
    for bs in base_stations:
        bs_P_t = bs['P_t']
        bs_B = bs['B'] * 1e6
        bs_n = bs['n']

        # 参考距离：路径损耗仅为 L0 时的距离，即 MIN_DISTANCE (1米)
        ref_distance = MIN_DISTANCE

        # 路径损耗为 L0
        L_d = L_0
        P_t_dBm = 10 * np.log10(bs_P_t * 1000)
        P_r_dBm = P_t_dBm - L_d
        P_r = 10 ** (P_r_dBm / 10) / 1000

        noise = N_0 * bs_B
        SNR = P_r / noise if noise != 0 else 0
        # 香农公式：C = B * log2(1 + SNR)
        bs_theoretical_capacity = bs_B * np.log2(1 + SNR) / 1e6
        theoretical_capacity += bs_theoretical_capacity

    return max(theoretical_capacity, COLORBAR_MIN)

def calculate_user_instant_speed(user_x, user_y):
    """计算单个用户的瞬时速率"""
    instant_speed = 0.0
    if not base_stations:
        return instant_speed

    for bs in base_stations:
        bs_x, bs_y = bs['x'], bs['y']
        bs_P_t, bs_n = bs['P_t'], bs['n']
        bs_B = bs['B'] * 1e6

        distance = np.sqrt((user_x - bs_x) ** 2 + (user_y - bs_y) ** 2)
        distance = max(distance, MIN_DISTANCE)

        L_d = L_0 + 10 * bs_n * np.log10(distance)
        P_t_dBm = 10 * np.log10(bs_P_t * 1000)
        P_r_dBm = P_t_dBm - L_d
        P_r = 10 ** (P_r_dBm / 10) / 1000

        noise = N_0 * bs_B
        SNR = P_r / noise if noise != 0 else 0
        speed = bs_B * np.log2(1 + SNR) / 1e6
        instant_speed += speed

    return max(instant_speed, COLORBAR_MIN)

def update_all_users_pf_speed():
    """比例公平调度算法"""
    if not base_stations or len(users) == 0:
        return

    base_total_capacity = calculate_base_total_capacity()
    if base_total_capacity < EPS:
        for user in users:
            user['pf_speed'] = 0
        return

    user_priorities = []
    for user in users:
        user['instant_speed'] = calculate_user_instant_speed(user['x'], user['y'])
        instant = user['instant_speed']
        avg = user['avg_speed']

        # 指数移动平均更新平均速率
        if avg < EPS:
            user['avg_speed'] = instant
        else:
            user['avg_speed'] = ALPHA * instant + (1 - ALPHA) * avg

        # 计算PF优先级
        priority = instant / max(user['avg_speed'], EPS)
        user_priorities.append(priority)

    total_priority = sum(user_priorities)
    if total_priority < EPS:
        equal_share = base_total_capacity / len(users)
        for user in users:
            user['pf_speed'] = max(min(equal_share, user['instant_speed']), COLORBAR_MIN)
        return

    # 按优先级分配速率
    for i, user in enumerate(users):
        pf_speed_unconstrained = (user_priorities[i] / total_priority) * base_total_capacity
        # 分配速率不能超过瞬时速率
        user['pf_speed'] = max(min(pf_speed_unconstrained, user['instant_speed']), COLORBAR_MIN)

def calculate_speed_vectorized():
    """计算瞬时速率网格"""
    x = np.linspace(0, MAX_DISTANCE, GRID_SIZE)
    y = np.linspace(0, MAX_DISTANCE, GRID_SIZE)
    X, Y = np.meshgrid(x, y)
    instant_grid = np.zeros_like(X)

    if not base_stations:
        return X, Y, instant_grid

    for bs in base_stations:
        bs_x, bs_y = bs['x'], bs['y']
        bs_P_t, bs_n = bs['P_t'], bs['n']
        bs_B = bs['B'] * 1e6

        distance = np.sqrt((X - bs_x) ** 2 + (Y - bs_y) ** 2)
        distance = np.maximum(distance, MIN_DISTANCE)

        L_d = L_0 + 10 * bs_n * np.log10(distance)
        P_t_dBm = 10 * np.log10(bs_P_t * 1000)
        P_r_dBm = P_t_dBm - L_d
        P_r = 10 ** (P_r_dBm / 10) / 1000

        noise = N_0 * bs_B
        SNR = P_r / noise if noise != 0 else 0
        speed = bs_B * np.log2(1 + SNR) / 1e6
        instant_grid += speed

    instant_grid = np.maximum(instant_grid, COLORBAR_MIN)
    return X, Y, instant_grid

# ===================== 绘图与交互 =====================
def update_plot():
    """刷新绘图"""
    ax_plot.cla()

    X, Y, instant_grid = calculate_speed_vectorized()
    user_count = len(users)
    caps_state = "开启" if is_capslock_on() else "关闭"

    # 绘制瞬时速率热力图
    max_instant = np.max(instant_grid) if np.max(instant_grid) > EPS else 1.0
    level_step = max_instant / 80 if max_instant > EPS else 0.01
    STATIC_LEVELS = np.arange(COLORBAR_MIN, max_instant + level_step, level_step)

    contour_filled = ax_plot.contourf(
        X, Y, instant_grid, levels=STATIC_LEVELS,
        cmap=DETAILED_CMAP,
        antialiased=True,
        extend='max',
        linewidths=0.1
    )

    # 绘制用户标记
    if users:
        for i, user in enumerate(users):
            u_x, u_y = user['x'], user['y']
            pf_speed = user['pf_speed']
            instant_speed = user['instant_speed']

            if i == selected_user_idx:
                ax_plot.scatter(u_x, u_y, c='yellow', s=200, marker='o', edgecolors='red', zorder=5, linewidth=3)
                label = f'用户{i}\n瞬时：{instant_speed:.1f} Mbps\nPF实际：{pf_speed:.1f} Mbps'
                ax_plot.text(u_x + 30, u_y + 30, label, fontsize=11, color='black', weight='bold',
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.9), zorder=6)
            else:
                ax_plot.scatter(u_x, u_y, c='lime', s=140, marker='o', edgecolors='black', zorder=4, linewidth=2)

        ax_plot.legend([f'用户({user_count}) - PF实际速率'], loc='upper left', fontsize=12, framealpha=0.9, shadow=True)

    # 坐标轴与标题
    ax_plot.set_aspect('equal', adjustable='box')
    ax_plot.set_xlim(0, MAX_DISTANCE)
    ax_plot.set_ylim(0, MAX_DISTANCE)
    ax_plot.set_title(
        f"华为5G网络强度分布（高对比度无裁切） | CapsLock{caps_state}：左键加{'用户' if is_capslock_on() else '基站'} | 用户数：{user_count}",
        fontsize=14, pad=15, weight='bold'
    )
    ax_plot.set_xlabel("距离 (m)", fontsize=15, labelpad=5)
    ax_plot.set_ylabel("距离 (m)", fontsize=15, labelpad=5)
    ax_plot.tick_params(axis='both', labelsize=13)

    # 颜色条
    cbar_ax.clear()
    cbar = fig.colorbar(contour_filled, cax=cbar_ax, extend='max')
    cbar_ticks = np.linspace(COLORBAR_MIN, max_instant, min(10, len(STATIC_LEVELS) // 8))
    cbar.set_ticks(cbar_ticks)
    cbar.set_label(f'网络强度（瞬时速率 Mbps）| 实际最大值：{max_instant:.1f} Mbps', fontsize=13, labelpad=5)
    cbar.ax.tick_params(labelsize=12)

    fig.canvas.draw_idle()

def find_nearest_node(x, y):
    """查找距离点击位置最近的用户/基站"""
    min_dist = float('inf')
    nearest_type = None
    nearest_idx = -1

    for i, user in enumerate(users):
        u_x, u_y = user['x'], user['y']
        dist = np.sqrt((u_x - x) ** 2 + (u_y - y) ** 2)
        if dist < USER_CLICK_THRESHOLD and dist < min_dist:
            min_dist = dist
            nearest_type = 'user'
            nearest_idx = i

    if nearest_idx == -1:
        for i, bs in enumerate(base_stations):
            dist = np.sqrt((bs['x'] - x) ** 2 + (bs['y'] - y) ** 2)
            if dist < CLICK_THRESHOLD and dist < min_dist:
                min_dist = dist
                nearest_type = 'base'
                nearest_idx = i

    return nearest_type, nearest_idx

def on_click(event):
    """鼠标单击事件：添加基站/用户 + 选中节点"""
    global selected_bs_idx, selected_user_idx

    if event.inaxes != ax_plot or event.xdata is None or event.ydata is None:
        return

    click_x, click_y = event.xdata, event.ydata

    if event.button == 1:
        if is_capslock_on(): # 添加用户
            new_user = {'x': click_x, 'y': click_y, 'instant_speed': 0, 'avg_speed': 0, 'pf_speed': 0}
            users.append(new_user)
            update_all_users_pf_speed()
            print(f"✅ 添加用户{len(users) - 1}，位置({click_x:.1f},{click_y:.1f})，当前用户数：{len(users)}")
        else: # 添加基站
            config = bs_type_config[current_bs_type]
            P_t = round_by_step(slider_P_t_add.val, P_T_STEP)
            f_GHz = get_closest_option(round_by_step(slider_f_add.val, F_STEP), FREQ_OPTIONS)
            B = get_closest_option(round_by_step(slider_B_add.val, B_STEP), config['B_options'])
            f = freq_config[f_GHz]
            n = config['n_value']

            new_bs = {'type_name': current_bs_type, 'x': click_x, 'y': click_y, 'P_t': P_t, 'f': f, 'B': B, 'n': n}
            base_stations.append(new_bs)
            update_all_users_pf_speed()
            print(f"✅ 添加【{current_bs_type}】，位置({click_x:.1f},{click_y:.1f})，功率{P_t}W，频率{f_GHz}GHz，带宽{B}MHz")

        update_plot()
        return

    elif event.button == 3: # 右键：选中节点
        selected_bs_idx = -1
        selected_user_idx = -1

        nearest_type, nearest_idx = find_nearest_node(click_x, click_y)

        if nearest_type == 'user':
            selected_user_idx = nearest_idx
            user = users[selected_user_idx]
            print(f"✅ 选中用户{selected_user_idx}，位置({user['x']:.1f},{user['y']:.1f})")
            print(f" 瞬时速率：{user['instant_speed']:.1f} Mbps | 平均速率：{user['avg_speed']:.1f} Mbps | PF分配速率：{user['pf_speed']:.1f} Mbps")
        elif nearest_type == 'base':
            selected_bs_idx = nearest_idx
            selected_bs = base_stations[selected_bs_idx]
            slider_P_t_edit.set_val(selected_bs['P_t'])
            slider_f_edit.set_val(selected_bs['f'] / 1e9)
            slider_B_edit.set_val(selected_bs['B'])
            config = bs_type_config[selected_bs['type_name']]
            slider_P_t_edit.valmin = config['P_t_range'][0]
            slider_P_t_edit.valmax = config['P_t_range'][1]
            print(f"✅ 选中【{selected_bs['type_name']}】(索引{selected_bs_idx})，位置({click_x:.1f},{click_y:.1f})")
        else:
            print("⚠️ 右键位置无基站/用户！")

        update_plot()

def on_bs_type_change(label):
    """切换基站类型"""
    global current_bs_type
    current_bs_type = label
    config = bs_type_config[label]

    slider_P_t_add.valmin = config['P_t_range'][0]
    slider_P_t_add.valmax = config['P_t_range'][1]
    slider_P_t_add.set_val((config['P_t_range'][0] + config['P_t_range'][1]) / 2)

    slider_B_add.valmin = config['B_range'][0]
    slider_B_add.valmax = config['B_range'][1]
    slider_B_add.set_val(config['B_options'][0])

    print(f"✅ 切换为【{label}】，华为参数：功率{config['P_t_range']}W，带宽{config['B_range']}MHz，n={config['n_value']}")

def on_edit_bs(event):
    """修改选中基站参数"""
    global selected_bs_idx
    if selected_bs_idx == -1:
        print("⚠️ 请先右键选中一个基站！")
        return

    bs = base_stations[selected_bs_idx]
    config = bs_type_config[bs['type_name']]

    P_t = round_by_step(slider_P_t_edit.val, P_T_STEP)
    f_GHz = get_closest_option(round_by_step(slider_f_edit.val, F_STEP), FREQ_OPTIONS)
    B = get_closest_option(round_by_step(slider_B_edit.val, B_STEP), config['B_options'])
    f = freq_config[f_GHz]

    bs['P_t'] = P_t
    bs['f'] = f
    bs['B'] = B

    print(f"✏️ 修改【{bs['type_name']}】(索引{selected_bs_idx})，新参数：功率{P_t}W，频率{f_GHz}GHz，带宽{B}MHz")

    update_all_users_pf_speed()
    update_plot()

def on_del_bs(event):
    """删除选中的基站/用户"""
    global selected_bs_idx, selected_user_idx
    if selected_user_idx != -1:
        del users[selected_user_idx]
        print(f"❌ 删除用户{selected_user_idx}，当前用户数：{len(users)}")
        selected_user_idx = -1
    elif selected_bs_idx != -1:
        del base_stations[selected_bs_idx]
        print(f"❌ 删除基站(索引{selected_bs_idx})")
        selected_bs_idx = -1
    else:
        print("⚠️ 请先右键选中一个基站/用户！")

    update_all_users_pf_speed()
    update_plot()

# ===================== 主程序 =====================
if __name__ == "__main__":
    fig = plt.figure(figsize=(18, 12))

    main_gs = gridspec.GridSpec(2, 1, height_ratios=[9, 1], hspace=0.05)
    plot_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=main_gs[0], width_ratios=[10, 1], wspace=0)
    ax_plot = fig.add_subplot(plot_gs[0, 0])
    cbar_ax = fig.add_subplot(plot_gs[0, 1])

    ctrl_gs = gridspec.GridSpecFromSubplotSpec(1, 9, subplot_spec=main_gs[1], wspace=0.5)
    ax_type = fig.add_subplot(ctrl_gs[0, 0])
    ax_btn_edit = fig.add_subplot(ctrl_gs[0, 1])
    ax_btn_del = fig.add_subplot(ctrl_gs[0, 2])
    ax_P_t_add = fig.add_subplot(ctrl_gs[0, 3])
    ax_f_add = fig.add_subplot(ctrl_gs[0, 4])
    ax_B_add = fig.add_subplot(ctrl_gs[0, 5])
    ax_P_t_edit = fig.add_subplot(ctrl_gs[0, 6])
    ax_f_edit = fig.add_subplot(ctrl_gs[0, 7])
    ax_B_edit = fig.add_subplot(ctrl_gs[0, 8])

    radio_type = RadioButtons(ax_type, ['宏基站', '微基站'], active=0)
    radio_type.on_clicked(on_bs_type_change)
    for label in radio_type.labels:
        label.set_fontsize(10)

    btn_edit = Button(ax_btn_edit, '修改', color='lightblue', hovercolor='blue')
    btn_del = Button(ax_btn_del, '删除', color='lightcoral', hovercolor='red')
    btn_edit.label.set_fontsize(10)
    btn_del.label.set_fontsize(10)
    btn_edit.on_clicked(on_edit_bs)
    btn_del.on_clicked(on_del_bs)

    init_config = bs_type_config['宏基站']
    slider_P_t_add = Slider(ax_P_t_add, '加功率', init_config['P_t_range'][0], init_config['P_t_range'][1], valinit=200, valstep=1)
    slider_f_add = Slider(ax_f_add, '加频率', 3.5, 28, valinit=3.5, valstep=0.1)
    slider_B_add = Slider(ax_B_add, '加带宽', 100, 1000, valinit=100, valstep=1)
    slider_P_t_edit = Slider(ax_P_t_edit, '改功率', init_config['P_t_range'][0], init_config['P_t_range'][1], valinit=150, valstep=1)
    slider_f_edit = Slider(ax_f_edit, '改频率', 3.5, 28, valinit=3.5, valstep=0.1)
    slider_B_edit = Slider(ax_B_edit, '改带宽', 100, 1000, valinit=100, valstep=1)

    for slider in [slider_P_t_add, slider_f_add, slider_B_add, slider_P_t_edit, slider_f_edit, slider_B_edit]:
        slider.label.set_fontsize(9)
        slider.valtext.set_fontsize(8)

    fig.canvas.mpl_connect('button_press_event', on_click)

    update_plot()
    plt.show(block=True)