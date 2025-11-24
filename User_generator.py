import numpy as np
import matplotlib

# 强制设置交互式后端，确保UI窗口弹出（优先TkAgg，Python自带）
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import csv
import os
from datetime import datetime
import sys
import subprocess
import platform


# 检查必要库是否安装
def check_dependencies():
    required_libs = ['numpy', 'matplotlib', 'Pillow']
    missing_libs = []
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing_libs.append(lib)

    if missing_libs:
        print(f"缺失依赖库: {', '.join(missing_libs)}")
        print("正在自动安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_libs],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("依赖库安装完成！")


# 检查依赖
check_dependencies()


class MultiAreaPointGenerator:
    def __init__(self):
        self.bg_image = None
        self.current_polygon = []  # 存储当前正在绘制的多边形顶点
        self.polygons = []  # 存储所有已确认的多边形（多区域核心）
        self.area_point_counts = []  # 存储每个区域对应的生成点数（核心新增：区域独立点数）
        self.reference_points = []
        self.origin_point = None
        self.scale_factor = 1.0
        self.generated_points = []  # 格式：[(区域编号, 像素x, 像素y), ...]
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        # 窗口标题强调区域独立点数
        self.fig.canvas.manager.set_window_title('区域独立生点器 - 点击窗口获取焦点 (退格=完成, 回车=确认)')
        self.fig.suptitle('多区域独立生点器 (每个区域可设置不同点数!)', fontsize=14, color='darkgreen')
        # 标记各阶段状态
        self.phase = 'polygon'  # polygon → reference → origin → generate
        # Matplotlib支持的单字符颜色缩写
        self.polygon_color_codes = ['r', 'b', 'g', 'orange', 'm']  # 单字符/短名称颜色
        self.polygon_fill_colors = ['red', 'blue', 'green', 'orange', 'purple']  # 填充用完整颜色名
        self._bind_events()
        # 强制窗口置顶+刷新事件循环
        self._bring_window_to_front()
        self.fig.canvas.draw_idle()  # 强制画布刷新

    def _bind_events(self):
        """绑定所有有效事件（全局监听，不屏蔽任何阶段）"""
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self._on_key_global)
        self.cid_draw = self.fig.canvas.mpl_connect('draw_event', lambda e: None)

    def _bring_window_to_front(self):
        """强制窗口置顶（不同系统适配）"""
        try:
            if platform.system() == 'Windows':
                self.fig.canvas.manager.window.attributes('-topmost', 1)
                self.fig.canvas.manager.window.attributes('-topmost', 0)
            elif platform.system() == 'Darwin':  # macOS
                self.fig.canvas.manager.window.raise_()
            else:  # Linux
                self.fig.canvas.manager.window.attributes('-topmost', True)
                self.fig.canvas.manager.window.attributes('-topmost', False)
        except:
            pass

    def _on_click(self, event):
        """统一处理鼠标点击事件（按阶段分发）"""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        if self.phase == 'polygon':
            self._handle_polygon_click(event)
        elif self.phase == 'reference':
            self._handle_reference_click(event)
        elif self.phase == 'origin':
            self._handle_origin_click(event)

    def _on_key_global(self, event):
        """全局按键处理（所有阶段都响应）"""
        if self.phase == 'polygon':
            self._handle_polygon_keys(event)
        elif self.phase == 'reference':
            self._handle_reference_keys(event)
        elif self.phase == 'origin':
            self._handle_origin_keys(event)
        elif self.phase == 'generate':
            return

    # -------------------------- 多边形阶段（核心修改：区域独立点数输入） --------------------------
    def _handle_polygon_click(self, event):
        """绘制当前多边形顶点"""
        self.current_polygon.append((event.xdata, event.ydata))
        color_idx = len(self.polygons) % len(self.polygon_color_codes)
        line_color = self.polygon_color_codes[color_idx]
        # 绘制顶点和连线
        self.ax.plot(event.xdata, event.ydata, marker='o', color=line_color, markersize=6)
        if len(self.current_polygon) > 1:
            self.ax.plot([self.current_polygon[-2][0], self.current_polygon[-1][0]],
                         [self.current_polygon[-2][1], self.current_polygon[-1][1]],
                         color=line_color, linewidth=2, linestyle='-')
        # 更新状态提示
        tip = f'[绘制阶段] 第{len(self.polygons) + 1}个区域 - 已选{len(self.current_polygon)}个顶点 | '
        tip += 'ESC=撤销 | 回车=确认区域并输入点数 | 退格=完成所有区域绘制'
        self.ax.set_title(tip, fontsize=9)
        self.fig.canvas.draw_idle()

    def _handle_polygon_keys(self, event):
        """多边形阶段按键：ESC=撤销，Enter=确认区域+输入点数，Backspace=完成绘制"""
        # ESC：撤销最后一个顶点
        if event.key == 'escape' and len(self.current_polygon) > 0:
            self.current_polygon.pop()
            self.ax.clear()
            if self.bg_image:
                self.ax.imshow(self.bg_image, alpha=0.5)
            # 重绘所有已确认的多边形
            self._redraw_all_polygons()
            # 重绘当前未确认的多边形
            color_idx = len(self.polygons) % len(self.polygon_color_codes)
            line_color = self.polygon_color_codes[color_idx]
            for i, (x, y) in enumerate(self.current_polygon):
                self.ax.plot(x, y, marker='o', color=line_color, markersize=6)
                if i > 0:
                    self.ax.plot([self.current_polygon[i - 1][0], x], [self.current_polygon[i - 1][1], y],
                                 color=line_color, linewidth=2)
            # 更新提示
            tip = f'[绘制阶段] 第{len(self.polygons) + 1}个区域 - 剩余{len(self.current_polygon)}个顶点 | '
            tip += 'ESC=撤销 | 回车=确认区域并输入点数 | 退格=完成所有区域绘制'
            self.ax.set_title(tip, fontsize=9)
            self.fig.canvas.draw_idle()

        # Enter：确认区域并输入该区域的独立点数（核心修改）
        elif event.key == 'enter' and len(self.current_polygon) >= 3:
            # 让用户输入该区域的生成点数
            while True:
                try:
                    point_count = int(input(f"\n请输入第{len(self.polygons) + 1}个区域的生成点数: "))
                    if point_count > 0:
                        break
                    print("错误：点数必须是大于0的整数！")
                except ValueError:
                    print("错误：请输入有效的整数！")
            # 保存区域和对应的点数
            self.polygons.append(self.current_polygon.copy())
            self.area_point_counts.append(point_count)
            # 绘制已确认的多边形（带填充）
            color_idx = (len(self.polygons) - 1) % len(self.polygon_fill_colors)
            fill_color = self.polygon_fill_colors[color_idx]
            line_color = self.polygon_color_codes[color_idx]
            self.ax.add_patch(
                Polygon(self.current_polygon, fill=True, color=fill_color, alpha=0.2, edgecolor=line_color,
                        linewidth=2))
            # 在画布上标注区域编号和点数
            self._annotate_area_info(len(self.polygons), self.current_polygon, fill_color, point_count)
            # 清空当前多边形，准备绘制下一个
            self.current_polygon.clear()
            # 更新提示
            tip = f'[绘制阶段] ✅ 已确认第{len(self.polygons)}个区域 (点数：{point_count}) | 点击绘制下一个区域或按退格完成'
            self.ax.set_title(tip, fontsize=10, color='green')
            self.fig.canvas.draw_idle()
            print(f"\n✅ 已确认第{len(self.polygons)}个区域，将在该区域生成{point_count}个点！")

        # Backspace：完成所有区域绘制（需至少1个区域）
        elif event.key == 'backspace' and len(self.polygons) >= 1:
            if len(self.current_polygon) > 0:
                print(f"\n提示：放弃未完成的第{len(self.polygons) + 1}个区域（{len(self.current_polygon)}个顶点）")
                self.current_polygon.clear()
            # 输出所有区域的点数信息
            print("\n📋 已确认的区域列表：")
            for i in range(len(self.polygons)):
                print(f"   第{i + 1}个区域 → 生成{self.area_point_counts[i]}个点")
            # 进入参考线阶段
            self.phase = 'reference'
            tip = '[参考线阶段] 点击2个点设置参考线 | 选完后按回车确认'
            self.ax.set_title(tip, fontsize=10, color='blue')
            self.fig.canvas.draw_idle()
            print("\n🔄 进入参考线阶段！请在窗口中点击2个点作为参考线，然后按回车确认")

    # -------------------------- 参考线阶段（逻辑不变） --------------------------
    def _handle_reference_click(self, event):
        """参考线端点选择"""
        if len(self.reference_points) < 2:
            self.reference_points.append((event.xdata, event.ydata))
            self.ax.plot(event.xdata, event.ydata, marker='o', color='g', markersize=8,
                         label='参考点' if len(self.reference_points) == 1 else "")
            if len(self.reference_points) == 2:
                self.ax.plot([self.reference_points[0][0], self.reference_points[1][0]],
                             [self.reference_points[0][1], self.reference_points[1][1]],
                             color='g', linewidth=3, linestyle='-', label='参考线')
                self.ax.legend(loc='upper left', fontsize=8)
            # 更新提示
            tip = f'[参考线阶段] 已选{len(self.reference_points)}/2个点 | 按回车确认参考线'
            self.ax.set_title(tip, fontsize=10, color='blue')
            self.fig.canvas.draw_idle()

    def _handle_reference_keys(self, event):
        """参考线阶段按键：仅处理Enter"""
        if event.key == 'enter' and len(self.reference_points) == 2:
            # 计算参考线像素长度
            x1, y1 = self.reference_points[0]
            x2, y2 = self.reference_points[1]
            pixel_dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # 输入实际长度
            while True:
                try:
                    actual_len = float(input(f"\n参考线像素长度: {pixel_dist:.2f}\n请输入参考线的实际长度（米）: "))
                    if actual_len > 0:
                        self.scale_factor = actual_len / pixel_dist
                        print(f"✅ 比例因子计算完成：1像素 = {self.scale_factor:.6f} 米")
                        break
                    print("错误：长度必须大于0！")
                except ValueError:
                    print("错误：请输入有效的数字！")
            # 进入原点阶段
            self.phase = 'origin'
            tip = '[原点阶段] 点击1个点设置坐标原点 | 选完后按回车确认并开始生点'
            self.ax.set_title(tip, fontsize=10, color='purple')
            self.fig.canvas.draw_idle()
            print("\n🔄 进入原点阶段！请在窗口中点击1个点作为坐标原点，然后按回车开始生点")

    # -------------------------- 原点阶段（逻辑不变，生点触发修改） --------------------------
    def _handle_origin_click(self, event):
        """坐标原点选择"""
        if self.origin_point is None:
            self.origin_point = (event.xdata, event.ydata)
            self.ax.plot(event.xdata, event.ydata, marker='*', color='b', markersize=12, label='原点')
            self.ax.axhline(y=event.ydata, color='blue', linestyle='--', alpha=0.7)
            self.ax.axvline(x=event.xdata, color='blue', linestyle='--', alpha=0.7)
            self.ax.legend(loc='upper left', fontsize=8)
            # 更新提示
            tip = f'[原点阶段] 原点已设置在({event.xdata:.2f}, {event.ydata:.2f}) | 按回车开始生点'
            self.ax.set_title(tip, fontsize=10, color='purple')
            self.fig.canvas.draw_idle()

    def _handle_origin_keys(self, event):
        """原点阶段按键：仅处理Enter（开始生点）"""
        if event.key == 'enter' and self.origin_point is not None:
            self.phase = 'generate'
            self._generate_points()  # 生点逻辑核心修改
            self._export_csv()  # 导出逻辑核心修改

    # -------------------------- 核心工具函数（新增/修改） --------------------------
    def _annotate_area_info(self, area_id, polygon, color, point_count):
        """在区域中心标注编号和点数（新增）"""
        # 计算多边形中心
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        center_x = (min(xs) + max(xs)) / 2
        center_y = (min(ys) + max(ys)) / 2
        # 标注区域信息
        self.ax.text(center_x, center_y, f'区域{area_id}\n{point_count}个点',
                     ha='center', va='center', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.5))

    def _is_point_in_polygon(self, point, polygon):
        """判断点是否在指定的单个多边形内（核心修改：替代原任意多边形判断）"""
        x, y = point
        inside = False
        n = len(polygon)
        for i in range(n):
            p1x, p1y = polygon[i]
            p2x, p2y = polygon[(i + 1) % n]
            if ((p1y > y) != (p2y > y)):
                x_inter = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if x < x_inter:
                    inside = not inside
        return inside

    def _generate_points(self):
        """在每个区域内独立生成指定数量的点（核心修改）"""
        self.ax.set_title('[生点阶段] 正在为每个区域生成点...', fontsize=10, color='orange')
        self.fig.canvas.draw_idle()
        print("\n🚀 开始为每个区域生成点...")

        # 遍历每个区域，独立生成点
        for area_id in range(len(self.polygons)):
            polygon = self.polygons[area_id]
            point_count = self.area_point_counts[area_id]
            generated = 0
            print(f"   正在为第{area_id + 1}个区域生成{point_count}个点...")

            # 获取该区域的边界（仅在该区域内随机，提高效率）
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # 生成指定数量的点
            while generated < point_count:
                x_rand = np.random.uniform(x_min, x_max)
                y_rand = np.random.uniform(y_min, y_max)
                if self._is_point_in_polygon((x_rand, y_rand), polygon):
                    self.generated_points.append((area_id + 1, x_rand, y_rand))  # 记录区域编号
                    generated += 1

            # 绘制该区域的点（用对应区域的颜色）
            color = self.polygon_color_codes[area_id % len(self.polygon_color_codes)]
            area_points = [p for p in self.generated_points if p[0] == area_id + 1]
            x_coords = [p[1] for p in area_points]
            y_coords = [p[2] for p in area_points]
            self.ax.scatter(x_coords, y_coords, color=color, s=20, alpha=0.7,
                            label=f'区域{area_id + 1}({point_count}个点)')

        # 更新状态和图例
        self.ax.legend(loc='upper right', fontsize=8)
        total_points = sum(self.area_point_counts)
        self.ax.set_title(f'[完成] ✅ 生点完成！共{len(self.polygons)}个区域，总计{total_points}个点', fontsize=10,
                          color='darkgreen')
        self.fig.canvas.draw_idle()
        print(f"\n✅ 生点完成！总计生成{total_points}个点（分布在{len(self.polygons)}个区域）")

    def _export_csv(self):
        """导出CSV（核心修改：增加区域编号列）"""
        if not self.generated_points:
            return
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'多区域独立生点结果_{timestamp}.csv'
        # 转换坐标：(像素坐标 - 原点) × 比例因子
        ox, oy = self.origin_point
        # 写入CSV
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:  # utf-8-sig解决Excel中文乱码
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(['区域编号', '像素X', '像素Y', '实际X坐标（米）', '实际Y坐标（米）'])
            # 写入数据
            for area_id, x, y in self.generated_points:
                real_x = (x - ox) * self.scale_factor
                real_y = (y - oy) * self.scale_factor
                writer.writerow([area_id, round(x, 2), round(y, 2), round(real_x, 6), round(real_y, 6)])
        # 输出保存信息
        save_path = os.path.abspath(filename)
        self.ax.text(0.5, -0.15, f'CSV已保存至：{save_path}', transform=self.ax.transAxes, ha='center', fontsize=8,
                     color='blue')
        self.fig.canvas.draw_idle()
        print(f"\n💾 CSV文件已保存至：{save_path}")
        print("   CSV包含：区域编号、像素坐标、实际米制坐标（基于参考线和原点校准）")

    def _redraw_all_polygons(self):
        """重新绘制所有已确认的多边形（带点数标注）"""
        for i, poly in enumerate(self.polygons):
            color_idx = i % len(self.polygon_fill_colors)
            fill_color = self.polygon_fill_colors[color_idx]
            line_color = self.polygon_color_codes[color_idx]
            self.ax.add_patch(Polygon(poly, fill=True, color=fill_color, alpha=0.2, edgecolor=line_color, linewidth=2))
            # 重新标注区域信息
            self._annotate_area_info(i + 1, poly, fill_color, self.area_point_counts[i])

    def load_background(self):
        """加载背景图片（支持中文路径）"""
        while True:
            img_path = input("请输入背景图片路径（直接回车则无背景）: ").strip()
            if not img_path:
                tip = '[绘制阶段] 点击绘制第一个区域（≥3个顶点）| 回车=确认并输入点数 | 退格=完成'
                self.ax.set_title(tip, fontsize=9)
                self.fig.canvas.draw_idle()
                return
            try:
                # 解决PIL中文路径问题
                if platform.system() == 'Windows':
                    from PIL import ImageFile
                    ImageFile.LOAD_TRUNCATED_IMAGES = True
                    img_path = img_path.encode('gbk').decode('utf-8', 'ignore')
                self.bg_image = Image.open(img_path)
                self.ax.imshow(self.bg_image, alpha=0.5)
                tip = '[绘制阶段] 背景图加载完成 | 点击绘制第一个区域（≥3个顶点）| 回车=确认并输入点数 | 退格=完成'
                self.ax.set_title(tip, fontsize=9)
                self.fig.canvas.draw_idle()
                return
            except Exception as e:
                print(f"加载图片失败：{str(e)}，请重新输入！")

    def run(self):
        """启动程序（全中文提示）"""
        print("=" * 80)
        print("      多区域独立生点器 v6.0 - 每个区域可设置不同生成点数（最终版）")
        print("=" * 80)
        print("📢 核心操作说明（必看）：")
        print("  1. 程序启动后，绘图窗口置顶弹出 → 先点击窗口内部获取键盘焦点")
        print("  2. 绘制区域：点击≥3个顶点 → 按【回车】→ 输入该区域的生成点数 → 确认")
        print("  3. 重复步骤2，可绘制N个区域，每个区域设置不同点数")
        print("  4. 完成绘制：按【退格键】→ 查看区域列表 → 进入参考线阶段")
        print("  5. 参考线：点击2个点 → 按回车 → 输入实际长度（米）")
        print("  6. 原点：点击1个点设为原点 → 按回车 → 自动为每个区域生点")
        print("  7. 结果：自动导出CSV，包含区域编号、像素坐标、实际米制坐标")
        print("=" * 80)
        self.load_background()
        # 强制显示窗口
        plt.show(block=True)


if __name__ == "__main__":
    # 解决Windows中文输出乱码
    if platform.system() == 'Windows':
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    try:
        app = MultiAreaPointGenerator()
        app.run()
    except KeyboardInterrupt:
        print("\n\n程序被用户手动中断！")
    except Exception as e:
        print(f"\n\n程序运行出错：{str(e)}")
        input("按Enter键退出...")