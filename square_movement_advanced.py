#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json, math, os, sys, threading, time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import rospy
from geometry_msgs.msg import Twist

try:  # 可选依赖：缺失时节点照常运行，相关功能自动关闭并提示
    from std_msgs.msg import Bool, String
    _HAS_STD_MSGS = True
except ImportError:
    _HAS_STD_MSGS = False
try:
    from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse
    _HAS_STD_SRVS = True
except ImportError:
    _HAS_STD_SRVS = False
try:
    from nav_msgs.msg import Odometry
    _HAS_NAV_MSGS = True
except ImportError:
    _HAS_NAV_MSGS = False
try:
    from sensor_msgs.msg import BatteryState, LaserScan
    _HAS_SENSOR_MSGS = True
except ImportError:
    _HAS_SENSOR_MSGS = False


DEFAULT_NODE_NAME = "square_movement_node"
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

# 时间比较容差（秒）。任何"等到某个时刻为止"的循环都必须带容差：
# deadline - now 会留下浮点残差（实测约 1e-15），一旦残差小于时钟分辨率，
# 循环就会永远等不到终点而空转；在 /use_sim_time 仿真时钟下这是真会卡死的。
TIME_EPS = 1e-3


class RobotState:
    """状态机取值。"""
    IDLE = "idle"           # 空闲，等待 start
    RUNNING = "running"     # 正在执行图形
    PAUSED = "paused"       # 暂停（手动或安全原因）
    STOPPING = "stopping"   # 正在减速停车
    FINISHED = "finished"   # 本轮任务正常完成
    FAULT = "fault"         # 故障（急停、堵转、超时等）
    SHUTDOWN = "shutdown"   # 节点正在退出


class MotionKind(Enum):
    """原子运动类型。"""
    DRIVE = "drive"   # 直行 value 米
    TURN = "turn"     # 原地旋转 value 度
    ARC = "arc"       # 弧线：半径 radius、圆心角 value 度
    WAIT = "wait"     # 原地等待 value 秒


class Direction:
    """方向常量：直行用 FORWARD/BACKWARD，转向用 LEFT/RIGHT。"""
    FORWARD = 1
    BACKWARD = -1
    LEFT = 1      # 逆时针，angular.z > 0
    RIGHT = -1    # 顺时针，angular.z < 0（老脚本的右转）


class SafetyAction:
    """遇到障碍时的策略：停车等待 / 跳过当前步骤 / 只告警不停车。"""
    STOP = "stop"
    SKIP = "skip"
    IGNORE = "ignore"


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def clamp(value: float, low: float, high: float) -> float:
    """把 value 限制在 [low, high]。"""
    if low > high:
        low, high = high, low
    return max(low, min(high, value))


def wrap_angle(angle_rad: float) -> float:
    """弧度角归一化到 (-pi, pi]，避免角度累加越界。"""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def wrap_deg(angle_deg: float) -> float:
    """角度归一化到 (-180, 180]。"""
    return math.degrees(wrap_angle(math.radians(angle_deg)))


def quaternion_to_yaw(q) -> float:
    """四元数 -> 偏航角(弧度)，自己算，不依赖 tf/tf2。"""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def ros_seconds_now() -> float:
    """当前 ROS 时间(秒)；仿真下跟随 /clock，比 time.time() 可靠。"""
    return rospy.Time.now().to_sec()


def format_seconds(seconds: float) -> str:
    """秒数 -> 易读字符串，用于日志。"""
    if seconds < 0:
        return "n/a"
    minutes, sec = divmod(seconds, 60.0)
    return "%.1fs" % sec if minutes <= 0 else "%dm%.1fs" % (minutes, sec)


def ensure_directory(path: str) -> bool:
    """确保目录存在，返回是否可用。"""
    try:
        if path and not os.path.isdir(path):
            os.makedirs(path)
        return bool(path)
    except OSError as exc:  # pragma: no cover
        rospy.logwarn("创建目录 %s 失败: %s", path, exc)
        return False


def get_private_param(name: str, default):
    """读取私有参数 ~name，失败时返回默认值。"""
    try:
        return rospy.get_param("~" + name, default)
    except Exception as exc:  # pragma: no cover
        rospy.logwarn("读取参数 ~%s 失败(%s)，使用默认值 %s", name, exc, default)
        return default


# ---------------------------------------------------------------------------
# PID 控制器
# ---------------------------------------------------------------------------
class PIDController:
    """一维 PID：误差=剩余距离/角度，输出=速度。带积分限幅、微分低通、输出限幅。"""

    def __init__(self, kp, ki=0.0, kd=0.0, output_limit=None, integral_limit=None,
                 derivative_lpf=0.6):
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.output_limit, self.integral_limit = output_limit, integral_limit
        self.derivative_lpf = clamp(float(derivative_lpf), 0.0, 1.0)
        self._integral = self._last_error = self._last_derivative = 0.0
        self._first_run = True

    def reset(self) -> None:
        """清空内部状态（每段新运动开始前调用）。"""
        self._integral = self._last_error = self._last_derivative = 0.0
        self._first_run = True

    def update(self, error: float, dt: float) -> float:
        """按误差与时间间隔计算控制量。"""
        dt = dt if dt > 1e-6 else 1e-3
        self._integral += error * dt
        if self.integral_limit is not None:  # anti-windup
            self._integral = clamp(self._integral, -self.integral_limit, self.integral_limit)
        if self._first_run:
            derivative, self._first_run = 0.0, False
        else:
            derivative = (error - self._last_error) / dt
        self._last_derivative = (self.derivative_lpf * self._last_derivative
                                 + (1.0 - self.derivative_lpf) * derivative)
        self._last_error = error
        output = self.kp * error + self.ki * self._integral + self.kd * self._last_derivative
        return clamp(output, -self.output_limit, self.output_limit) if self.output_limit else output


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class MotionConfig:
    """
    全部可调参数。默认值刻意与老脚本一致（0.12m/s、0.22rad/s、3.5s、7.6s、右转），
    所以不加任何参数直接运行，行为等价于老脚本，只是更安全、更可观测、可中断。
    """

    # 话题
    publish_topic: str = "/cmd_vel"
    odom_topic: str = "/odom"
    scan_topic: str = "/scan"
    battery_topic: str = "/battery_state"
    estop_topic: str = "~/estop"
    teleop_topic: str = "~/teleop_override"
    status_topic: str = "~/status"
    shape_command_topic: str = "~/shape_command"
    # 速度与加速度
    forward_speed: float = 0.12       # 直行速度 m/s（老脚本 forward_speed）
    back_speed: float = 0.10          # 后退速度 m/s
    turn_speed: float = 0.22          # 原地转向角速度 rad/s（老脚本 turn_speed）
    arc_speed: float = 0.12           # 弧线线速度 m/s
    max_linear_speed: float = 0.50    # 安全上限 m/s
    max_angular_speed: float = 1.50   # 安全上限 rad/s
    max_linear_accel: float = 0.25    # 线加速度上限 m/s^2（速度平滑）
    max_angular_accel: float = 0.60   # 角加速度上限 rad/s^2
    min_linear_speed: float = 0.04    # 闭环最小线速度（更小电机可能带不动）
    min_angular_speed: float = 0.08   # 闭环最小角速度
    # 闭环控制
    use_odom: bool = True
    odom_timeout: float = 1.0         # 里程计超时即视为失效
    drive_kp: float = 1.20            # 距离环增益
    drive_ki: float = 0.10
    drive_kd: float = 0.02
    turn_kp: float = 1.00             # 角度环增益
    turn_ki: float = 0.05
    turn_kd: float = 0.01
    position_tolerance: float = 0.010  # 距离死区 m
    angle_tolerance_deg: float = 1.5   # 角度死区 度
    # 图形与尺寸
    shape: str = "square"
    laps: int = 1
    loop_forever: bool = False
    side_length: float = 0.42          # 正方形边长 = 老脚本 0.12m/s x 3.5s
    rectangle_length: float = 0.60
    rectangle_width: float = 0.40
    polygon_sides: int = 5
    polygon_side_length: float = 0.40
    star_arm: float = 0.50             # 五角星单臂长度
    circle_radius: float = 0.30
    figure8_radius: float = 0.30
    spiral_start_radius: float = 0.15
    spiral_step: float = 0.08
    spiral_turns: int = 3
    line_length: float = 1.00
    rotate_angle_deg: float = 360.0
    turn_after_last_edge: bool = False  # 老脚本最后一条边后不再转弯
    drive_direction: str = "forward"    # forward / backward
    turn_direction: str = "cw"          # cw=右转（老脚本）/ ccw=左转
    # 安全
    use_scan: bool = True
    obstacle_distance: float = 0.35           # 前向刹车距离 m
    obstacle_release_distance: float = 0.50   # 障碍释放距离（滞回）
    obstacle_fov_deg: float = 60.0            # 前向检测扇区宽度
    obstacle_action: str = SafetyAction.STOP
    scan_timeout: float = 1.0
    obstacle_wait_timeout: float = 0.0        # >0 时等待超时则中止任务
    use_battery: bool = True
    min_battery_percent: float = 15.0
    estop_auto_clear: bool = False
    stall_timeout: float = 3.0                # 指令在动而里程计不动 -> 故障
    stall_distance: float = 0.01
    stall_angle_deg: float = 3.0              # 原地转向时的角度判据(度)
    # 节奏与收尾
    wait_after_drive: float = 1.0     # 每段直行后停稳（老脚本 1.0s）
    wait_after_turn: float = 0.5      # 每次转向后停稳（老脚本 0.5s）
    startup_delay: float = 1.0        # 启动等待（老脚本 1.0s）
    mission_timeout: float = 0.0      # 整轮超时，0=不限
    publish_rate: float = 10.0        # 控制频率 Hz（老脚本 Rate(10)）
    status_rate: float = 1.0          # 状态发布频率 Hz
    # 运行方式
    auto_start: bool = True
    dry_run: bool = False             # true=只打印不真发速度
    record_csv: str = ""              # 非空则记录轨迹到该目录
    verbose: bool = True

    @classmethod
    def from_ros_params(cls) -> "MotionConfig":
        """从参数服务器(私有命名空间)装载配置，再叠加命令行覆盖并校验。"""
        cfg = cls()
        for name in cfg.__dataclass_fields__:  # type: ignore[attr-defined]
            default = getattr(cfg, name)
            value = get_private_param(name, default)
            if isinstance(default, bool) and isinstance(value, str):
                # rosparam 常把布尔写成字符串("true")，这里纠正
                value = value.strip().lower() in ("1", "true", "yes", "on", "t")
            setattr(cfg, name, value)
        cfg.apply_cli_overrides()
        cfg.validate()
        return cfg

    def apply_cli_overrides(self) -> None:
        """
        命令行参数优先级最高，方便 rosrun 时快速试不同图形，例如：
            rosrun my_pkg square_movement_advanced.py --shape star --laps 2 --dry-run
        rospy.myargv() 会先把 ROS 的 __name:=/__log:= 等重映射参数剥掉。
        """
        argv = [a for a in rospy.myargv(argv=sys.argv)[1:] if not a.startswith("__")]
        if not argv:
            return
        parser = argparse.ArgumentParser(
            prog="square_movement_advanced.py",
            description="通用几何轨迹运动节点（ROS 1）：%s" % ", ".join(
                ("line", "rotate", "square", "rectangle", "triangle", "polygon",
                 "star", "circle", "figure8", "spiral")))
        parser.add_argument("--shape", help="图形名称，如 square/star/circle")
        parser.add_argument("--laps", type=int, help="重复圈数")
        parser.add_argument("--side-length", type=float, dest="side_length", help="正方形边长(m)")
        parser.add_argument("--forward-speed", type=float, dest="forward_speed", help="直行速度(m/s)")
        parser.add_argument("--turn-speed", type=float, dest="turn_speed", help="转向角速度(rad/s)")
        parser.add_argument("--turn-direction", dest="turn_direction", choices=["cw", "ccw"])
        parser.add_argument("--record-csv", dest="record_csv", help="轨迹 CSV 输出目录")
        parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="只规划不发布")
        parser.add_argument("--no-odom", action="store_true", dest="no_odom", help="强制开环")
        parser.add_argument("--no-scan", action="store_true", dest="no_scan", help="关闭避障")
        parser.add_argument("--loop-forever", action="store_true", dest="loop_forever")
        parser.add_argument("--auto-start", action="store_true", dest="auto_start", default=None)
        parser.add_argument("--no-auto-start", action="store_false", dest="auto_start", default=None)
        args = parser.parse_args(argv)

        for attr in ("shape", "laps", "side_length", "forward_speed", "turn_speed",
                     "turn_direction", "record_csv"):
            value = getattr(args, attr)
            if value is not None:
                setattr(self, attr, value)
        if args.dry_run:
            self.dry_run = True
        if args.no_odom:
            self.use_odom = False
        if args.no_scan:
            self.use_scan = False
        if args.loop_forever:
            self.loop_forever = True
        if args.auto_start is not None:
            self.auto_start = bool(args.auto_start)

    def validate(self) -> None:
        """参数合法性检查与纠正，避免非法值送到机器人上。"""
        self.shape = str(self.shape).strip().lower()
        self.laps = max(1, int(self.laps))
        self.publish_rate = clamp(float(self.publish_rate), 1.0, 100.0)
        self.status_rate = clamp(float(self.status_rate), 0.0, 20.0)
        self.forward_speed = clamp(abs(float(self.forward_speed)), 0.0, self.max_linear_speed)
        self.back_speed = clamp(abs(float(self.back_speed)), 0.0, self.max_linear_speed)
        self.turn_speed = clamp(abs(float(self.turn_speed)), 0.0, self.max_angular_speed)
        self.arc_speed = clamp(abs(float(self.arc_speed)), 0.0, self.max_linear_speed)
        self.obstacle_fov_deg = clamp(float(self.obstacle_fov_deg), 5.0, 180.0)
        self.obstacle_release_distance = max(float(self.obstacle_release_distance),
                                            float(self.obstacle_distance) + 0.01)
        self.polygon_sides = int(clamp(float(self.polygon_sides), 3, 60))
        self.spiral_turns = int(clamp(float(self.spiral_turns), 1, 50))
        self.turn_direction = str(self.turn_direction).lower()
        if self.turn_direction not in ("cw", "ccw"):
            self.turn_direction = "cw"
        self.drive_direction = str(self.drive_direction).lower()
        if self.drive_direction not in ("forward", "backward"):
            self.drive_direction = "forward"
        if self.obstacle_action not in (SafetyAction.STOP, SafetyAction.SKIP, SafetyAction.IGNORE):
            self.obstacle_action = SafetyAction.STOP
        for name, low, high in (("side_length", 0.01, 100.0), ("rectangle_length", 0.01, 100.0),
                                ("rectangle_width", 0.01, 100.0),
                                ("polygon_side_length", 0.01, 100.0), ("star_arm", 0.01, 100.0),
                                ("circle_radius", 0.05, 100.0), ("figure8_radius", 0.05, 100.0),
                                ("spiral_start_radius", 0.05, 100.0), ("spiral_step", 0.0, 100.0),
                                ("line_length", 0.01, 100.0)):
            setattr(self, name, clamp(float(getattr(self, name)), low, high))

    @property
    def turn_sign(self) -> int:
        """cw -> -1（右转，与老脚本一致），ccw -> +1（左转）。"""
        return Direction.RIGHT if self.turn_direction == "cw" else Direction.LEFT

    @property
    def drive_sign(self) -> int:
        return Direction.BACKWARD if self.drive_direction == "backward" else Direction.FORWARD

    @property
    def drive_speed(self) -> float:
        return self.forward_speed if self.drive_sign == Direction.FORWARD else self.back_speed

    @property
    def angle_tolerance_rad(self) -> float:
        return float(self.angle_tolerance_deg) * DEG2RAD

    @property
    def stall_angle_rad(self) -> float:
        return float(self.stall_angle_deg) * DEG2RAD

    def describe(self) -> str:
        """一行摘要，启动时打印，便于确认参数没配错。"""
        return ("shape=%s laps=%s odom=%s scan=%s v=%.3fm/s w=%.3frad/s turn=%s side=%.3fm"
                % (self.shape, "inf" if self.loop_forever else self.laps,
                   "on" if self.use_odom else "off", "on" if self.use_scan else "off",
                   self.forward_speed, self.turn_speed, self.turn_direction, self.side_length))


# ---------------------------------------------------------------------------
# 里程计跟踪
# ---------------------------------------------------------------------------
class OdomTracker:
    """
    订阅里程计，维护三个供控制律使用的量：

        path_length  从节点启动至今的累计路程，用来控制「直行多远」
        yaw_cum      连续累积偏航角，用来控制「转多少度」
        x / y / yaw  当前位姿，用于状态话题与轨迹记录

    这是本节点相对老脚本最关键的一处升级。老脚本用时间推算距离，会把
    轮胎打滑、起步加速延迟都当成"已经走过的距离"，误差在每一次直行和
    每一次转弯上累积，最后方框不闭合也回不到起点。改用累计里程后，
    单次打滑只会带来一次小小的误差，不会再被后续动作放大。

    另外注意这里用的是"累计偏航角 yaw_cum"而不是原始偏航角：原始偏航角
    在 ±pi 处会跳变，做连续旋转（比如原地转 360°）时无法直接比较大小；
    改成每帧把增量 wrap 到 (-pi, pi] 后累加，就能得到单调、连续的角度。
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.has_data = False
        self.x = self.y = self.yaw = self.yaw_cum = self.path_length = 0.0
        self.linear_vel = self.angular_vel = 0.0
        self.stamp: Optional[rospy.Time] = None
        self.msg_count = 0
        self._last_x = self._last_y = self._last_yaw = None

    def reset(self) -> None:
        """清空跟踪状态（重新开始记录轨迹时使用）。"""
        with self.lock:
            self.has_data = False
            self.x = self.y = self.yaw = self.yaw_cum = self.path_length = 0.0
            self._last_x = self._last_y = self._last_yaw = None
            self.msg_count, self.stamp = 0, None

    def callback(self, msg) -> None:
        """Odometry 回调：更新位姿，并把增量累加进路程与累计偏航角。"""
        try:
            px = float(msg.pose.pose.position.x)
            py = float(msg.pose.pose.position.y)
            yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        except AttributeError:  # 消息结构异常时直接忽略
            return
        with self.lock:
            self.has_data = True
            self.msg_count += 1
            self.stamp = getattr(msg.header, "stamp", rospy.Time.now())
            self.linear_vel = float(getattr(msg.twist.twist.linear, "x", 0.0))
            self.angular_vel = float(getattr(msg.twist.twist.angular, "z", 0.0))
            if self._last_x is not None:
                # 用位移增量的模长累加，而不是首尾直线距离，
                # 这样"绕圈再回来"也能正确反映真实路程。
                self.path_length += math.hypot(px - self._last_x, py - self._last_y)
            if self._last_yaw is not None:
                self.yaw_cum += wrap_angle(yaw - self._last_yaw)
            self.x, self.y, self.yaw = px, py, yaw
            self._last_x, self._last_y, self._last_yaw = px, py, yaw

    def is_fresh(self, timeout: float) -> bool:
        """里程计数据是否新鲜（防止用陈旧数据做闭环控制）。"""
        with self.lock:
            if not self.has_data:
                return False
            if self.stamp is None:
                return True
            age = (rospy.Time.now() - self.stamp).to_sec()
        return age <= float(timeout)

    def snapshot(self) -> Dict[str, float]:
        """返回一份线程安全的状态拷贝，供日志/状态话题/CSV 使用。"""
        with self.lock:
            return {"x": self.x, "y": self.y, "yaw_deg": self.yaw * RAD2DEG,
                    "yaw_cum_deg": self.yaw_cum * RAD2DEG, "path": self.path_length,
                    "v": self.linear_vel, "w": self.angular_vel}

    @property
    def age(self) -> float:
        """距上一帧里程计的时间(秒)；从未收到返回 -1。"""
        with self.lock:
            if self.stamp is None:
                return -1.0
            return (rospy.Time.now() - self.stamp).to_sec()


# ---------------------------------------------------------------------------
# 安全监控
# ---------------------------------------------------------------------------
class SafetyMonitor:
    """
    汇总四路安全信息，统一回答「现在能不能继续动」：

        1. 激光前向扇区最小距离 —— 避障
        2. 急停话题             —— 锁存，需显式复位
        3. 遥控接管话题         —— 人在操作时屏蔽自动运动
        4. 电池电量             —— 低于阈值时告警

    避障带滞回（hysteresis）：进入 obstacle_distance 以内刹车，必须退到
    obstacle_release_distance 以外才重新放行。没有滞回的话，机器人在阈值
    边界上会反复"停车-起步-停车"，既抖又吵，还会磨损电机。
    """

    def __init__(self, cfg: MotionConfig) -> None:
        self.cfg = cfg
        self.min_scan_distance = float("inf")
        self.scan_stamp: Optional[rospy.Time] = None
        self.scan_count = 0
        self.obstacle_blocked = False
        self.estop = False
        self.teleop_override = False
        self.battery_percent = float("nan")
        self.battery_stamp: Optional[rospy.Time] = None
        self.battery_low = False
        self._warned = set()

    # --- 回调 ---
    def on_scan(self, msg) -> None:
        """只统计前向扇区内的最小距离；NaN、inf、超量程的点全部跳过。"""
        if not self.cfg.use_scan:
            return
        fov_half = self.cfg.obstacle_fov_deg * DEG2RAD / 2.0
        angle_min = float(getattr(msg, "angle_min", 0.0))
        angle_increment = float(getattr(msg, "angle_increment", 0.0))
        range_min = float(getattr(msg, "range_min", 0.0))
        range_max = float(getattr(msg, "range_max", 10.0))
        closest = float("inf")
        for index, raw in enumerate(msg.ranges):
            distance = float(raw)
            if math.isnan(distance) or math.isinf(distance):
                continue
            if distance < range_min or distance > range_max:
                continue
            angle = wrap_angle(angle_min + index * angle_increment)
            if abs(angle) > fov_half:
                continue
            if distance < closest:
                closest = distance

        self.min_scan_distance = closest
        self.scan_stamp = rospy.Time.now()
        self.scan_count += 1

        if math.isinf(closest):          # 扇区内没有任何有效回波
            self.obstacle_blocked = False
        elif self.obstacle_blocked:      # 已在刹车状态：要退够远才解除
            self.obstacle_blocked = closest < self.cfg.obstacle_release_distance
        elif closest <= self.cfg.obstacle_distance:
            self.obstacle_blocked = True
            self._warn_once("obstacle", "前向 %.2fm 处检测到障碍（阈值 %.2fm），减速停车",
                            closest, self.cfg.obstacle_distance)

    def on_battery(self, msg) -> None:
        """电池回调：兼容 0~1 与 0~100 两种百分比写法。"""
        if not self.cfg.use_battery:
            return
        percent = float(getattr(msg, "percentage", float("nan")))
        if percent != percent:  # NaN，说明驱动器没提供百分比
            return
        if percent <= 1.0:
            percent *= 100.0
        self.battery_percent = percent
        self.battery_stamp = rospy.Time.now()
        self.battery_low = percent <= self.cfg.min_battery_percent
        if self.battery_low:
            self._warn_once("battery", "电池电量偏低：%.1f%%（阈值 %.1f%%）",
                            percent, self.cfg.min_battery_percent)

    def on_estop(self, msg) -> None:
        """急停消息：默认锁存，收到 true 后必须显式重发 false 才解除。"""
        value = bool(msg.data)
        if value and not self.estop:
            rospy.logerr("收到急停信号，任务中止")
        if value or self.cfg.estop_auto_clear:
            self.estop = value

    def on_teleop(self, msg) -> None:
        """遥控接管：true 时自动运动暂停，false 时自动恢复。"""
        value = bool(msg.data)
        if value and not self.teleop_override:
            rospy.logwarn("检测到遥控接管，自动运动暂停")
        self.teleop_override = value

    # --- 查询 ---
    def scan_is_fresh(self) -> bool:
        """激光数据是否新鲜；过期数据不能用来判断障碍。"""
        return (self.scan_stamp is not None
                and (rospy.Time.now() - self.scan_stamp).to_sec() <= self.cfg.scan_timeout)

    def obstacle_holds(self) -> bool:
        """当前是否应该因为障碍而停下。"""
        return bool(self.cfg.use_scan and self.scan_is_fresh() and self.obstacle_blocked)

    def blocking_reason(self, ignore_obstacle: bool = False) -> Optional[str]:
        """返回阻塞原因字符串；None 表示可以继续运动。"""
        if self.estop:
            return "紧急停止已触发"
        if self.teleop_override:
            return "遥控接管中"
        if not ignore_obstacle and self.obstacle_holds():
            return "前向障碍物 %.2fm" % self.min_scan_distance
        return None

    def summary(self) -> Dict[str, object]:
        """安全状态摘要，写进状态话题。"""
        return {"estop": self.estop, "teleop_override": self.teleop_override,
                "obstacle_blocked": self.obstacle_blocked, "scan_ok": self.scan_is_fresh(),
                "min_scan_distance": (None if math.isinf(self.min_scan_distance)
                                      else round(self.min_scan_distance, 3)),
                "battery_percent": (None if self.battery_percent != self.battery_percent
                                    else round(self.battery_percent, 1)),
                "battery_low": self.battery_low}

    def _warn_once(self, key: str, message: str, *args) -> None:
        """同一类告警只打印一次，避免日志被反复刷屏。"""
        if key not in self._warned:
            self._warned.add(key)
            rospy.logwarn(message, *args)


# ---------------------------------------------------------------------------
# 速度指令输出（带加速度限幅）
# ---------------------------------------------------------------------------
class VelocityCommander:
    """
    统一封装速度指令的发布。

    与老脚本"把目标速度直接写进 Twist 发出去"的区别在于这里做了加速度
    限幅：目标速度变化时，按 max_linear_accel / max_angular_accel 逐帧
    逼近，相当于给机器人加了梯形速度曲线。带来的好处：

        * 起步不打滑（老脚本从 0 直接跳到 0.22rad/s，轮子容易空转）；
        * 到点不过冲（老脚本要靠把 7.14s 改成 7.6s 来补偿惯性）；
        * 急停时能滑行到 0 而不是瞬间归零，底盘不会甩尾。

    同时它也是唯一往 /cmd_vel 写数据的地方，因此 dry_run、急停归零、
    退出兜底这些行为只需要在这一处实现。
    """

    def __init__(self, cfg: MotionConfig, publisher, rate, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.pub = publisher
        self.rate = rate
        self.dry_run = bool(dry_run)
        self.lock = threading.Lock()
        self.linear = self.angular = 0.0
        self.sent_count = 0
        self.max_linear_seen = self.max_angular_seen = 0.0
        self._last_stamp = ros_seconds_now()

    def _dt(self) -> float:
        """两次指令之间的时间间隔；异常时退化为标称控制周期。"""
        now = ros_seconds_now()
        dt = now - self._last_stamp
        self._last_stamp = now
        if dt <= 1e-4 or dt > 1.0:
            return 1.0 / max(self.cfg.publish_rate, 1.0)
        return dt

    @staticmethod
    def _approach(current: float, target: float, max_delta: float) -> float:
        """把 current 以最大步长 max_delta 推向 target（加速度限幅的核心）。"""
        if abs(target - current) <= max_delta:
            return target
        return current + math.copysign(max_delta, target - current)

    def _publish(self) -> None:
        """真正发布一条 Twist；dry_run 时只更新内部状态不发送。"""
        if self.dry_run or self.pub is None:
            return
        twist = Twist()
        twist.linear.x = self.linear
        twist.angular.z = self.angular
        try:
            self.pub.publish(twist)
        except Exception as exc:  # pragma: no cover - 话题异常
            rospy.logerr_throttle(5.0, "发布速度指令失败: %s", exc)
            return
        self.sent_count += 1
        self.max_linear_seen = max(self.max_linear_seen, abs(self.linear))
        self.max_angular_seen = max(self.max_angular_seen, abs(self.angular))

    def command(self, linear: float, angular: float, ramped: bool = True) -> None:
        """发布目标速度；ramped=True 时按加速度限制平滑过渡。"""
        target_v = clamp(float(linear), -self.cfg.max_linear_speed, self.cfg.max_linear_speed)
        target_w = clamp(float(angular), -self.cfg.max_angular_speed, self.cfg.max_angular_speed)
        with self.lock:
            if ramped:
                dt = self._dt()
                self.linear = self._approach(self.linear, target_v,
                                             self.cfg.max_linear_accel * dt)
                self.angular = self._approach(self.angular, target_w,
                                              self.cfg.max_angular_accel * dt)
            else:
                self.linear, self.angular = target_v, target_w
            self._publish()

    def stop(self, ramped: bool = True, timeout: float = 2.0) -> None:
        """停车。ramped=True 时按减速度滑行到 0；否则立即归零。"""
        if not ramped:
            self.force_zero()
            return
        deadline = ros_seconds_now() + max(float(timeout), 0.1)
        while not rospy.is_shutdown():
            with self.lock:
                if abs(self.linear) <= 1e-3 and abs(self.angular) <= 1e-3:
                    self.linear = self.angular = 0.0
                    self._publish()
                    return
            self.command(0.0, 0.0, ramped=True)
            if ros_seconds_now() >= deadline - TIME_EPS:
                break
            self.rate.sleep()
        self.force_zero()

    def force_zero(self) -> None:
        """立即发布一条全零 Twist（急停、异常、退出时的兜底动作）。"""
        with self.lock:
            self.linear = self.angular = 0.0
            self._publish()

    @property
    def is_moving(self) -> bool:
        return abs(self.linear) > 1e-3 or abs(self.angular) > 1e-3


# ---------------------------------------------------------------------------
# 轨迹记录
# ---------------------------------------------------------------------------
class TrajectoryRecorder:
    """
    把运动过程中的位姿与速度写成 CSV，方便事后画轨迹复盘（用 Excel 或
    Python 的 matplotlib 直接读就行）。只有在参数 record_csv 非空时才
    真正打开文件；任何写入异常都只告警，不影响运动控制。

    CSV 列：ros_time, x, y, yaw_deg, yaw_cum_deg, path_m, v_mps, w_radps, state
    """

    HEADER = ["ros_time", "x", "y", "yaw_deg", "yaw_cum_deg", "path_m",
              "v_mps", "w_radps", "state"]

    def __init__(self, directory: str, prefix: str = "trajectory") -> None:
        self.directory = directory or ""
        self.prefix = prefix
        self.enabled = bool(directory)
        self.path = ""
        self.rows = 0
        self._file = None
        self._first_write = 0.0
        self._last_flush = 0.0

    def open(self) -> bool:
        """创建文件并写表头，返回是否可用。"""
        if not self.enabled or not ensure_directory(self.directory):
            self.enabled = False
            return False
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(self.directory, "%s_%s.csv" % (self.prefix, stamp))
        try:
            self._file = open(self.path, "w", encoding="utf-8")
            self._file.write(",".join(self.HEADER) + "\n")
            self._file.flush()
        except OSError as exc:  # pragma: no cover - 磁盘/权限问题
            rospy.logwarn("无法写入轨迹文件 %s: %s", self.path, exc)
            self.enabled = False
            return False
        self._first_write = ros_seconds_now()
        rospy.loginfo("轨迹将记录到 %s", self.path)
        return True

    def append(self, snapshot: Dict[str, float], state: str) -> None:
        """追加一行；约每 2 秒落盘一次，兼顾安全与性能。"""
        if not self.enabled or self._file is None:
            return
        try:
            self._file.write("%.3f,%.4f,%.4f,%.2f,%.2f,%.4f,%.4f,%.4f,%s\n" % (
                ros_seconds_now(), snapshot.get("x", 0.0), snapshot.get("y", 0.0),
                snapshot.get("yaw_deg", 0.0), snapshot.get("yaw_cum_deg", 0.0),
                snapshot.get("path", 0.0), snapshot.get("v", 0.0), snapshot.get("w", 0.0),
                state))
            self.rows += 1
            now = ros_seconds_now()
            if now - self._last_flush > 2.0:
                self._file.flush()
                self._last_flush = now
        except Exception as exc:  # pragma: no cover
            rospy.logwarn_throttle(10.0, "写轨迹失败: %s", exc)

    def close(self) -> None:
        """刷盘、关闭文件，并打印记录行数。"""
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except OSError:  # pragma: no cover
                pass
        if self.enabled and self.path:
            rospy.loginfo("轨迹记录结束：%s（%d 行）", self.path, self.rows)
        self._file = None


# ---------------------------------------------------------------------------
# 图形描述
# ---------------------------------------------------------------------------
@dataclass
class MotionStep:
    """
    一个原子运动步骤。整个任务就是若干步骤排成的队列，例如正方形就是：
        DRIVE 0.42m -> TURN 90° -> DRIVE 0.42m -> TURN 90° -> ...

    这样做的好处是"图形"和"执行"彻底解耦：想加一个新图形，只需要在
    ShapeFactory 里把它翻译成步骤序列，闭环控制、避障、暂停、统计、
    轨迹记录这些逻辑一行都不用改、也不会漏改。
    """

    kind: MotionKind
    value: float                       # drive/arc: 长度(m)或圆心角(°)；turn: 角度(°)；wait: 秒
    direction: int = Direction.FORWARD
    radius: float = 0.0                # 仅 ARC 使用
    label: str = ""                    # 日志里显示的步骤名

    def describe(self) -> str:
        """生成一行人类可读的步骤描述，用于日志和状态话题。"""
        side = "左/逆时针" if self.direction > 0 else "右/顺时针"
        if self.kind is MotionKind.DRIVE:
            return "直行 %.3fm %s" % (abs(self.value), side)
        if self.kind is MotionKind.TURN:
            return "原地转 %.1f° %s" % (abs(self.value), side)
        if self.kind is MotionKind.ARC:
            return "弧线 r=%.3fm 转角 %.1f° %s" % (self.radius, abs(self.value), side)
        return "等待 %.2fs" % abs(self.value)


class ShapeFactory:
    """
    把所有支持的图形统一翻译成 MotionStep 列表。

    几何要点：
        * 正 n 边形：每次转"外角" = 360/n。正方形是 90°、正三角形是 120°、
          五边形是 72°。老脚本里 7.6s x 0.22rad/s ≈ 95.8° 就是这个外角，
          多出来的 5.8° 是给惯性过冲留的余量（闭环模式下不再需要）。
        * 五角星：走一条臂后转 144° 外角（{5/2} 星形多边形），5 次后闭合。
        * 圆/8 字/螺旋：都是 ARC 步骤，角速度 ω = v / r，半径越小转得越快。
    """

    SHAPES = ("line", "rotate", "square", "rectangle", "triangle", "polygon",
              "star", "circle", "figure8", "spiral")

    @classmethod
    def supported(cls) -> Sequence[str]:
        return cls.SHAPES

    @classmethod
    def build(cls, cfg: MotionConfig) -> List[MotionStep]:
        """按配置生成「一圈」图形的步骤序列。"""
        builders = {"line": cls._line, "rotate": cls._rotate, "square": cls._square,
                    "rectangle": cls._rectangle, "triangle": cls._triangle,
                    "polygon": cls._polygon, "star": cls._star, "circle": cls._circle,
                    "figure8": cls._figure8, "spiral": cls._spiral}
        builder = builders.get(cfg.shape)
        if builder is None:
            rospy.logwarn("未知图形 %r，回退到 square", cfg.shape)
            builder = cls._square
        return builder(cfg)

    # --- 各图形构造 ---
    @classmethod
    def _polygon_steps(cls, cfg: MotionConfig, sides: int, side_length: float,
                       close_loop_turn: bool) -> List[MotionStep]:
        """正多边形：走一条边 + 转一个外角，外角 = 360/n。"""
        steps: List[MotionStep] = []
        exterior = 360.0 / float(sides)
        for index in range(sides):
            steps.append(MotionStep(MotionKind.DRIVE, side_length, cfg.drive_sign,
                                    label="边 %d/%d" % (index + 1, sides)))
            if index < sides - 1 or close_loop_turn:
                steps.append(MotionStep(MotionKind.TURN, exterior, cfg.turn_sign,
                                        label="外角 %.1f°" % exterior))
        return steps

    @classmethod
    def _line(cls, cfg: MotionConfig) -> List[MotionStep]:
        """直线：只走一段，适合快速验证直行距离标定。"""
        return [MotionStep(MotionKind.DRIVE, cfg.line_length, cfg.drive_sign, label="直线")]

    @classmethod
    def _rotate(cls, cfg: MotionConfig) -> List[MotionStep]:
        """原地旋转，支持超过 180°（靠累计偏航角积分，不怕角度环绕）。"""
        return [MotionStep(MotionKind.TURN, abs(cfg.rotate_angle_deg), cfg.turn_sign,
                           label="原地旋转")]

    @classmethod
    def _square(cls, cfg: MotionConfig) -> List[MotionStep]:
        """正方形：与老脚本完全一致的图形（4 条边，每角 90°）。"""
        return cls._polygon_steps(cfg, 4, cfg.side_length, cfg.turn_after_last_edge)

    @classmethod
    def _triangle(cls, cfg: MotionConfig) -> List[MotionStep]:
        """正三角形：外角 120°。"""
        return cls._polygon_steps(cfg, 3, cfg.side_length, cfg.turn_after_last_edge)

    @classmethod
    def _polygon(cls, cfg: MotionConfig) -> List[MotionStep]:
        """任意正多边形，边数由 polygon_sides 指定（3~60）。"""
        return cls._polygon_steps(cfg, cfg.polygon_sides, cfg.polygon_side_length,
                                  cfg.turn_after_last_edge)

    @classmethod
    def _rectangle(cls, cfg: MotionConfig) -> List[MotionStep]:
        """长方形：两条长边 + 两条短边，每角 90°。"""
        steps: List[MotionStep] = []
        for index, side in enumerate((cfg.rectangle_length, cfg.rectangle_width,
                                      cfg.rectangle_length, cfg.rectangle_width)):
            steps.append(MotionStep(MotionKind.DRIVE, side, cfg.drive_sign,
                                    label="边 %d/4" % (index + 1)))
            if index < 3 or cfg.turn_after_last_edge:
                steps.append(MotionStep(MotionKind.TURN, 90.0, cfg.turn_sign, label="直角"))
        return steps

    @classmethod
    def _star(cls, cfg: MotionConfig) -> List[MotionStep]:
        """五角星：走一条臂 + 转 144° 外角，5 次后回到起点。"""
        steps: List[MotionStep] = []
        for index in range(5):
            steps.append(MotionStep(MotionKind.DRIVE, cfg.star_arm, cfg.drive_sign,
                                    label="星臂 %d/5" % (index + 1)))
            steps.append(MotionStep(MotionKind.TURN, 144.0, cfg.turn_sign, label="星角 144°"))
        return steps

    @classmethod
    def _circle(cls, cfg: MotionConfig) -> List[MotionStep]:
        """圆：一整段 360° 弧线，角速度 ω = v / r。"""
        return [MotionStep(MotionKind.ARC, 360.0, cfg.turn_sign,
                           radius=cfg.circle_radius, label="整圆")]

    @classmethod
    def _figure8(cls, cfg: MotionConfig) -> List[MotionStep]:
        """数字 8：一个左圆接一个右圆，两圆相切即构成 8 字。"""
        return [MotionStep(MotionKind.ARC, 360.0, Direction.LEFT,
                           radius=cfg.figure8_radius, label="8-左环"),
                MotionStep(MotionKind.ARC, 360.0, Direction.RIGHT,
                           radius=cfg.figure8_radius, label="8-右环")]

    @classmethod
    def _spiral(cls, cfg: MotionConfig) -> List[MotionStep]:
        """螺旋：每转一圈半径增加 spiral_step，用于测试连续变化的曲率。"""
        return [MotionStep(MotionKind.ARC, 360.0, cfg.turn_sign,
                           radius=cfg.spiral_start_radius + turn * cfg.spiral_step,
                           label="螺旋第 %d 圈" % (turn + 1))
                for turn in range(cfg.spiral_turns)]


@dataclass
class MissionStats:
    """一轮任务的统计结果：写进日志与状态话题，并决定这轮算成功还是失败。"""

    shape: str = ""
    laps_requested: int = 0
    laps_done: int = 0
    steps_total: int = 0
    steps_done: int = 0
    steps_skipped: int = 0            # 因避障策略 skip 而跳过的步骤数
    current_step: str = ""
    distance_commanded: float = 0.0   # 计划走过的距离（指令层面）
    distance_measured: float = 0.0    # 里程计实测累计路程
    turn_deg_commanded: float = 0.0
    duration: float = 0.0
    obstacle_holds: int = 0           # 因障碍停车的次数
    finished: bool = False
    abort_reason: str = ""

    def summary(self) -> Dict[str, object]:
        """整理成字典，便于塞进 JSON 状态话题。"""
        return {"shape": self.shape,
                "laps": "%d/%d" % (self.laps_done, self.laps_requested),
                "steps": "%d/%d" % (self.steps_done, self.steps_total),
                "steps_skipped": self.steps_skipped,
                "current": self.current_step,
                "distance_planned_m": round(self.distance_commanded, 3),
                "distance_measured_m": round(self.distance_measured, 3),
                "turn_planned_deg": round(self.turn_deg_commanded, 1),
                "duration_s": round(self.duration, 1),
                "obstacle_holds": self.obstacle_holds,
                "finished": self.finished,
                "abort_reason": self.abort_reason}


class MissionAborted(Exception):
    """任务被中止（收到 stop、急停、故障、节点关闭等）。"""


class StepSkipped(Exception):
    """当前运动步骤被跳过（避障策略设为 skip 时使用）。"""


# ---------------------------------------------------------------------------
# 运动执行器
# ---------------------------------------------------------------------------
class MotionExecutor:
    """
    把 MotionStep 队列翻译成实际的速度指令。每个运动循环都做四件事：

        1. _checkpoint() —— 检查停止/暂停/急停/避障/超时，必要时就地停车等待。
           它是所有运动循环唯一的守门人，保证任何时刻都能被打断。
        2. 计算目标速度 —— 有里程计走 PID 闭环（自动减速逼近目标），
           没有里程计则回退到老脚本的「速度 x 时间」开环逻辑。
        3. 交给 VelocityCommander 做加速度限幅后发布。
        4. 堵转检测 —— 指令要求运动但里程计长时间无变化，按故障中止，
           避免电机长时间堵转发热（老脚本完全依赖人工发现）。
    """

    def __init__(self, node) -> None:
        self.node = node
        self.cfg: MotionConfig = node.cfg
        self.commander: VelocityCommander = node.commander
        self.safety: SafetyMonitor = node.safety
        self._drive_pid = PIDController(self.cfg.drive_kp, self.cfg.drive_ki, self.cfg.drive_kd,
                                        output_limit=self.cfg.max_linear_speed, integral_limit=1.0)
        self._turn_pid = PIDController(self.cfg.turn_kp, self.cfg.turn_ki, self.cfg.turn_kd,
                                       output_limit=self.cfg.max_angular_speed, integral_limit=1.0)
        self.obstacle_holds = 0
        # 堵转看门狗的基准值：路程 + 累计偏航角，两者任一变化都算"在动"
        self._progress_path = 0.0
        self._progress_yaw = 0.0
        self._progress_time = ros_seconds_now()
        self._fallback_logged = False

    # ------------------------------------------------------------------
    # 检查点：所有运动循环的唯一守门人
    # ------------------------------------------------------------------
    def _checkpoint(self, ignore_obstacle: bool = False) -> None:
        """每次循环迭代都经过这里，保证可中断、可暂停、可避障。"""
        self._check_abort()
        self._check_mission_timeout()
        self._wait_while_paused()

        reason = self.safety.blocking_reason(ignore_obstacle=ignore_obstacle)
        if reason is None:
            if self.node.state != RobotState.RUNNING:
                self.node.set_state(RobotState.RUNNING)
            return

        # 急停优先级最高：直接判故障，不再自动恢复
        if self.safety.estop:
            self.node.fault("急停触发")
            raise MissionAborted("急停触发")

        # 遥控接管：按暂停处理，人手操作结束后自动继续
        if self.safety.teleop_override:
            self._hold_until_clear("遥控接管中", lambda: self.safety.teleop_override)
            return

        # 障碍物：按参数选择策略
        if self.cfg.obstacle_action == SafetyAction.IGNORE:
            rospy.logwarn_throttle(5.0, "忽略障碍（obstacle_action=ignore）：%s", reason)
            return
        if self.cfg.obstacle_action == SafetyAction.SKIP:
            rospy.logwarn("跳过当前步骤：%s", reason)
            raise StepSkipped(reason)
        self._hold_until_clear(reason, self.safety.obstacle_holds, is_obstacle=True)

    def _check_abort(self) -> None:
        """节点关闭或收到 stop 请求时立刻抛出，让整条调用栈快速退出。"""
        if rospy.is_shutdown():
            raise MissionAborted("节点正在关闭")
        if self.node.stop_requested.is_set():
            raise MissionAborted("收到停止请求")

    def _check_mission_timeout(self) -> None:
        """整轮任务的兜底超时：防止因为逻辑异常一直跑下去。"""
        if self.cfg.mission_timeout > 0 and self.node.mission_elapsed() > self.cfg.mission_timeout:
            self.node.fault("任务总时长超过 %.1fs" % self.cfg.mission_timeout)
            raise MissionAborted("任务超时")

    def _wait_while_paused(self) -> None:
        """暂停期间持续发零速度，避免底盘因指令超时而抽动或保持旧速度。"""
        if not self.node.pause_requested.is_set():
            return
        self.node.set_state(RobotState.PAUSED)
        rospy.loginfo("任务已暂停")
        self.commander.stop(ramped=True)
        while self.node.pause_requested.is_set():
            self._check_abort()
            self.commander.force_zero()
            self.node.sleep(0.1)
        rospy.loginfo("任务已恢复")
        self.node.set_state(RobotState.RUNNING)

    def _hold_until_clear(self, reason: str, predicate: Callable[[], bool],
                          is_obstacle: bool = False) -> None:
        """停车等待某个安全条件消失（障碍物 / 遥控接管）。"""
        rospy.logwarn("安全停车：%s，等待条件解除", reason)
        self.commander.stop(ramped=True)
        self.node.set_state(RobotState.PAUSED)
        self.obstacle_holds += 1
        self.node.stats.obstacle_holds = self.obstacle_holds
        wait_start = ros_seconds_now()
        self._reset_progress_watch()

        while predicate():
            self._check_abort()
            self._check_mission_timeout()
            self.commander.force_zero()
            if is_obstacle and self.cfg.obstacle_wait_timeout > 0 and (
                    ros_seconds_now() - wait_start > self.cfg.obstacle_wait_timeout):
                self.node.fault("障碍物等待超过 %.1fs" % self.cfg.obstacle_wait_timeout)
                raise MissionAborted("障碍物长时间未清除")
            self.node.sleep(0.2)

        rospy.loginfo("安全条件解除，继续运动（本次等待 %.1fs）", ros_seconds_now() - wait_start)
        self.node.set_state(RobotState.RUNNING)

    # ------------------------------------------------------------------
    # 堵转检测与闭环可用性判断
    # ------------------------------------------------------------------
    def _current_progress_value(self) -> float:
        """
        用两个量一起衡量"机器人是否真的在动"：
            path_length 累计路程 —— 直行/弧线时会变
            yaw_cum     累计偏航角 —— 原地转向时会变

        只用累计路程是不够的：原地转向时轮子在转、在消耗电流，但里程计
        位置不变，会被误判成"堵转"而错误中止任务。这是本节点开发时实测
        踩到的坑（原地转 3 秒就被判故障）。
        """
        return self.node.odom.path_length, self.node.odom.yaw_cum

    def _reset_progress_watch(self) -> None:
        self._progress_path, self._progress_yaw = self._current_progress_value()
        self._progress_time = ros_seconds_now()

    def _check_stall(self) -> None:
        """
        指令要求运动，但路程与偏航角都长时间没有变化 -> 判定为堵转/悬空。

        两个量任意一个有变化都算"在动"，因此直行段看路程、转向段看角度，
        互不干扰，不会出现"原地转着转着被判堵转"的误报。
        """
        if self.cfg.stall_timeout <= 0 or not self._closed_loop_available():
            return
        path, yaw = self._current_progress_value()
        moved = abs(path - self._progress_path) > self.cfg.stall_distance
        turned = abs(yaw - self._progress_yaw) > self.cfg.stall_angle_rad
        if moved or turned:
            self._reset_progress_watch()
        elif ros_seconds_now() - self._progress_time > self.cfg.stall_timeout:
            self.node.fault("疑似堵转：下发运动指令 %.1fs 但里程计无变化" % self.cfg.stall_timeout)
            raise MissionAborted("疑似堵转")

    def _closed_loop_available(self) -> bool:
        """
        只有里程计存在、新鲜、且用户没关掉时才启用闭环；
        否则安静地回退到开环时间控制（只提示一次，不反复刷日志）。

       """
        reason = None
        if self.cfg.dry_run:
            # dry-run 不发真实速度，机器人不会动，闭环永远收敛不了，
            # 所以这种模式下强制用开环计时，任务才能"空跑"一遍完整流程。
            reason = "dry-run 模式没有真实运动反馈"
        elif not self.cfg.use_odom:
            reason = None                     # 用户主动关闭，不必提示
        elif not _HAS_NAV_MSGS:
            reason = "环境缺少 nav_msgs"
        elif not self.node.odom.has_data:
            reason = "尚未收到里程计"
        elif not self.node.odom.is_fresh(self.cfg.odom_timeout):
            reason = "里程计数据超时"
        if reason is None and self.cfg.use_odom:
            self._fallback_logged = False
            return True
        if reason and not self._fallback_logged:
            rospy.logwarn("%s，回退到开环时间控制", reason)
            self._fallback_logged = True
        return False

    # ------------------------------------------------------------------
    # 三个基本动作
    # ------------------------------------------------------------------
    def drive_straight(self, step: MotionStep) -> None:
        """
        直行指定距离。

        闭环：记下当前累计路程，目标 = 当前 + 距离；每帧用 PID 把
              "剩余距离"换算成速度，快到时自动减速，走够即停。
        开环：老脚本逻辑 —— 时长 = 距离 / 速度，发速度发够时间就停。
        """
        distance = abs(float(step.value))
        if distance <= 1e-6:
            return
        sign, label = int(step.direction), step.label or "直行"
        closed = self._closed_loop_available()
        self._drive_pid.reset()

        if closed:
            target = self.node.odom.path_length + distance
            rospy.loginfo("[%s] 闭环直行 %.3fm（当前里程 %.3fm -> 目标 %.3fm）",
                          label, distance, self.node.odom.path_length, target)
        else:
            speed = self.cfg.drive_speed if sign > 0 else self.cfg.back_speed
            duration = distance / max(speed, 1e-3)
            deadline = ros_seconds_now() + duration
            rospy.loginfo("[%s] 开环直行 %.3fm（%.1fs @ %.3fm/s）",
                          label, distance, duration, speed)

        self._reset_progress_watch()
        self.node.stats.current_step = label
        last_stamp = ros_seconds_now()

        while True:
            self._checkpoint()
            now = ros_seconds_now()
            dt, last_stamp = max(now - last_stamp, 1e-3), now
            if closed:
                remaining = target - self.node.odom.path_length
                if remaining <= self.cfg.position_tolerance:
                    break
                # PID 输出按当前速度上限裁剪，并保证不低于最小速度
                raw = self._drive_pid.update(remaining, dt)
                magnitude = clamp(max(raw, 0.0), self.cfg.min_linear_speed,
                                  max(self.cfg.drive_speed, 1e-3))
                self.commander.command(sign * magnitude, 0.0)
            else:
                if now >= deadline:
                    break
                self.commander.command(sign * speed, 0.0)
            self._check_stall()
            self.node.rate.sleep()

        self.commander.stop(ramped=True)
        self._settle(self.cfg.wait_after_drive)
        self._record_step(distance, 0.0)
        rospy.loginfo("[%s] 完成：累计路程 %.3fm（本段计划 %.3fm）",
                      label, self.node.odom.path_length, distance)

    def turn_in_place(self, step: MotionStep) -> None:
        """
        原地旋转指定角度（度）。

        闭环：目标 = 当前累计偏航角 + 符号 x 角度，用 PID 把"剩余角度"
              换算成角速度，进入死区即停。用累计偏航角比较，所以原地转
              360°、720° 都不会因为角度环绕而出错。
        开环：时长 = 弧度 / 角速度，即老脚本 turn_time = 7.6s 的逻辑。
        """
        angle_deg = abs(float(step.value))
        if angle_deg <= 1e-6:
            return
        angle_rad = angle_deg * DEG2RAD
        sign, label = int(step.direction), step.label or "原地转向"
        closed = self._closed_loop_available()
        self._turn_pid.reset()

        if closed:
            target = self.node.odom.yaw_cum + sign * angle_rad
            rospy.loginfo("[%s] 闭环转向 %.1f°（累计偏航 %.1f° -> 目标 %.1f°）",
                          label, angle_deg, self.node.odom.yaw_cum * RAD2DEG, target * RAD2DEG)
        else:
            omega = max(self.cfg.turn_speed, 1e-3)
            duration = angle_rad / omega
            deadline = ros_seconds_now() + duration
            rospy.loginfo("[%s] 开环转向 %.1f°（%.1fs @ %.3frad/s）",
                          label, angle_deg, duration, omega)

        self._reset_progress_watch()
        self.node.stats.current_step = label
        last_stamp = ros_seconds_now()

        while True:
            self._checkpoint()
            now = ros_seconds_now()
            dt, last_stamp = max(now - last_stamp, 1e-3), now
            if closed:
                # 取"朝目标方向还差多少"为正的组合误差，方便统一处理左右转
                remaining = sign * (target - self.node.odom.yaw_cum)
                if remaining <= self.cfg.angle_tolerance_rad:
                    break
                raw = self._turn_pid.update(remaining, dt)
                magnitude = clamp(max(raw, 0.0), self.cfg.min_angular_speed,
                                  max(self.cfg.turn_speed, 1e-3))
                self.commander.command(0.0, sign * magnitude)
            else:
                if now >= deadline:
                    break
                self.commander.command(0.0, sign * self.cfg.turn_speed)
            self._check_stall()
            self.node.rate.sleep()

        self.commander.stop(ramped=True)
        self._settle(self.cfg.wait_after_turn)
        self._record_step(0.0, angle_deg)
        rospy.loginfo("[%s] 完成：累计偏航 %.1f°", label, self.node.odom.yaw_cum * RAD2DEG)

    def drive_arc(self, step: MotionStep) -> None:
        """
        圆弧运动：线速度与角速度同时下发，满足 ω = v / r。

        半径越小，同样线速度下需要的角速度越大，所以 ω 由 arc_speed/radius
        算出并限幅。最后剩余 10° 左右按比例降速（软着陆），避免到点急停
        造成过冲 —— 这正是老脚本要用 7.6s 硬补惯性、而闭环模式不需要的原因。
        """
        angle_deg = abs(float(step.value))
        if angle_deg <= 1e-6:
            return
        angle_rad = angle_deg * DEG2RAD
        sign, label = int(step.direction), step.label or "弧线"
        radius = max(float(step.radius), 0.05)
        omega = clamp(self.cfg.arc_speed / radius, self.cfg.min_angular_speed,
                      self.cfg.max_angular_speed)
        linear = min(self.cfg.arc_speed, omega * radius)
        closed = self._closed_loop_available()

        if closed:
            target = self.node.odom.yaw_cum + sign * angle_rad
            rospy.loginfo("[%s] 闭环弧线 r=%.3fm %.1f°（v=%.3fm/s ω=%.3frad/s）",
                          label, radius, angle_deg, linear, omega)
        else:
            duration = angle_rad / max(omega, 1e-3)
            deadline = ros_seconds_now() + duration
            rospy.loginfo("[%s] 开环弧线 r=%.3fm %.1f°（%.1fs）",
                          label, radius, angle_deg, duration)

        self._reset_progress_watch()
        self.node.stats.current_step = label
        soft_window = max(10.0 * DEG2RAD, self.cfg.angle_tolerance_rad * 2.0)

        while True:
            self._checkpoint()
            now = ros_seconds_now()
            if closed:
                remaining = sign * (target - self.node.odom.yaw_cum)
                if remaining <= self.cfg.angle_tolerance_rad:
                    break
                scale = clamp(remaining / soft_window, 0.15, 1.0)
                self.commander.command(sign * linear * scale, sign * omega * scale)
            else:
                if now >= deadline:
                    break
                self.commander.command(sign * linear, sign * omega)
            self._check_stall()
            self.node.rate.sleep()

        self.commander.stop(ramped=True)
        self._settle(self.cfg.wait_after_turn)
        self._record_step(angle_rad * radius, angle_deg)
        rospy.loginfo("[%s] 完成：绕行弧长约 %.3fm", label, angle_rad * radius)

    def wait(self, step: MotionStep) -> None:
        """原地等待，期间持续输出零速度并保持安全检查。"""
        duration = abs(float(step.value))
        if duration <= 0:
            return
        self.commander.stop(ramped=True)
        deadline = ros_seconds_now() + duration
        while ros_seconds_now() < deadline - TIME_EPS:
            self._checkpoint()
            self.commander.force_zero()
            self.node.sleep(0.1)
        self._record_step(0.0, 0.0)

    # ------------------------------------------------------------------
    # 组合执行
    # ------------------------------------------------------------------
    def execute_step(self, step: MotionStep) -> None:
        """把一个步骤分派给对应的动作实现。"""
        handlers = {MotionKind.DRIVE: self.drive_straight,
                    MotionKind.TURN: self.turn_in_place,
                    MotionKind.ARC: self.drive_arc,
                    MotionKind.WAIT: self.wait}
        handler = handlers.get(step.kind)
        if handler is None:  # pragma: no cover - 枚举齐全，防御性分支
            rospy.logwarn("未知步骤类型：%r", step.kind)
            return
        handler(step)

    def execute_plan(self, steps: List[MotionStep], laps: int,
                     loop_forever: bool = False) -> MissionStats:
        """
        按圈执行步骤队列，返回统计信息。

        这里是唯一捕获 MissionAborted 的地方：无论任务是被 stop 服务、
        急停、堵转还是超时中止，都会走到 finally 里停车、结算耗时。
        """
        stats = self.node.stats
        stats.shape = self.cfg.shape
        stats.laps_requested = 0 if loop_forever else int(laps)
        stats.steps_total = 0 if loop_forever else len(steps) * int(laps)
        lap_index = 0
        start_time = ros_seconds_now()

        try:
            while True:
                lap_index += 1
                rospy.loginfo("=== 第 %d 圈开始（本圈 %d 步）===", lap_index, len(steps))
                for step_index, step in enumerate(steps):
                    try:
                        # 注意：检查点必须放在 try 里面。避障策略为 skip 时，
                        # 这里抛出的 StepSkipped 就是"跳过本步骤"的正常流程，
                        # 放到外面会变成未捕获异常、把整轮任务判成故障。
                        self._checkpoint()
                        self.node.set_progress(0 if loop_forever else lap_index,
                                               stats.steps_done, stats.steps_total,
                                               step.describe())
                        if self.cfg.verbose:
                            rospy.loginfo("步骤 %d/%d：%s", step_index + 1, len(steps),
                                          step.describe())
                        self.execute_step(step)
                    except StepSkipped as exc:
                        # 避障策略为 skip 时，跳过的步骤也计入完成，任务继续往下走
                        rospy.logwarn("已跳过「%s」：%s", step.describe(), exc)
                        stats.steps_skipped += 1
                    stats.steps_done += 1
                    stats.steps_total = max(stats.steps_total, stats.steps_done)
                    self.node.publish_status(force=True)

                stats.laps_done = lap_index
                rospy.loginfo("=== 第 %d 圈完成（本圈跳过 %d 步）===", lap_index,
                              stats.steps_skipped)
                if loop_forever or lap_index < int(laps):
                    continue
                break
            stats.finished = True
        except MissionAborted as exc:
            stats.abort_reason = str(exc)
            stats.finished = False
            rospy.logwarn("任务中止：%s", exc)
        finally:
            self.commander.stop(ramped=True)
            stats.duration = ros_seconds_now() - start_time
            stats.distance_measured = self.node.odom.path_length
        return stats

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def _settle(self, duration: float) -> None:
        """
        每段运动之间的停稳时间，对应老脚本里的 time.sleep(1.0) 与 sleep(0.5)。
        机器人减速需要时间，停稳再开始下一段能显著减少转弯时的位置漂移。
        """
        if duration <= 0:
            return
        self.commander.stop(ramped=False)
        deadline = ros_seconds_now() + float(duration)
        while ros_seconds_now() < deadline - TIME_EPS:
            self._check_abort()
            self.commander.force_zero()
            self.node.sleep(0.1)

    def _record_step(self, distance: float, angle_deg: float) -> None:
        """累计"计划走过多少路、转了多少度"，用于任务结束后的对比统计。"""
        self.node.stats.distance_commanded += abs(distance)
        self.node.stats.turn_deg_commanded += abs(angle_deg)


# ---------------------------------------------------------------------------
# 节点主体
# ---------------------------------------------------------------------------
class SquareMovementNode:
    """
    节点的组装与调度：初始化参数 -> 建立话题与服务 -> 可选自动开始 ->
    在后台线程里执行任务，主线程留给 rospy.spin() 处理回调。

    任务跑在独立线程里是这一版最重要的结构改动。老脚本从头到尾跑在主线程，
    中间还用 time.sleep() 长时间阻塞，导致运行期间：
        * 收不到任何外部指令，想让它停只能 Ctrl+C（还会留下残余速度指令）；
        * 无法提供任何服务或状态查询；
        * 一旦某段运动出错，整个程序直接异常退出。
    拆成"主线程收回调 + 子线程跑任务"之后，运动过程中随时可以：
        ros2 风格的服务调用停止/暂停、查询进度、动态切换图形。
    """

    def __init__(self) -> None:
        rospy.init_node(DEFAULT_NODE_NAME, anonymous=False)
        self.cfg = MotionConfig.from_ros_params()
        self.rate = rospy.Rate(self.cfg.publish_rate)

        # --- 功能组件 ---
        self.odom = OdomTracker()
        self.safety = SafetyMonitor(self.cfg)
        self.recorder = TrajectoryRecorder(self.cfg.record_csv)
        self.pub = rospy.Publisher(self.cfg.publish_topic, Twist, queue_size=1)
        self.commander = VelocityCommander(self.cfg, self.pub, self.rate, self.cfg.dry_run)
        self.status_pub = None
        self._last_status_publish = 0.0

        # --- 任务状态 ---
        self.steps: List[MotionStep] = ShapeFactory.build(self.cfg)
        self.stats = MissionStats(shape=self.cfg.shape, laps_requested=self.cfg.laps,
                                  steps_total=len(self.steps) * self.cfg.laps)
        self.stop_requested = threading.Event()
        self.pause_requested = threading.Event()
        self._state_lock = threading.Lock()
        self._state = RobotState.IDLE
        self._mission_start: Optional[float] = None
        self._mission_thread: Optional[threading.Thread] = None
        self._progress = {"lap": 0, "steps_done": 0,
                          "steps_total": len(self.steps) * self.cfg.laps, "current": ""}
        self._shutdown_handled = False
        self._services: Dict[str, object] = {}

        # --- 对外接口 ---
        self._setup_publishers()
        self._setup_subscribers()
        self._setup_services()
        self._status_timer = None
        if self.cfg.status_rate > 0:
            self._status_timer = rospy.Timer(
                rospy.Duration(1.0 / max(self.cfg.status_rate, 0.1)), self._status_tick)
        rospy.on_shutdown(self._on_shutdown)
        rospy.loginfo("节点已就绪：%s", self.cfg.describe())
        rospy.loginfo("支持的图形：%s", ", ".join(ShapeFactory.supported()))

    # ------------------------------------------------------------------
    # 初始化：发布器 / 订阅器 / 服务
    # ------------------------------------------------------------------
    def _setup_publishers(self) -> None:
        """建立状态话题发布器；缺少 std_msgs 时只影响状态上报。"""
        if not _HAS_STD_MSGS:
            rospy.logwarn("缺少 std_msgs，状态话题与动态图形话题不可用")
            return
        self.status_pub = rospy.Publisher(self.cfg.status_topic, String, queue_size=5)
        rospy.loginfo("状态话题：%s", rospy.resolve_name(self.cfg.status_topic))

    def _setup_subscribers(self) -> None:
        """按需订阅里程计/激光/电池/急停/遥控接管/图形指令。"""
        topics = [self.cfg.publish_topic]

        if self.cfg.use_odom and _HAS_NAV_MSGS:
            self.odom_sub = rospy.Subscriber(self.cfg.odom_topic, Odometry, self._on_odom,
                                             queue_size=10)
            topics.append(self.cfg.odom_topic)
        else:
            self.odom_sub = None
            if self.cfg.use_odom:
                rospy.logwarn("缺少 nav_msgs，里程计闭环关闭，使用开环时间控制")

        if self.cfg.use_scan and _HAS_SENSOR_MSGS:
            self.scan_sub = rospy.Subscriber(self.cfg.scan_topic, LaserScan, self._on_scan,
                                             queue_size=5)
            topics.append(self.cfg.scan_topic)
        else:
            self.scan_sub = None
            if self.cfg.use_scan:
                rospy.logwarn("缺少 sensor_msgs，避障功能关闭")

        if self.cfg.use_battery and _HAS_SENSOR_MSGS:
            self.battery_sub = rospy.Subscriber(self.cfg.battery_topic, BatteryState,
                                                self._on_battery, queue_size=1)
            topics.append(self.cfg.battery_topic)
        else:
            self.battery_sub = None

        if _HAS_STD_MSGS:
            self.estop_sub = rospy.Subscriber(self.cfg.estop_topic, Bool, self._on_estop,
                                              queue_size=1)
            self.teleop_sub = rospy.Subscriber(self.cfg.teleop_topic, Bool, self._on_teleop,
                                               queue_size=1)
            self.shape_sub = rospy.Subscriber(self.cfg.shape_command_topic, String,
                                              self._on_shape_command, queue_size=1)
        else:
            self.estop_sub = self.teleop_sub = self.shape_sub = None

        rospy.loginfo("订阅话题：%s", ", ".join(topics))

    def _setup_services(self) -> None:
        """建立四个运行时服务，让任务可以远程启停与查询。"""
        if not _HAS_STD_SRVS:
            rospy.logwarn("缺少 std_srvs，start/stop/pause/status 服务不可用")
            return
        self._services["start"] = rospy.Service("~start", Trigger, self._srv_start)
        self._services["stop"] = rospy.Service("~stop", Trigger, self._srv_stop)
        self._services["pause"] = rospy.Service("~pause", SetBool, self._srv_pause)
        self._services["status"] = rospy.Service("~status", Trigger, self._srv_status)
        rospy.loginfo("服务：~start ~stop ~pause ~status")

    # ------------------------------------------------------------------
    # 话题回调
    # ------------------------------------------------------------------
    def _on_odom(self, msg) -> None:
        """里程计回调：喂给跟踪器，并（可选）写一行轨迹。"""
        self.odom.callback(msg)
        if self.recorder.enabled and self.state == RobotState.RUNNING:
            self.recorder.append(self.odom.snapshot(), self.state)

    def _on_scan(self, msg) -> None:
        self.safety.on_scan(msg)

    def _on_battery(self, msg) -> None:
        self.safety.on_battery(msg)

    def _on_estop(self, msg) -> None:
        """急停回调：除了置位状态，还立刻补发一条零速度。"""
        self.safety.on_estop(msg)
        if self.safety.estop:
            self.commander.force_zero()

    def _on_teleop(self, msg) -> None:
        self.safety.on_teleop(msg)

    def _on_shape_command(self, msg) -> None:
        """
        运行时通过话题切换图形，例如：
            rostopic pub -1 ~/shape_command std_msgs/String "data: 'star'"
        当前正在执行的那一圈不受影响，新图形从下一圈或下次 start 开始生效。
        """
        name = str(msg.data).strip().lower()
        if name in ShapeFactory.SHAPES:
            self.cfg.shape = name
            self.steps = ShapeFactory.build(self.cfg)
            rospy.loginfo("图形已切换为 %s（%d 步）", name, len(self.steps))
        else:
            rospy.logwarn("未知图形 %r，可选：%s", name, ", ".join(ShapeFactory.supported()))

    def _status_tick(self, event=None) -> None:
        """定时器回调：周期发布状态。"""
        self.publish_status()

    # ------------------------------------------------------------------
    # 状态管理
    # ------------------------------------------------------------------
    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    def set_state(self, state: str) -> None:
        """切换状态并记录一次日志（相同状态不重复打印）。"""
        with self._state_lock:
            if self._state == state:
                return
            previous, self._state = self._state, state
        rospy.loginfo("状态：%s -> %s", previous, state)

    def fault(self, reason: str) -> None:
        """进入故障状态并记录错误原因。"""
        self.set_state(RobotState.FAULT)
        rospy.logerr("故障：%s", reason)

    def set_progress(self, lap: int, steps_done: int, steps_total: int, current: str) -> None:
        """更新进度信息（会被状态话题与 ~status 服务读到）。"""
        self._progress.update({"lap": lap, "steps_done": steps_done,
                               "steps_total": steps_total, "current": current})

    def mission_elapsed(self) -> float:
        """本轮任务已经跑了多少秒。"""
        return 0.0 if self._mission_start is None else ros_seconds_now() - self._mission_start

    @property
    def mission_running(self) -> bool:
        return bool(self._mission_thread and self._mission_thread.is_alive())

    # ------------------------------------------------------------------
    # 服务实现
    # ------------------------------------------------------------------
    def _srv_start(self, request=None):
        ok, message = self.start_mission()
        return TriggerResponse(ok, message)

    def _srv_stop(self, request=None):
        self.request_stop()
        return TriggerResponse(True, "停止请求已下发")

    def _srv_pause(self, request):
        """data=true 暂停，data=false 继续；暂停期间机器人停在原地等。"""
        if bool(getattr(request, "data", False)):
            self.pause_requested.set()
            return SetBoolResponse(True, "已暂停")
        self.pause_requested.clear()
        return SetBoolResponse(True, "已恢复")

    def _srv_status(self, request=None):
        """把完整状态以 JSON 文本返回，方便脚本解析。"""
        return TriggerResponse(True, json.dumps(self.status_dict(), ensure_ascii=False))

    # ------------------------------------------------------------------
    # 任务控制
    # ------------------------------------------------------------------
    def start_mission(self) -> Tuple[bool, str]:
        """
        启动一轮任务，返回 (是否成功, 说明)。

        每次启动都会重建步骤队列与统计对象，因此想换图形只要改参数再调
        一次 ~start 即可；上一轮的历史数据不会被混进来。
        """
        if self.mission_running:
            return False, "任务正在运行，请先调用 ~stop"
        if self.safety.estop:
            return False, "急停未解除，拒绝启动"

        self.steps = ShapeFactory.build(self.cfg)
        self.stats = MissionStats(shape=self.cfg.shape,
                                  laps_requested=0 if self.cfg.loop_forever else self.cfg.laps,
                                  steps_total=(0 if self.cfg.loop_forever
                                               else len(self.steps) * self.cfg.laps))
        self.set_progress(0, 0, self.stats.steps_total, self.cfg.shape)
        self.stop_requested.clear()
        self.pause_requested.clear()

        if self.cfg.record_csv:
            self.recorder = TrajectoryRecorder(self.cfg.record_csv)
            self.recorder.open()

        self._mission_start = ros_seconds_now()
        self.set_state(RobotState.RUNNING)
        self._mission_thread = threading.Thread(target=self._run_mission, name="mission",
                                                daemon=True)
        self._mission_thread.start()
        rospy.loginfo("任务启动：%s（%d 步 x %s 圈）", self.cfg.shape, len(self.steps),
                      "无限" if self.cfg.loop_forever else self.cfg.laps)
        return True, "任务已启动"

    def _run_mission(self) -> None:
        """
        后台线程的任务主体。

        所有异常都在这里兜住：运动逻辑内部的异常不会让节点崩溃，最多把状态
        置为 fault，并且一定会在 finally 里停车、关闭轨迹文件。
        """
        # 对应老脚本开头的 time.sleep(1.0)：给发布器/订阅者一点建立连接的时间，
        # 否则前几条速度指令可能丢在"还没连上"的空档里；期间也响应 stop 请求。
        if self.cfg.startup_delay > 0:
            rospy.loginfo("等待 %.1fs 让话题连接建立", self.cfg.startup_delay)
            delay_end = ros_seconds_now() + self.cfg.startup_delay
            while (ros_seconds_now() < delay_end - TIME_EPS
                   and not self.stop_requested.is_set() and not rospy.is_shutdown()):
                self.sleep(0.05)

        executor = MotionExecutor(self)
        try:
            stats = executor.execute_plan(self.steps,
                                          self.cfg.laps if not self.cfg.loop_forever else 1,
                                          loop_forever=self.cfg.loop_forever)
        except rospy.ROSInterruptException:
            stats = self.stats
            stats.abort_reason = "ROS 中断"
        except Exception as exc:  # pragma: no cover - 防御性兜底
            self.fault("未预期异常：%s" % exc)
            stats = self.stats
            stats.abort_reason = "未预期异常：%s" % exc
        finally:
            self.commander.stop(ramped=True)
            self.recorder.close()

        if stats.finished:
            self.set_state(RobotState.FINISHED)
            rospy.loginfo("全部完成：%s，用时 %s，实测路程 %.3fm，计划路程 %.3fm",
                          stats.shape, format_seconds(stats.duration),
                          stats.distance_measured, stats.distance_commanded)
            if stats.distance_commanded > 1e-6:
                rospy.loginfo("路程偏差：%+.3fm（%.1f%%）",
                              stats.distance_measured - stats.distance_commanded,
                              100.0 * (stats.distance_measured - stats.distance_commanded)
                              / stats.distance_commanded)
        elif self.stop_requested.is_set():
            self.set_state(RobotState.IDLE)
            rospy.logwarn("任务已被用户停止")
        else:
            self.set_state(RobotState.FAULT)
            rospy.logerr("任务异常结束：%s", stats.abort_reason or "未知原因")
        self.publish_status(force=True)

    def request_stop(self, join_timeout: float = 5.0) -> None:
        """
        请求停止并等待任务线程退出，确保速度指令确实已归零。

        join 是必要的：如果不等线程结束就返回，调用方（比如 ~stop 服务）
        可能先看到了成功响应，而机器人其实还在几十毫秒的减速过程中。
        """
        self.stop_requested.set()
        self.pause_requested.clear()
        self.set_state(RobotState.STOPPING)
        self.commander.force_zero()
        thread = self._mission_thread
        if thread and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(join_timeout)
        if not self.mission_running:
            self.set_state(RobotState.IDLE)

    # ------------------------------------------------------------------
    # 状态上报
    # ------------------------------------------------------------------
    def status_dict(self) -> Dict[str, object]:
        """汇总一份完整状态：状态机、进度、里程计、安全、统计、关键参数。"""
        progress = dict(self._progress)
        total = progress.get("steps_total") or 0
        progress["percent"] = (round(100.0 * progress.get("steps_done", 0) / total, 1)
                               if total else None)
        return {"ros_time": round(ros_seconds_now(), 3),
                "state": self.state,
                "shape": self.cfg.shape,
                "laps": "inf" if self.cfg.loop_forever else self.cfg.laps,
                "mission_elapsed": round(self.mission_elapsed(), 1),
                "progress": progress,
                "odom": {key: round(value, 4) for key, value in self.odom.snapshot().items()},
                "odom_age": round(self.odom.age, 3),
                "safety": self.safety.summary(),
                "command": {"v": round(self.commander.linear, 3),
                            "w": round(self.commander.angular, 3),
                            "sent": self.commander.sent_count},
                "stats": self.stats.summary(),
                "config": {"use_odom": self.cfg.use_odom, "use_scan": self.cfg.use_scan,
                           "dry_run": self.cfg.dry_run,
                           "forward_speed": self.cfg.forward_speed,
                           "turn_speed": self.cfg.turn_speed,
                           "turn_direction": self.cfg.turn_direction,
                           "obstacle_distance": self.cfg.obstacle_distance}}

    def publish_status(self, force: bool = False) -> None:
        """
        把状态以 JSON 发布到状态话题。

        默认限制在 5Hz 以内：状态话题是给人看的，没必要按控制频率刷屏；
        但步骤切换、任务结束这类关键时刻用 force=True 立即发一次。
        """
        if self.status_pub is None:
            return
        now = ros_seconds_now()
        if not force and now - self._last_status_publish < 0.2:
            return
        self._last_status_publish = now
        message = String()
        message.data = json.dumps(self.status_dict(), ensure_ascii=False)
        try:
            self.status_pub.publish(message)
        except Exception:  # pragma: no cover - 话题异常
            pass

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def sleep(self, duration: float) -> None:
        """
        可被中断的等待，使用 ROS 时间（仿真时钟同样适用）。

        用小步长轮询而不是一次 rospy.sleep(duration)，是为了让"等待 0.5 秒"
        这种调用也能及时响应 stop 请求和节点关闭。

        这里必须带 TIME_EPS 容差：deadline - now 会留下浮点残差（实测约
        1e-15），没有容差时可能出现"剩余时间小于时钟分辨率、循环永远等不到
        终点"的空转，仿真时钟（/use_sim_time）下会直接卡死。
        """
        if duration <= 0:
            return
        deadline = ros_seconds_now() + float(duration)
        while not rospy.is_shutdown():
            remaining = deadline - ros_seconds_now()
            if remaining <= TIME_EPS:
                return
            rospy.sleep(min(0.05, remaining))

    def _on_shutdown(self) -> None:
        """
        节点退出时的兜底动作，对应老脚本缺失的那一块。

        无论是 Ctrl+C、rosnode kill 还是 rospy 因异常触发关闭，这里都会：
            1. 把状态标记为 shutdown，阻止新的运动循环继续下发速度；
            2. 立刻补发一条全零 Twist（多一层保险）；
            3. 等任务线程退出（最多 3 秒）后再关轨迹文件。
        """
        if self._shutdown_handled:
            return
        self._shutdown_handled = True
        self.set_state(RobotState.SHUTDOWN)
        self.stop_requested.set()
        self.commander.force_zero()
        rospy.logwarn("节点退出，已发送停车指令")
        if self._mission_thread and self._mission_thread.is_alive():
            self._mission_thread.join(3.0)
        self.recorder.close()


# ---------------------------------------------------------------------------
# 程序入口
# ---------------------------------------------------------------------------
def main() -> int:
    """
    入口函数：
        1. 建立节点（参数、话题、服务、定时器）；
        2. 按 auto_start 决定是立刻开跑还是等 ~start 服务；
        3. 进入 rospy.spin() 处理回调，任务本身在后台线程里跑；
        4. 无论怎么退出，最后再补一条零速度指令。
    """
    try:
        node = SquareMovementNode()
    except rospy.ROSException as exc:  # pragma: no cover - 例如重复初始化
        print("节点初始化失败：%s" % exc, file=sys.stderr)
        return 1

    if node.cfg.dry_run:
        rospy.logwarn("dry-run 模式：只打印规划与日志，不会真正发布速度指令")
    if node.cfg.record_csv:
        rospy.loginfo("轨迹输出目录：%s", os.path.abspath(node.cfg.record_csv))

    if node.cfg.auto_start:
        ok, message = node.start_mission()
        if not ok:
            rospy.logwarn("自动开始失败：%s", message)
    else:
        rospy.loginfo("auto_start=false，等待调用 ~start 服务")

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logwarn("程序中断")
    finally:
        node.commander.force_zero()
    return 0


if __name__ == "__main__":
    sys.exit(main())
