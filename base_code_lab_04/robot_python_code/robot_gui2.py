# External libraries
import asyncio
import cv2
import math
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib
from nicegui import ui, app, run
import numpy as np
import os
import time
from fastapi import Response
from fastapi.responses import FileResponse
from time import time

# Local libraries
from robot import Robot
import robot_python_code
import parameters
import particle_filter

# Global variables
logging = False
stream_video = False


# Frame converter for the video stream
def convert(frame: np.ndarray) -> bytes:
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()

def connect_with_camera():
    video_capture = cv2.VideoCapture(1)
    return video_capture

def update_video(video_image):
    if stream_video:
        video_image.force_reload()

def get_time_in_ms():
    return int(time() * 1000)


# ══════════════════════════════════════════════════
# ENHANCED GUI
# ══════════════════════════════════════════════════
@ui.page('/')
def main():

    # ── inject custom CSS for premium look ──
    ui.add_head_html('''
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { font-family: 'Inter', sans-serif !important; }
        body { background: #0a0e14 !important; }

        /* glassmorphism card */
        .glass-card {
            background: rgba(22, 27, 45, 0.75) !important;
            backdrop-filter: blur(16px) !important;
            border: 1px solid rgba(255,255,255,0.06) !important;
            border-radius: 16px !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.35) !important;
            transition: transform 0.22s ease, box-shadow 0.22s ease !important;
        }
        .glass-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.5) !important;
        }

        /* gradient accent bar */
        .accent-bar {
            height: 3px;
            background: linear-gradient(90deg, #6366f1, #06b6d4, #10b981);
            border-radius: 999px;
        }

        /* status dot */
        .status-dot {
            width: 10px; height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
            animation: pulse-dot 2s infinite;
        }
        @keyframes pulse-dot {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .dot-green { background: #10b981; box-shadow: 0 0 8px #10b981; }
        .dot-red   { background: #ef4444; box-shadow: 0 0 8px #ef4444; }
        .dot-amber { background: #f59e0b; box-shadow: 0 0 8px #f59e0b; }

        /* slider accent */
        .q-slider .q-slider__track { background: linear-gradient(90deg, #6366f1, #06b6d4) !important; }

        /* section label */
        .section-label {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: #64748b;
        }

        /* stat value */
        .stat-value {
            font-size: 22px;
            font-weight: 700;
            color: #e2e8f0;
            font-variant-numeric: tabular-nums;
        }
        .stat-unit {
            font-size: 12px;
            color: #64748b;
            margin-left: 4px;
        }

        /* button styling */
        .btn-primary {
            background: linear-gradient(135deg, #6366f1, #818cf8) !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
            transition: all 0.2s ease !important;
        }
        .btn-primary:hover {
            transform: scale(1.03);
            box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
        }
        .btn-secondary {
            background: linear-gradient(135deg, #10b981, #34d399) !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }
        .btn-danger {
            background: linear-gradient(135deg, #ef4444, #f87171) !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }

        .nicegui-content { padding: 0 !important; }
    </style>
    ''')

    # ── Robot variables ──
    robot = Robot()

    # Lidar data
    max_lidar_range = 12
    lidar_angle_res = 2
    num_angles = int(360 / lidar_angle_res)
    lidar_distance_list = [max_lidar_range] * num_angles
    lidar_cos_angle_list = [math.cos(i * lidar_angle_res / 180 * math.pi) for i in range(num_angles)]
    lidar_sin_angle_list = [math.sin(i * lidar_angle_res / 180 * math.pi) for i in range(num_angles)]

    dark = ui.dark_mode()
    dark.value = True

    if stream_video:
        video_capture = cv2.VideoCapture(parameters.camera_id)

    @app.get('/video/frame')
    async def grab_video_frame() -> Response:
        if not video_capture.isOpened():
            return placeholder
        _, frame = await run.io_bound(video_capture.read)
        if frame is None:
            return placeholder
        jpeg = await run.cpu_bound(convert, frame)
        return Response(content=jpeg, media_type='image/jpeg')

    # ── helper functions ──
    def update_lidar_data():
        for i in range(robot.robot_sensor_signal.num_lidar_rays):
            distance_in_mm = robot.robot_sensor_signal.distances[i]
            angle = 360 - robot.robot_sensor_signal.angles[i]
            if distance_in_mm > 20 and abs(angle) < 360:
                index = max(0, min(int(360 / lidar_angle_res - 1), int((angle - (lidar_angle_res / 2)) / lidar_angle_res)))
                lidar_distance_list[index] = distance_in_mm / 1000

    def update_commands():
        if robot.running_trial:
            delta_time = get_time_in_ms() - robot.trial_start_time
            if delta_time > parameters.trial_time:
                robot.running_trial = False
                speed_switch.value = False
                steering_switch.value = False
                robot.extra_logging = True
        if robot.extra_logging:
            delta_time = get_time_in_ms() - robot.trial_start_time
            if delta_time > parameters.trial_time + parameters.extra_trial_log_time:
                logging_switch.value = False
                robot.extra_logging = False

        cmd_speed = slider_speed.value if speed_switch.value else 0
        cmd_steering_angle = slider_steering.value if steering_switch.value else 0
        return cmd_speed, cmd_steering_angle

    def update_connection_to_robot():
        if udp_switch.value:
            if not robot.connected_to_hardware:
                udp, udp_success = robot_python_code.create_udp_communication(
                    parameters.arduinoIP, parameters.localIP, parameters.arduinoPort, parameters.localPort, parameters.bufferSize)
                if udp_success:
                    robot.setup_udp_connection(udp)
                    robot.connected_to_hardware = True
                else:
                    udp_switch.value = False
                    robot.connected_to_hardware = False
        else:
            if robot.connected_to_hardware:
                robot.eliminate_udp_connection()
                robot.connected_to_hardware = False

    def run_trial():
        robot.trial_start_time = get_time_in_ms()
        robot.running_trial = True
        steering_switch.value = True
        speed_switch.value = True
        logging_switch.value = True

    # ── LIDAR radar plot ──
    def show_lidar_plot():
        with lidar_plot:
            fig = lidar_plot.fig
            fig.patch.set_facecolor('#0f1729')
            plt.clf()
            ax = fig.add_subplot(111, polar=True)
            ax.set_facecolor('#0f1729')
            angles = [i * lidar_angle_res * math.pi / 180 for i in range(num_angles)]
            ax.fill(angles, lidar_distance_list, alpha=0.15, color='#06b6d4')
            ax.plot(angles, lidar_distance_list, color='#06b6d4', linewidth=1.2, alpha=0.9)
            ax.set_rmax(2.5)
            ax.set_rticks([0.5, 1.0, 1.5, 2.0])
            ax.tick_params(colors='#475569', labelsize=7)
            ax.grid(True, color='#1e293b', alpha=0.6)
            ax.set_title('LIDAR', color='#94a3b8', fontsize=10, pad=12, fontweight='600')
            lidar_plot.update()

    # ── Particle filter map plot ──
    def show_pf_plot():
        with pf_plot:
            fig = pf_plot.fig
            fig.patch.set_facecolor('#0f1729')
            plt.clf()
            ax = fig.add_subplot(111)
            ax.set_facecolor('#0f1729')

            map_obj = robot.particle_filter.map

            # Walls
            for wall in map_obj.wall_list:
                ax.plot([wall.corner1.x, wall.corner2.x],
                        [wall.corner1.y, wall.corner2.y],
                        color='#e2e8f0', linewidth=2)

            # LIDAR rays from mean
            mean = robot.particle_filter.particle_set.mean_state
            for i in range(robot.robot_sensor_signal.num_lidar_rays):
                d = robot.robot_sensor_signal.distances[i] / 1000
                a = -robot.robot_sensor_signal.angles[i] * math.pi / 180 + mean.theta
                if 0.02 < d < 8.0:
                    ax.plot([mean.x, mean.x + d * math.cos(a)],
                            [mean.y, mean.y + d * math.sin(a)],
                            color='#ef4444', linewidth=0.4, alpha=0.3)

            # Particles
            particle_list = robot.particle_filter.particle_set.particle_list
            print(f"DEBUG: PF Particle List Length: {len(particle_list)}")
            px = [p.state.x for p in particle_list]
            py = [p.state.y for p in particle_list]
            ax.scatter(px, py, s=5, color='#10b981', alpha=0.6, zorder=3)

            # Mean estimate
            ax.plot(mean.x, mean.y, 'o', color='#f59e0b', markersize=7, zorder=5)
            dx = 0.12 * math.cos(mean.theta)
            dy = 0.12 * math.sin(mean.theta)
            ax.annotate('', xy=(mean.x + dx, mean.y + dy), xytext=(mean.x, mean.y),
                        arrowprops=dict(arrowstyle='->', color='#f59e0b', lw=2))

            ax.set_xlim(map_obj.plot_range[0], map_obj.plot_range[1])
            ax.set_ylim(map_obj.plot_range[2], map_obj.plot_range[3])
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.1, color='#334155')
            ax.tick_params(colors='#475569', labelsize=7)
            ax.set_xlabel('X (m)', color='#64748b', fontsize=8)
            ax.set_ylabel('Y (m)', color='#64748b', fontsize=8)
            ax.set_title('Particle Filter Map', color='#94a3b8', fontsize=10, pad=10, fontweight='600')
            for spine in ax.spines.values():
                spine.set_color('#1e293b')
            pf_plot.update()

    # ── update stat labels ──
    def update_stats():
        mean = robot.particle_filter.particle_set.mean_state
        encoder_label.set_text(f'{robot.robot_sensor_signal.encoder_counts}')
        pf_x_label.set_text(f'{mean.x:.3f}')
        pf_y_label.set_text(f'{mean.y:.3f}')
        pf_th_label.set_text(f'{math.degrees(mean.theta):.1f}°')
        particle_count_label.set_text(f'{parameters.num_particles}')

        if robot.connected_to_hardware:
            conn_dot.classes(remove='dot-red dot-amber', add='dot-green')
            conn_text.set_text('Connected')
        else:
            conn_dot.classes(remove='dot-green dot-amber', add='dot-red')
            conn_text.set_text('Disconnected')

    # ══════════════════════════════════════
    # PAGE LAYOUT
    # ══════════════════════════════════════

    with ui.column().classes('w-full min-h-screen').style('padding: 20px 28px; gap: 20px;'):

        # ── Header ──
        with ui.row().classes('w-full items-center justify-between').style('margin-bottom: 4px;'):
            with ui.row().classes('items-center gap-4'):
                ui.html('<div style="font-size:28px; font-weight:700; background:linear-gradient(135deg,#6366f1,#06b6d4); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">ROB-GY 6213</div>')
                ui.label('Robot Navigation & Localization — Lab 4').style('color:#64748b; font-size:14px; font-weight:400;')
            with ui.row().classes('items-center gap-2'):
                conn_dot = ui.html('<span class="status-dot dot-red"></span>')
                conn_text = ui.label('Disconnected').style('color:#94a3b8; font-size:13px;')

        # gradient accent bar
        ui.html('<div class="accent-bar"></div>')

        # ── Main three-panel row ──
        with ui.row().classes('w-full').style('gap: 20px; flex-wrap: nowrap;'):

            # LEFT: LIDAR
            with ui.card().classes('glass-card').style('flex: 1; min-width: 280px; padding: 16px;'):
                ui.label('LIDAR SCAN').classes('section-label')
                lidar_plot = ui.pyplot(figsize=(3.5, 3.5)).style('margin-top: 8px;')

            # CENTER: PF Map
            with ui.card().classes('glass-card').style('flex: 1.6; min-width: 380px; padding: 16px;'):
                ui.label('PARTICLE FILTER — MAP VIEW').classes('section-label')
                pf_plot = ui.pyplot(figsize=(5, 4)).style('margin-top: 8px;')

            # RIGHT: Status Dashboard
            with ui.card().classes('glass-card').style('flex: 0.8; min-width: 220px; padding: 20px;'):
                ui.label('STATUS').classes('section-label')
                ui.html('<div style="height:12px"></div>')

                # Encoder
                with ui.column().style('gap: 2px; margin-bottom: 16px;'):
                    ui.label('Encoder Count').style('color:#64748b; font-size:11px; font-weight:500;')
                    encoder_label = ui.label('0').classes('stat-value')

                # Particles
                with ui.column().style('gap: 2px; margin-bottom: 16px;'):
                    ui.label('Particles').style('color:#64748b; font-size:11px; font-weight:500;')
                    particle_count_label = ui.label(str(parameters.num_particles)).classes('stat-value')

                ui.separator().style('background:#1e293b; margin: 8px 0;')

                ui.label('PF ESTIMATE').classes('section-label')
                ui.html('<div style="height:8px"></div>')
                with ui.row().classes('w-full').style('gap: 16px;'):
                    with ui.column().style('gap:2px; flex:1;'):
                        ui.label('X').style('color:#64748b; font-size:11px;')
                        pf_x_label = ui.label('0.000').classes('stat-value').style('font-size:18px;')
                        ui.label('m').classes('stat-unit')
                    with ui.column().style('gap:2px; flex:1;'):
                        ui.label('Y').style('color:#64748b; font-size:11px;')
                        pf_y_label = ui.label('0.000').classes('stat-value').style('font-size:18px;')
                        ui.label('m').classes('stat-unit')
                with ui.column().style('gap:2px; margin-top:12px;'):
                    ui.label('θ (Heading)').style('color:#64748b; font-size:11px;')
                    pf_th_label = ui.label('0.0°').classes('stat-value').style('font-size:18px;')

                ui.separator().style('background:#1e293b; margin: 16px 0;')

                # Camera / image
                if stream_video:
                    video_image = ui.interactive_image('/video/frame').classes('w-full').style('border-radius:10px;')
                else:
                    ui.image('./a_robot_image.jpg').classes('w-full').style('border-radius:10px; max-height:160px; object-fit:cover;')
                    video_image = None

        # ── Controls bar ──
        with ui.card().classes('glass-card w-full').style('padding: 16px 24px;'):
            ui.label('CONTROLS').classes('section-label').style('margin-bottom: 10px;')
            with ui.row().classes('w-full items-center').style('gap: 28px; flex-wrap: wrap;'):

                # Speed
                with ui.column().style('flex:1; min-width:200px; gap:4px;'):
                    ui.label('Speed').style('color:#94a3b8; font-size:12px; font-weight:600;')
                    with ui.row().classes('items-center w-full').style('gap:12px;'):
                        slider_speed = ui.slider(min=0, max=100, value=0).style('flex:1;')
                        speed_val = ui.label('0').style('color:#e2e8f0; font-size:14px; min-width:30px; font-weight:600; font-variant-numeric:tabular-nums;')
                        slider_speed.on('update:model-value', lambda e: speed_val.set_text(str(int(e.args))))
                        speed_switch = ui.switch('Enable').style('color:#94a3b8;')

                # Steering
                with ui.column().style('flex:1; min-width:200px; gap:4px;'):
                    ui.label('Steering').style('color:#94a3b8; font-size:12px; font-weight:600;')
                    with ui.row().classes('items-center w-full').style('gap:12px;'):
                        slider_steering = ui.slider(min=-20, max=20, value=0).style('flex:1;')
                        steer_val = ui.label('0').style('color:#e2e8f0; font-size:14px; min-width:30px; font-weight:600; font-variant-numeric:tabular-nums;')
                        slider_steering.on('update:model-value', lambda e: steer_val.set_text(str(int(e.args))))
                        steering_switch = ui.switch('Enable').style('color:#94a3b8;')

                # Action buttons
                with ui.column().style('gap:8px; min-width:180px;'):
                    with ui.row().style('gap:8px;'):
                        logging_switch = ui.switch('Data Logging').style('color:#94a3b8;')
                        udp_switch = ui.switch('Robot Connect').style('color:#94a3b8;')
                    with ui.row().style('gap:8px;'):
                        ui.button('Run Trial', on_click=lambda: run_trial()).classes('btn-primary')
                        ui.button('Generate GIF', on_click=lambda: run_gif_generation()).classes('btn-secondary')

    # ── GIF generation (runs in background) ──
    async def run_gif_generation():
        ui.notify('🎬 Generating particle filter GIF...', type='info', timeout=5000)
        await run.io_bound(_do_gif)
        gif_path = os.path.join(os.path.dirname(__file__), 'pf_animation.gif')
        if os.path.exists(gif_path):
            ui.notify('✅ GIF saved as pf_animation.gif', type='positive', timeout=8000)
        else:
            ui.notify('❌ GIF generation failed', type='negative', timeout=5000)

    def _do_gif():
        import subprocess, sys
        subprocess.run([sys.executable, 'particle_filter.py', '--gif'], cwd=os.path.dirname(__file__))

    # ── Control loop ──
    async def control_loop():
        update_connection_to_robot()
        cmd_speed, cmd_steering_angle = update_commands()
        robot.control_loop(cmd_speed, cmd_steering_angle, logging_switch.value)
        update_lidar_data()
        show_lidar_plot()
        show_pf_plot()
        update_stats()

    ui.timer(0.15, control_loop)


# Run
ui.run(title='Lab 4 — Particle Filter', dark=True, port=8080)