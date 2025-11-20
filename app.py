import cv2
import numpy as np
from PIL import Image
import gradio as gr
from ultralytics import YOLO
import os
import time
import asyncio
import smtplib
import socket
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import threading
from queue import Queue
import requests
import json
from datetime import datetime
import logging

# Import your custom modules
from detection import DetectionSystem
from camera import CameraManager
from utils import to_jpg_bytes, FPSCounter, resize_frame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('security_alerts.log'),
        logging.StreamHandler()
    ]
)

class AlertSystem:
    """Unified alert system with multiple notification methods"""
    
    def __init__(self):
        self.email_sender = None
        self.telegram_bot_token = None
        self.telegram_chat_id = None
        self.alert_methods = []
        self.alert_history = []
        
    def setup_email(self, smtp_server, smtp_port, username, password, recipient):
        """Setup email alerts"""
        try:
            self.email_sender = EmailSender(smtp_server, smtp_port, username, password, recipient)
            self.alert_methods.append('email')
            logging.info("✅ Email alerts configured")
            return True, "Email alerts configured"
        except Exception as e:
            logging.error(f"❌ Email setup failed: {e}")
            return False, f"Email setup failed: {e}"
    
    def setup_telegram(self, bot_token, chat_id):
        """Setup Telegram alerts"""
        try:
            self.telegram_bot_token = bot_token
            self.telegram_chat_id = chat_id
            self.alert_methods.append('telegram')
            logging.info("✅ Telegram alerts configured")
            return True, "Telegram alerts configured"
        except Exception as e:
            logging.error(f"❌ Telegram setup failed: {e}")
            return False, f"Telegram setup failed: {e}"
    
    def send_alert(self, image_bytes, subject, body):
        """Send alert through all configured methods"""
        alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_data = {
            'time': alert_time,
            'subject': subject,
            'body': body,
            'methods_used': []
        }
        
        # Save alert to local file
        self._save_local_alert(alert_time, subject, body, image_bytes)
        
        # Try email
        if 'email' in self.alert_methods and self.email_sender:
            try:
                self.email_sender.enqueue(image_bytes, subject, body)
                alert_data['methods_used'].append('email')
                logging.info("📧 Email alert queued")
            except Exception as e:
                logging.error(f"❌ Email alert failed: {e}")
        
        # Try Telegram
        if 'telegram' in self.alert_methods:
            try:
                self._send_telegram_alert(image_bytes, subject, body)
                alert_data['methods_used'].append('telegram')
                logging.info("📱 Telegram alert sent")
            except Exception as e:
                logging.error(f"❌ Telegram alert failed: {e}")
        
        self.alert_history.append(alert_data)
        return len(alert_data['methods_used']) > 0
    
    def _save_local_alert(self, timestamp, subject, body, image_bytes):
        """Save alert locally as file"""
        try:
            # Create alerts directory
            os.makedirs('security_alerts', exist_ok=True)
            
            # Save image
            filename = f"security_alerts/alert_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
            with open(filename, 'wb') as f:
                f.write(image_bytes)
            
            # Save alert info to log file
            log_entry = f"{timestamp} - {subject}\n{body}\nImage: {filename}\n{'-'*50}\n"
            with open('security_alerts/alerts_log.txt', 'a', encoding='utf-8') as f:
                f.write(log_entry)
                
            logging.info(f"💾 Local alert saved: {filename}")
            
        except Exception as e:
            logging.error(f"❌ Failed to save local alert: {e}")
    
    def _send_telegram_alert(self, image_bytes, subject, body):
        """Send alert via Telegram bot"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return False
        
        try:
            # Send photo with caption
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendPhoto"
            
            files = {'photo': ('alert.jpg', image_bytes, 'image/jpeg')}
            data = {
                'chat_id': self.telegram_chat_id,
                'caption': f"🚨 {subject}\n\n{body}",
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, files=files, data=data, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            logging.error(f"❌ Telegram send failed: {e}")
            return False
    
    def get_alert_status(self):
        """Get status of all alert methods"""
        status = {
            'email': 'email' in self.alert_methods,
            'telegram': 'telegram' in self.alert_methods,
            'local': True,  # Always available
            'recent_alerts': len([a for a in self.alert_history if time.time() - datetime.strptime(a['time'], "%Y-%m-%d %H:%M:%S").timestamp() < 3600])
        }
        return status

class EmailSender:
    def __init__(self, smtp_server, smtp_port, username, password, recipient):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipient = recipient
        self.queue = Queue()
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
    
    def enqueue(self, image_bytes, subject, body):
        """Add email to queue for sending"""
        self.queue.put((image_bytes, subject, body))
    
    def _process_queue(self):
        """Process email queue in background thread"""
        while self.running:
            try:
                if not self.queue.empty():
                    image_bytes, subject, body = self.queue.get()
                    success = self._send_email_simple(image_bytes, subject, body)
                    if not success:
                        logging.warning("📧 Email failed, will retry later")
                        # Re-queue for retry
                        self.queue.put((image_bytes, subject, body))
                        time.sleep(30)  # Wait before retry
                    self.queue.task_done()
                time.sleep(1)
            except Exception as e:
                logging.error(f"📧 Email queue error: {e}")
                time.sleep(5)
    
    def _send_email_simple(self, image_bytes, subject, body):
        """Simple email sending with basic error handling"""
        try:
            socket.setdefaulttimeout(10)
            
            # Try multiple ports
            ports = [587, 465, 25]
            for port in ports:
                try:
                    if port == 465:
                        # SSL connection
                        with smtplib.SMTP_SSL(self.smtp_server, port, timeout=10) as server:
                            server.login(self.username, self.password)
                            
                            msg = MIMEMultipart()
                            msg['Subject'] = subject
                            msg['From'] = self.username
                            msg['To'] = self.recipient
                            
                            text_part = MIMEText(body)
                            msg.attach(text_part)
                            
                            image_part = MIMEImage(image_bytes)
                            image_part.add_header('Content-Disposition', 'attachment', filename='alert.jpg')
                            msg.attach(image_part)
                            
                            server.send_message(msg)
                            logging.info(f"✅ Email sent via SSL (port {port})")
                            return True
                    else:
                        # TLS connection
                        with smtplib.SMTP(self.smtp_server, port, timeout=10) as server:
                            server.ehlo()
                            if server.has_extn('STARTTLS'):
                                server.starttls()
                                server.ehlo()
                            server.login(self.username, self.password)
                            
                            msg = MIMEMultipart()
                            msg['Subject'] = subject
                            msg['From'] = self.username
                            msg['To'] = self.recipient
                            
                            text_part = MIMEText(body)
                            msg.attach(text_part)
                            
                            image_part = MIMEImage(image_bytes)
                            image_part.add_header('Content-Disposition', 'attachment', filename='alert.jpg')
                            msg.attach(image_part)
                            
                            server.send_message(msg)
                            logging.info(f"✅ Email sent via TLS (port {port})")
                            return True
                except Exception as e:
                    logging.warning(f"📧 Port {port} failed: {e}")
                    continue
            
            return False
            
        except Exception as e:
            logging.error(f"📧 Email sending failed: {e}")
            return False
    
    def stop(self):
        """Stop the email worker"""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

class SecurityMonitorApp:
    """
    Advanced AI-Powered Security Monitoring Application
    with real-time object detection and alerting capabilities.
    """
    
    def __init__(self):
        """Initialize the security monitoring application."""
        self.detector = None
        self.camera = CameraManager()
        self.fps_counter = FPSCounter()
        self.alert_system = AlertSystem()
        
        self.current_state = {
            'live_running': False,
            'last_alert': 0,
            'current_detections': [],
            'alert_cooldown': 10,
            'camera_index': 0,
            'stream_status': "🔴 Stopped",
            'model_stats': {},
            'processing_mode': 'balanced',
            'alert_methods': {
                'email': False,
                'telegram': False,
                'local': True
            }
        }
    
    def initialize_detector(self, model_path: str, confidence: float, device: str) -> tuple:
        """Initialize the detection system with error handling."""
        try:
            # Only reinitialize if parameters changed
            if (self.detector is None or 
                self.detector.model_path != model_path or 
                self.detector.confidence != confidence or 
                self.detector.device != device):
                
                self.detector = DetectionSystem(
                    model_path=model_path,
                    confidence=confidence,
                    device=device
                )
                
                if hasattr(self.detector, 'model_loaded') and self.detector.model_loaded:
                    return True, "✅ AI Model loaded successfully!"
                elif hasattr(self.detector, 'model') and self.detector.model is not None:
                    return True, "✅ AI Model loaded successfully!"
                else:
                    return False, "❌ Failed to load AI model"
            return True, "✅ Model already loaded"
            
        except Exception as e:
            return False, f"🚨 Detector initialization failed: {str(e)}"
    
    def setup_alert_system(self, email_config=None, telegram_config=None):
        """Setup the alert system with multiple methods"""
        try:
            # Setup email if provided
            if email_config and all(email_config.values()):
                success, message = self.alert_system.setup_email(
                    email_config['smtp_server'],
                    email_config['smtp_port'],
                    email_config['smtp_user'],
                    email_config['smtp_pass'],
                    email_config['alert_recipient']
                )
                self.current_state['alert_methods']['email'] = success
            
            # Setup Telegram if provided
            if telegram_config and all(telegram_config.values()):
                success, message = self.alert_system.setup_telegram(
                    telegram_config['bot_token'],
                    telegram_config['chat_id']
                )
                self.current_state['alert_methods']['telegram'] = success
            
            status = self.alert_system.get_alert_status()
            active_methods = [method for method, active in status.items() if active and method != 'recent_alerts']
            
            if active_methods:
                return True, f"✅ Alert system configured: {', '.join(active_methods)}"
            else:
                return False, "⚠️ No alert methods configured - using local logging only"
                
        except Exception as e:
            return False, f"❌ Alert system setup failed: {str(e)}"
    
    def save_alert_configuration(self, smtp_server, smtp_port, smtp_user, smtp_pass, alert_recipient,
                               telegram_bot_token, telegram_chat_id):
        """Save alert configuration settings."""
        try:
            # Email configuration
            email_config = {
                'smtp_server': smtp_server.strip(),
                'smtp_port': int(smtp_port) if smtp_port else 587,
                'smtp_user': smtp_user.strip(),
                'smtp_pass': smtp_pass,
                'alert_recipient': alert_recipient.strip()
            }
            
            # Telegram configuration
            telegram_config = {
                'bot_token': telegram_bot_token.strip(),
                'chat_id': telegram_chat_id.strip()
            }
            
            # Setup alert system
            success, message = self.setup_alert_system(email_config, telegram_config)
            
            if success:
                status = self.alert_system.get_alert_status()
                active_methods = [method for method, active in status.items() if active and method != 'recent_alerts']
                
                config_summary = f"✅ **Alert configuration saved!**\n\n"
                config_summary += f"**Active Methods:** {', '.join(active_methods) if active_methods else 'Local logging only'}\n\n"
                
                if status['email']:
                    config_summary += f"📧 **Email:** {smtp_user} → {alert_recipient}\n"
                if status['telegram']:
                    config_summary += f"📱 **Telegram:** Bot configured\n"
                if status['local']:
                    config_summary += f"💾 **Local Logging:** Always active\n"
                
                config_summary += f"\n🔔 **Alerts will be sent via:** {', '.join(active_methods) if active_methods else 'local files'}"
                
                return config_summary
            else:
                return f"⚠️ **Configuration saved with limitations:** {message}\n\n💾 **Local logging is always active** - alerts saved to 'security_alerts' folder"
                
        except Exception as e:
            return f"❌ **Error saving configuration:** {str(e)}"
    
    def test_alert_system(self):
        """Test the alert system"""
        try:
            # Create test image
            test_image = np.zeros((200, 300, 3), dtype=np.uint8)
            cv2.putText(test_image, "TEST ALERT", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(test_image, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            jpg_bytes = to_jpg_bytes(test_image)
            
            # Send test alert
            success = self.alert_system.send_alert(
                jpg_bytes,
                "Security Monitor - Test Alert",
                "This is a test alert from your AI Security Monitor system.\n\nIf you receive this, your alert system is working correctly!"
            )
            
            if success:
                return "✅ Test alert sent! Check your configured alert methods."
            else:
                return "⚠️ Test alert queued but some methods may fail. Check local 'security_alerts' folder."
                
        except Exception as e:
            return f"❌ Test failed: {str(e)}"
    
    def get_alert_status(self):
        """Get current alert system status"""
        status = self.alert_system.get_alert_status()
        
        status_text = "**🔔 Alert System Status:**\n\n"
        status_text += f"📧 **Email Alerts:** {'✅ Active' if status['email'] else '❌ Inactive'}\n"
        status_text += f"📱 **Telegram Alerts:** {'✅ Active' if status['telegram'] else '❌ Inactive'}\n"
        status_text += f"💾 **Local Logging:** ✅ Always Active\n"
        status_text += f"🕐 **Recent Alerts (1h):** {status['recent_alerts']}\n\n"
        status_text += f"**Alert Files:** Check 'security_alerts' folder"
        
        return status_text

    # ... (keep all your existing methods like process_image_interface, single_capture_interface, etc.)
    # Just replace the email sending parts with alert_system.send_alert()

    def process_image_interface(self, image, confidence, device, model_path):
        """Process uploaded images for object detection."""
        if image is None:
            return None, "Please upload an image first"
        
        # Initialize detector
        success, message = self.initialize_detector(model_path, confidence, device)
        if not success:
            return image, message
        
        try:
            # Process image
            annotated, detections = self.detector.process_image(image)
            self.current_state['current_detections'] = detections
            
            # Convert BGR to RGB for display
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            # Create detection summary
            if detections:
                summary = f"✅ Found {len(detections)} objects:\n\n"
                for i, detection in enumerate(detections[:10]):  # Limit to 10
                    summary += f"{i+1}. **{detection['label']}** ({(detection['confidence']*100):.1f}%)\n"
                    summary += f"   Position: ({detection['x1']}, {detection['y1']}) to ({detection['x2']}, {detection['y2']})\n"
                    summary += f"   Size: {detection['x2']-detection['x1']}x{detection['y2']-detection['y1']}px\n\n"
            else:
                summary = "🔍 No objects detected"
            
            return annotated_rgb, summary
            
        except Exception as e:
            return image, f"❌ Analysis failed: {e}"
    
    def single_capture_interface(self, confidence, device, model_path, camera_index):
        """Capture and analyze single frame from camera."""
        success, message = self.initialize_detector(model_path, confidence, device)
        if not success:
            return None, message
        
        try:
            if self.camera.start_stream(camera_index):
                # Allow camera to stabilize
                time.sleep(1.0)
                ret, frame = self.camera.get_frame()
                self.camera.stop_stream()
                
                if ret and frame is not None:
                    # Process the captured frame
                    annotated, detections, alert = self.detector.process_frame(frame)
                    self.current_state['current_detections'] = detections
                    
                    # Convert BGR to RGB for display
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    
                    # Create summary
                    if detections:
                        summary = f"✅ Captured & analyzed - Found {len(detections)} objects"
                    else:
                        summary = "✅ Captured - No objects detected"
                    
                    return annotated_rgb, summary
                else:
                    return None, "❌ Failed to capture frame - check camera connection"
            else:
                return None, "❌ Camera unavailable - check camera selection and permissions"
                
        except Exception as e:
            return None, f"❌ Processing failed: {e}"
    
    def start_live_stream(self, confidence, device, model_path, camera_index, enable_alerts, resolution):
        """Start live camera stream."""
        success, message = self.initialize_detector(model_path, confidence, device)
        if not success:
            return None, message, "🔴 Failed"
        
        try:
            # If camera index changed, restart the stream
            if self.current_state['camera_index'] != int(camera_index):
                self.camera.stop_stream()
                self.current_state['camera_index'] = int(camera_index)
            
            if self.camera.start_stream(int(camera_index)):
                self.current_state['live_running'] = True
                self.current_state['stream_status'] = "🟢 Streaming"
                
                if enable_alerts:
                    alert_status = self.alert_system.get_alert_status()
                    active_methods = [method for method, active in alert_status.items() if active and method != 'recent_alerts']
                    
                    if active_methods:
                        status_message = f"✅ Live stream started with alerts: {', '.join(active_methods)}"
                    else:
                        status_message = "✅ Live stream started (Alerts: local logging only)"
                    
                    return self._get_black_frame(), status_message, "🟢 Streaming"
                else:
                    return self._get_black_frame(), "✅ Live stream started (Alerts disabled)", "🟢 Streaming"
            else:
                return self._get_black_frame(), "❌ Failed to start camera stream", "🔴 Failed"
                
        except Exception as e:
            return self._get_black_frame(), f"❌ Stream start failed: {e}", "🔴 Failed"
    
    def stop_live_stream(self):
        """Stop live camera stream."""
        try:
            self.camera.stop_stream()
            self.current_state['live_running'] = False
            self.current_state['stream_status'] = "🔴 Stopped"
            return self._get_black_frame(), "⏹️ Live stream stopped", "🔴 Stopped"
        except Exception as e:
            return self._get_black_frame(), f"❌ Error stopping stream: {e}", "🔴 Error"
    
    def process_live_frame(self, confidence, device, model_path, enable_alerts, alert_cooldown, resolution):
        """Process a single frame for live streaming."""
        if not self.current_state['live_running']:
            return self._get_black_frame(), "Stream not running", "🔴 Stopped"
        
        try:
            # Get frame from camera
            ret, frame = self.camera.get_frame()
            
            if ret and frame is not None and frame.size > 0:
                # Apply resolution settings
                if resolution == "320x240":
                    frame = resize_frame(frame, width=320)
                elif resolution == "1280x720":
                    frame = resize_frame(frame, width=1280)
                
                # Process with detection
                if self.detector and hasattr(self.detector, 'process_frame'):
                    try:
                        annotated, detections, alert_triggered = self.detector.process_frame(frame)
                    except Exception as e:
                        annotated = frame
                        detections = []
                        alert_triggered = False
                else:
                    annotated = frame
                    detections = []
                    alert_triggered = False
                
                # Update FPS counter
                fps = self.fps_counter.update()
                
                # Convert BGR to RGB for display
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                
                # Create status message
                status_msg = f"🎥 Live - {fps:.1f} FPS - {len(detections)} objects"
                
                # Handle alerts
                alert_msg = ""
                if enable_alerts and alert_triggered:
                    current_time = time.time()
                    if (current_time - self.current_state['last_alert']) >= alert_cooldown:
                        self.send_alert(annotated, current_time, detections)
                        self.current_state['last_alert'] = current_time
                        alert_msg = " 🚨 Alert sent!"
                        status_msg += alert_msg
                
                return annotated_rgb, status_msg, "🟢 Streaming"
            else:
                return self._get_black_frame(), "⚠️ No frame received from camera", "🟡 Buffering"
                
        except Exception as e:
            return self._get_black_frame(), f"❌ Stream error: {e}", "🔴 Error"
    
    def _get_black_frame(self):
        """Return a black placeholder frame."""
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def send_alert(self, frame, timestamp, detections):
        """Send alert with captured frame."""
        try:
            jpg_bytes = to_jpg_bytes(frame)
            
            # Create alert message
            detection_summary = ", ".join([det['label'] for det in detections[:3]]) if detections else "No specific objects"
            
            subject = f"🚨 Security Alert - {len(detections)} objects detected"
            body = f"""AI Security System detected movement at {time.ctime(timestamp)}

Detected Objects: {detection_summary}
Total Detections: {len(detections)}

Please check your security feed."""
            
            # Send via alert system
            self.alert_system.send_alert(jpg_bytes, subject, body)
            
        except Exception as e:
            logging.error(f"❌ Alert sending failed: {e}")
    
    def get_camera_info(self):
        """Get camera information for diagnostics."""
        camera_info = self.camera.get_camera_info()
        available_cams = self.camera.get_available_cameras()
        
        info_text = f"**Camera Status:** {camera_info['status']}\n\n"
        info_text += f"**Available Cameras:** {available_cams if available_cams else 'Auto-detecting...'}\n\n"
        info_text += f"**Current Camera:** {self.current_state['camera_index']}\n\n"
        info_text += f"**Stream Status:** {self.current_state['stream_status']}"
        
        return info_text
    
    def continuous_stream_generator(self, confidence, device, model_path, enable_alerts, alert_cooldown, resolution):
        """Generator for continuous livestream frames."""
        while self.current_state['live_running']:
            frame, status, indicator = self.process_live_frame(
                confidence, device, model_path, enable_alerts, alert_cooldown, resolution
            )
            yield frame
            # Small delay to prevent CPU overload
            time.sleep(0.01)
    
    def register_email(self, email_address):
        """Register email address for alerts."""
        if not email_address or '@' not in email_address:
            return "❌ Invalid email address"
        
        try:
            # This is now handled in save_alert_configuration
            return f"✅ Email registered: **{email_address}**\n\n📧 Configure SMTP settings below to enable email alerts."
        except Exception as e:
            return f"❌ Error registering email: {str(e)}"

def create_gradio_interface():
    """Create the Gradio interface for the security monitor."""
    app = SecurityMonitorApp()
    
    with gr.Blocks(theme=gr.themes.Soft(), title="AI Security Monitor") as demo:
        gr.Markdown("# 🔍 AI Security Monitoring System")
        gr.Markdown("*Real-time object detection and security monitoring powered by YOLOv8 AI*")
        
        with gr.Tab("📷 Image Analysis"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        label="Upload Image for Analysis",
                        type="pil",
                        sources=["upload"],
                        height=300
                    )
                    
                    with gr.Row():
                        image_confidence = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                            label="Confidence Threshold"
                        )
                        image_device = gr.Dropdown(
                            choices=["cpu", "0"], value="cpu",
                            label="Processing Device"
                        )
                    
                    image_model_path = gr.Textbox(
                        label="Model Path",
                        value="security_system/security_AI/yolov8n.pt"
                    )
                    
                    image_button = gr.Button("🔍 Analyze Image", variant="primary")
                
                with gr.Column():
                    image_output = gr.Image(
                        label="Detection Results",
                        height=300
                    )
                    image_summary = gr.Markdown(
                        label="Analysis Results",
                        value="Upload an image and click 'Analyze Image'"
                    )
        
        with gr.Tab("📸 Single Capture"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        capture_confidence = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                            label="Confidence Threshold"
                        )
                        capture_device = gr.Dropdown(
                            choices=["cpu", "0"], value="cpu",
                            label="Processing Device"
                        )
                    
                    with gr.Row():
                        capture_camera = gr.Number(
                            value=0, label="Camera Index", precision=0
                        )
                        capture_model_path = gr.Textbox(
                            label="Model Path",
                            value="security_system/security_AI/yolov8n.pt"
                        )
                    
                    capture_button = gr.Button("📷 Capture & Analyze", variant="primary")
                
                with gr.Column():
                    capture_output = gr.Image(
                        label="Captured Frame Analysis",
                        height=300
                    )
                    capture_summary = gr.Markdown(
                        label="Capture Results",
                        value="Click 'Capture & Analyze' to take a picture"
                    )
        
        with gr.Tab("🎥 Live Camera"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        live_confidence = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.20, step=0.05,
                            label="Confidence Threshold"
                        )
                        live_device = gr.Dropdown(
                            choices=["cpu", "0"], value="cpu",
                            label="Processing Device"
                        )
                    
                    with gr.Row():
                        live_camera = gr.Number(
                            value=0, label="Camera Index", precision=0
                        )
                        live_model_path = gr.Textbox(
                            label="Model Path",
                            value="security_system/security_AI/yolov8n.pt"
                        )
                    
                    with gr.Row():
                        live_enable_alerts = gr.Checkbox(
                            label="Enable Smart Alerts", value=False
                        )
                        live_alert_cooldown = gr.Number(
                            value=10, label="Alert Cooldown (seconds)", precision=0
                        )
                    
                    live_resolution = gr.Dropdown(
                        choices=["320x240", "640x480", "1280x720"],
                        value="320x240", label="Stream Resolution"
                    )
                    
                    with gr.Row():
                        start_button = gr.Button("▶️ Start Stream", variant="primary")
                        stop_button = gr.Button("⏹️ Stop Stream", variant="secondary")
                        refresh_button = gr.Button("🔄 Refresh Frame")
                
                with gr.Column():
                    live_output = gr.Image(
                        label="Live Stream",
                        height=300,
                        streaming=True
                    )
                    live_status = gr.Markdown(
                        label="Stream Status",
                        value="**Status:** 🔴 Stopped"
                    )
        
        with gr.Tab("⚙️ System Configuration"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔧 Camera Diagnostics")
                    camera_info = gr.Markdown(
                        label="Camera Information",
                        value="Click 'Refresh Camera Info' to get current status"
                    )
                    refresh_camera = gr.Button("🔄 Refresh Camera Info")
                    
                    gr.Markdown("### 🔔 Alert System Status")
                    alert_status = gr.Markdown(
                        label="Alert Status",
                        value="Click 'Refresh Alert Status' to get current status"
                    )
                    refresh_alerts = gr.Button("🔄 Refresh Alert Status")
                    
                    gr.Markdown("### 📧 Email Alert Configuration")
                    with gr.Row():
                        smtp_server = gr.Textbox(
                            label="SMTP Server",
                            value="smtp.gmail.com",
                            placeholder="e.g., smtp.gmail.com"
                        )
                        smtp_port = gr.Number(
                            label="SMTP Port",
                            value=587,
                            precision=0
                        )
                    
                    with gr.Row():
                        smtp_user = gr.Textbox(
                            label="SMTP Username (Email)",
                            value="philippchukwuebukat33@gmail.com",
                            placeholder="Your email address"
                        )
                        smtp_pass = gr.Textbox(
                            label="SMTP Password",
                            type="password",
                            placeholder="Your app password",
                            value=""
                        )
                    
                    alert_recipient = gr.Textbox(
                        label="Alert Recipient Email",
                        value="philippchukwuebukat33@gmail.com",
                        placeholder="Where alerts will be sent"
                    )
                    
                    gr.Markdown("### 📱 Telegram Alert Configuration")
                    with gr.Row():
                        telegram_bot_token = gr.Textbox(
                            label="Telegram Bot Token",
                            placeholder="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
                            type="password"
                        )
                        telegram_chat_id = gr.Textbox(
                            label="Telegram Chat ID",
                            placeholder="123456789",
                            type="password"
                        )
                    
                    with gr.Row():
                        config_submit = gr.Button("💾 Save Alert Configuration", variant="primary")
                        test_alerts = gr.Button("🧪 Test Alert System", variant="secondary")
                    
                    config_status = gr.Markdown(
                        label="Configuration Status",
                        value="No configuration saved"
                    )
                    
                    gr.Markdown("### 📊 System Information")
                    system_info = gr.Markdown("""
                    **AI Security Monitor System**
                    - **Model:** YOLOv8n
                    - **Default Confidence:** 0.25
                    - **Target Objects:** Person, Animals
                    - **Device:** CPU
                    - **Architecture:** Real-time Processing
                    - **Alert System:** Multiple methods (Email, Telegram, Local)
                    """)
                
                with gr.Column():
                    gr.Markdown("### 🎯 Quick Start Guide")
                    gr.Markdown("""
                    **Getting Started:**
                    1. **Image Analysis:** Upload any image for object detection
                    2. **Single Capture:** Test your camera with single frame capture
                    3. **Live Camera:** Start real-time monitoring
                    4. **Configure alerts** in System Configuration
                    
                    **Alert Setup Options:**
                    
                    **📧 Email (Troubleshooting):**
                    - Use **App Password**, not regular password
                    - Enable 2FA in Google Account
                    - Generate App Password: Security → 2-Step Verification → App passwords
                    
                    **📱 Telegram (Recommended):**
                    1. Message @BotFather on Telegram
                    2. Send `/newbot` command
                    3. Follow instructions to create bot
                    4. Get bot token from @BotFather
                    5. Message your bot and visit:
                       `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
                    6. Find your chat ID in the response
                    
                    **💾 Local Logging (Always Works):**
                    - Alerts saved to 'security_alerts' folder
                    - Includes images and timestamps
                    - No internet required
                    
                    **Troubleshooting:**
                    - Check 'security_alerts' folder for local alerts
                    - Try Telegram for reliable notifications
                    - Use mobile hotspot if network blocks emails
                    """)
        
        # Event handlers for Image Analysis
        image_button.click(
            fn=app.process_image_interface,
            inputs=[image_input, image_confidence, image_device, image_model_path],
            outputs=[image_output, image_summary]
        )
        
        # Event handlers for Single Capture
        capture_button.click(
            fn=app.single_capture_interface,
            inputs=[capture_confidence, capture_device, capture_model_path, capture_camera],
            outputs=[capture_output, capture_summary]
        )
        
        # Event handlers for Live Camera
        start_button.click(
            fn=app.start_live_stream,
            inputs=[live_confidence, live_device, live_model_path, live_camera, 
                   live_enable_alerts, live_resolution],
            outputs=[live_output, live_status, live_status]
        ).then(
            fn=app.continuous_stream_generator,
            inputs=[live_confidence, live_device, live_model_path, live_enable_alerts,
                   live_alert_cooldown, live_resolution],
            outputs=live_output
        )
        
        stop_button.click(
            fn=app.stop_live_stream,
            outputs=[live_output, live_status, live_status]
        )
        
        refresh_button.click(
            fn=app.process_live_frame,
            inputs=[live_confidence, live_device, live_model_path, live_enable_alerts,
                   live_alert_cooldown, live_resolution],
            outputs=[live_output, live_status, live_status]
        )
        
        # Event handlers for System Configuration
        refresh_camera.click(
            fn=app.get_camera_info,
            outputs=[camera_info]
        )
        
        refresh_alerts.click(
            fn=app.get_alert_status,
            outputs=[alert_status]
        )
        
        # Event handler for Save Alert Configuration
        config_submit.click(
            fn=app.save_alert_configuration,
            inputs=[smtp_server, smtp_port, smtp_user, smtp_pass, alert_recipient,
                   telegram_bot_token, telegram_chat_id],
            outputs=[config_status]
        )
        
        # Event handler for Test Alerts
        test_alerts.click(
            fn=app.test_alert_system,
            outputs=[config_status]
        )
        
        # Add some CSS for better styling
        demo.css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        """
    
    return demo

# Application Entry Point
if __name__ == "__main__":
    # Set event loop policy for Windows if needed
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    
    # Create and launch the interface
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        debug=True
    )