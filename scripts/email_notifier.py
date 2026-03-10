import smtplib
import os
import yaml
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.image     import MIMEImage

EMAIL_CONFIG_PATH = os.path.expanduser(
    '~/Documents/camonitor/scripts/email_config.yaml')

def load_email_config():
    if not os.path.exists(EMAIL_CONFIG_PATH):
        print("WARNING: email_config.yaml not found — email disabled.")
        return None
    with open(EMAIL_CONFIG_PATH) as f:
        return yaml.safe_load(f)

class AlertEmailer:
    def __init__(self):
        self.config       = load_email_config()
        self.enabled      = self.config is not None
        self.last_sent    = {}     # alert_type → last sent timestamp
        self.cooldown_sec = 300     # min seconds between same alert type emails

        if self.enabled:
            print(f"Email notifications enabled → {self.config['receiver_email']}")
        else:
            print("Email notifications disabled — email_config.yaml not found.")

    def should_send(self, alert_type):
        """
        Rate limit per alert type — avoid email flooding.
        Same alert type can only send once per cooldown_sec.
        """
        now  = time.time()
        last = self.last_sent.get(alert_type, 0)
        return (now - last) >= self.cooldown_sec

    def send_alert(self, alerts, frame_img_path=None, timestamp_str=''):
        """
        Send alert email for a list of alert dicts.
        Attaches the saved frame image if available.
        Rate limited per alert type.
        """
        if not self.enabled:
            return

        # Filter to alerts that haven't been recently emailed
        to_send = [a for a in alerts if self.should_send(a['type'])]
        if not to_send:
            return

        cfg = self.config
        try:
            msg = MIMEMultipart()
            msg['From']    = cfg['sender_email']
            msg['To']      = cfg['receiver_email']

            # Subject line
            types     = '+'.join(a['type'] for a in to_send)
            severities = '+'.join(a['severity'] for a in to_send)
            msg['Subject'] = f"[CaMonitor] {severities} ALERT — {types}"

            # Email body
            body_lines = [
                "CaMonitor Alert Notification",
                "=" * 40,
                f"Time      : {timestamp_str}",
                f"Alerts    : {len(to_send)}",
                "",
            ]
            for i, alert in enumerate(to_send, 1):
                body_lines.append(f"Alert {i}: [{alert['severity']}] {alert['type']}")
                if 'zone' in alert:
                    body_lines.append(f"  Zone    : {alert['zone']}")
                if 'detail' in alert:
                    body_lines.append(f"  Detail  : {alert['detail']}")
                body_lines.append("")

            body_lines += [
                "=" * 40,
                "CaMonitor — Edge AI Nursery Guard",
                "Raspberry Pi local inference — no cloud processing.",
            ]

            msg.attach(MIMEText('\n'.join(body_lines), 'plain'))

            # Attach frame image if available
            if frame_img_path and os.path.exists(frame_img_path):
                with open(frame_img_path, 'rb') as f:
                    img_data = f.read()
                image = MIMEImage(img_data, name=os.path.basename(frame_img_path))
                image.add_header('Content-Disposition', 'attachment',
                                 filename=os.path.basename(frame_img_path))
                msg.attach(image)

            # Send via Gmail SMTP
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(cfg['sender_email'], cfg['app_password'])
                server.sendmail(cfg['sender_email'],
                                cfg['receiver_email'],
                                msg.as_string())

            # Update last sent times
            for alert in to_send:
                self.last_sent[alert['type']] = time.time()

            sent_types = ', '.join(a['type'] for a in to_send)
            print(f"  [EMAIL] Sent: {sent_types} → {cfg['receiver_email']}")

        except smtplib.SMTPAuthenticationError:
            print("  [EMAIL] ERROR: Authentication failed — check app password.")
        except smtplib.SMTPException as e:
            print(f"  [EMAIL] SMTP error: {e}")
        except Exception as e:
            print(f"  [EMAIL] Failed: {e}")
