import requests
import logging
import json
from datetime import datetime

log = logging.getLogger("telegram")

class TelegramNotifier:
    """
    Send surveillance event summaries to Telegram bot.
    """
    
    def __init__(self, bot_token=None, chat_id=None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = False  # ADD THIS
        
        if not bot_token or bot_token == "YOUR_BOT_TOKEN":
            log.warning("[Telegram] ⚠️ Bot token not configured - Telegram disabled")
            self.enabled = False
        else:
            self.enabled = True
            self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
            log.info("[Telegram] ✅ Initialized")
    
    def send_message(self, text, parse_mode="Markdown"):
        """Send text message to Telegram chat."""
        if not self.enabled:
            log.info("[Telegram] Skipped (disabled)")
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                log.info("[Telegram] ✅ Message sent")
                return True
            else:
                log.error(f"[Telegram] Failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            log.error(f"[Telegram] Error: {e}")
            return False
    
    def send_photo(self, photo_path, caption=""):
        """Send photo with caption to Telegram."""
        if not self.enabled:
            log.info("[Telegram] Skipped (disabled)")
            return False
        
        try:
            url = f"{self.base_url}/sendPhoto"
            
            with open(photo_path, 'rb') as photo_file:
                files = {'photo': photo_file}
                data = {
                    'chat_id': self.chat_id,
                    'caption': caption,
                    'parse_mode': 'Markdown'
                }
                
                response = requests.post(url, data=data, files=files, timeout=30)
            
            if response.status_code == 200:
                log.info(f"[Telegram] ✅ Photo sent: {photo_path}")
                return True
            else:
                log.error(f"[Telegram] Photo failed: {response.status_code}")
                return False
                
        except Exception as e:
            log.error(f"[Telegram] Photo error: {e}")
            return False
    
    def send_event_summary(self, video_name, vlm_results, cv_stats, salient_frames):
        if not self.enabled:
            log.info("[Telegram] Event summary skipped (disabled)")
            return True  # Return True so pipeline continues
        
        """
        Send comprehensive event summary to Telegram.
        """
        try:
            # Build summary message
            summary = f"""🎥 *SURVEILLANCE EVENT SUMMARY*
━━━━━━━━━━━━━━━━━━━━━━

📹 *Video:* `{video_name}`
⏰ *Analyzed:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 *CV STATISTICS:*
• Frames analyzed: {len(salient_frames)}
• Total events: {cv_stats.get('total_events', 0)}
• Persons detected: {cv_stats.get('total_persons', 0)}
• Objects tracked: {cv_stats.get('total_objects', 0)}

"""
            
            # Add VLM insights
            if vlm_results:
                summary += "🧠 *VLM INTELLIGENCE ANALYSIS:*\n"
                
                for idx, vlm in enumerate(vlm_results, 1):
                    behavioral = vlm.get('behavioral_assessment', {})
                    scene_intel = vlm.get('scene_intelligence', {})
                    
                    risk = behavioral.get('risk_level', 'unknown')
                    intent = behavioral.get('inferred_intent', 'unknown')
                    confidence = behavioral.get('confidence', 0)
                    
                    risk_emoji = "🔴" if risk == "high" else "🟡" if risk == "medium" else "🟢"
                    
                    summary += f"""
*Frame {idx}:*
{risk_emoji} Risk: `{risk}` | Intent: `{intent}`
📈 Confidence: {confidence:.0%}
📝 {vlm.get('image_description', 'No description')[:100]}...
"""
                    
                    # Add suspicious patterns
                    justification = behavioral.get('justification', [])
                    if justification and any(j for j in justification if j):
                        summary += f"⚠️ Concerns: {', '.join(justification[:2])}\n"
            
            # Add key events
            summary += "\n🎯 *KEY EVENTS:*\n"
            event_types = cv_stats.get('event_types', {})
            for event_type, count in list(event_types.items())[:5]:
                summary += f"• {event_type}: {count}x\n"
            
            # Zone activity
            summary += "\n📍 *ZONE ACTIVITY:*\n"
            zone_activity = cv_stats.get('zone_activity', {})
            for zone, count in list(zone_activity.items())[:5]:
                summary += f"• {zone}: {count} events\n"
            
            summary += "\n━━━━━━━━━━━━━━━━━━━━━━"
            
            # Send message
            self.send_message(summary)
            
            # Send salient frames with captions - INCREASED TO 5
            for idx, frame_meta in enumerate(salient_frames[:5], 1):  # Changed from 3 to 5
                frame_path = frame_meta.get('image_path')
                if frame_path:
                    # Find corresponding VLM result
                    vlm_data = next((v for v in vlm_results if v.get('frame_id') == frame_meta.get('frame_id')), {})
                    behavioral = vlm_data.get('behavioral_assessment', {})
                    
                    caption = f"""🖼 *Frame {frame_meta.get('frame_id')}* ({idx}/5)
Risk: {behavioral.get('risk_level', 'N/A')}
Activity: {behavioral.get('inferred_intent', 'N/A')}
Confidence: {behavioral.get('confidence', 0):.0%}
"""
                    
                    self.send_photo(frame_path, caption)
        
            log.info("[Telegram] ✅ Event summary sent with 5 frames")
            return True
            
        except Exception as e:
            log.error(f"[Telegram] Summary error: {e}", exc_info=True)
            return False
    
    def send_alert(self, alert_type, message, frame_path=None):
        """Send urgent alert notification."""
        alert_emoji = {
            "high_risk": "🚨",
            "suspicious": "⚠️",
            "anomaly": "❗",
            "theft": "🔴",
            "loitering": "👁️"
        }
        
        emoji = alert_emoji.get(alert_type, "📢")
        
        alert_text = f"""{emoji} *SURVEILLANCE ALERT*
━━━━━━━━━━━━━━━━━━━━━━

⚠️ *Type:* `{alert_type.upper()}`
📝 *Details:* {message}
⏰ *Time:* {datetime.now().strftime('%H:%M:%S')}

━━━━━━━━━━━━━━━━━━━━━━
"""
        
        self.send_message(alert_text)
        
        if frame_path:
            self.send_photo(frame_path, f"{emoji} Alert Frame")