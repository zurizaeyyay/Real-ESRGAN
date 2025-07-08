import time
import math

class ProgressTracker:
    """Handles progress tracking and message formatting"""
    def __init__(self, callback=None):
        self.callback = callback
        self.start_time = time.time()
        self.last_progress_percent = 0
    
    def update(self, progress, processed, total, stage, remaining=None, elapsed=None):
        """Update progress with smart formatting"""
        if self.callback is None:
            return
        
        # Only update at 5% intervals during processing
        if stage == "processing":
            current_progress_percent = int((processed / total) * 100)
            if current_progress_percent < self.last_progress_percent + 5:
                return
            self.last_progress_percent = current_progress_percent
        
        # Format message based on stage
        message = self._format_message(stage, processed, total, remaining, elapsed)
        self.callback(progress, message)
    
    def _format_message(self, stage, processed, total, remaining=None, elapsed=None):
        """Lazy string formatting"""
        if stage == "preparing":
            return "Preparing image..."
        elif stage == "processing":
            if remaining is not None and remaining > 0:
                return f"Processing {processed}/{total} img parts (est. {remaining:.1f}s remaining)"
            else:
                return f"Processing {processed}/{total} img parts..."
        elif stage == "reconstructing":
            return "Reconstructing image..."
        elif stage == "finalizing":
            return "Finalizing..."
        elif stage == "complete":
            if elapsed is not None:
                return f"Complete! ({elapsed:.1f}s)"
            else:
                return "Complete!"
        return "Processing..."
    
    def calculate_remaining_time(self, processed, total):
        """Calculate estimated remaining time"""
        if processed <= 0:
            return None
        
        elapsed = time.time() - self.start_time
        estimated_total = elapsed * (total / processed)
        return max(0, estimated_total - elapsed)