import os
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import logging
import sys

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileServer:
    def __init__(self):
        self.config = Config()
        self.server = None
        self.server_thread = None

    def start_server(self):
        """Start HTTP server to serve uploaded files"""
        try:
            # Navigate to data directory to serve files
            data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
            original_dir = os.getcwd()
            os.chdir(data_dir)

            # Create server on localhost
            server_address = ('localhost', self.config.SERVER_PORT)
            self.server = HTTPServer(server_address, SimpleHTTPRequestHandler)

            # Run server in background thread
            def run_server():
                try:
                    logger.info(f"File server started on http://localhost:{self.config.SERVER_PORT}")
                    self.server.serve_forever()
                except Exception as e:
                    logger.error(f"File server error: {str(e)}")
                finally:
                    os.chdir(original_dir)  # Restore original directory

            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            
            logger.info("File server started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start file server: {str(e)}")
            return False

    def get_file_url(self, filename: str, folder: str = "docs_upload") -> str:
        """Generate URL for accessing uploaded files"""
        return f"http://localhost:{self.config.SERVER_PORT}/{folder}/{filename}"

    def is_running(self) -> bool:
        """Check if file server is active"""
        return (self.server is not None and 
                self.server_thread is not None and 
                self.server_thread.is_alive())

    def stop_server(self):
        """Gracefully shutdown file server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
            logger.info("File server stopped")

def main():
    """Standalone file server runner"""
    config = Config()
    config.validate_config()

    file_server = FileServer()
    
    if file_server.start_server():
        logger.info("File server running. Press Ctrl+C to stop.")

        import signal

        def signal_handler(sig, frame):
            logger.info("Shutting down file server...")
            file_server.stop_server()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        try:
            # Keep server running
            while file_server.is_running():
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            file_server.stop_server()

if __name__ == "__main__":
    main()
