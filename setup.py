"""Setup script for Sign Language Detection project."""

import os
import sys
import subprocess
import venv
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProjectSetup:
    """Class to manage project setup."""
    
    def __init__(self):
        """Initialize setup configuration."""
        self.project_root = Path(__file__).parent.absolute()
        self.venv_dir = self.project_root / 'venv'
        self.required_dirs = [
            'checkpoints',
            'logs',
            'processed',
            'processed/analysis',
            'tests/results',
            'video'
        ]
    
    def create_virtual_environment(self):
        """Create a virtual environment."""
        logger.info("Creating virtual environment...")
        try:
            venv.create(self.venv_dir, with_pip=True)
            logger.info("Virtual environment created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create virtual environment: {str(e)}")
            return False
    
    def get_python_executable(self):
        """Get the virtual environment Python executable."""
        if os.name == 'nt':  # Windows
            return self.venv_dir / 'Scripts' / 'python.exe'
        return self.venv_dir / 'bin' / 'python'
    
    def install_dependencies(self):
        """Install project dependencies."""
        logger.info("Installing dependencies...")
        python_exec = self.get_python_executable()
        
        try:
            # Upgrade pip
            subprocess.run(
                [str(python_exec), '-m', 'pip', 'install', '--upgrade', 'pip'],
                check=True
            )
            
            # Install requirements
            subprocess.run(
                [str(python_exec), '-m', 'pip', 'install', '-r', 'requirements.txt'],
                check=True
            )
            
            logger.info("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {str(e)}")
            return False
    
    def create_directories(self):
        """Create required project directories."""
        logger.info("Creating project directories...")
        try:
            for dir_name in self.required_dirs:
                dir_path = self.project_root / dir_name
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create directories: {str(e)}")
            return False
    
    def setup_git_hooks(self):
        """Set up git pre-commit hooks."""
        logger.info("Setting up git hooks...")
        python_exec = self.get_python_executable()
        
        try:
            subprocess.run(
                [str(python_exec), '-m', 'pre_commit', 'install'],
                check=True
            )
            logger.info("Git hooks installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to set up git hooks: {str(e)}")
            return False
    
    def run_tests(self):
        """Run initial tests."""
        logger.info("Running tests...")
        python_exec = self.get_python_executable()
        
        try:
            result = subprocess.run(
                [str(python_exec), 'tests/run_tests.py'],
                capture_output=True,
                text=True
            )
            
            # Print test output
            print("\nTest Output:")
            print(result.stdout)
            
            if result.stderr:
                print("\nTest Errors:")
                print(result.stderr)
            
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run tests: {str(e)}")
            return False
    
    def setup_project(self):
        """Run complete project setup."""
        logger.info("Starting project setup...")
        
        # Create virtual environment
        if not self.create_virtual_environment():
            return False
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Create directories
        if not self.create_directories():
            return False
        
        # Setup git hooks
        if not self.setup_git_hooks():
            logger.warning("Git hooks setup failed, continuing anyway...")
        
        # Run tests
        if not self.run_tests():
            logger.warning("Some tests failed, please check the test output")
        
        logger.info("\nProject setup completed!")
        logger.info("\nTo activate the virtual environment:")
        if os.name == 'nt':  # Windows
            logger.info(f"    {self.venv_dir}\\Scripts\\activate.bat")
        else:
            logger.info(f"    source {self.venv_dir}/bin/activate")
        
        return True

def main():
    """Run project setup."""
    try:
        setup = ProjectSetup()
        success = setup.setup_project()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
