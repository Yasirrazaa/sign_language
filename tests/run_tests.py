"""Test runner script for running all project tests."""

import unittest
import sys
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestRunner:
    """Class to manage test execution and reporting."""
    
    def __init__(self):
        """Initialize test runner."""
        self.test_modules = [
            'test_data_loader',
            'test_data_preprocessing',
            'test_cnn_lstm',
            'test_video_transformer'
        ]
        
        self.results_dir = Path('tests/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _import_test_module(self, module_name: str) -> unittest.TestLoader:
        """Import a test module dynamically."""
        try:
            return unittest.TestLoader().loadTestsFromName(module_name)
        except Exception as e:
            logger.error(f"Failed to import {module_name}: {str(e)}")
            return None
    
    def _format_test_results(self, result: unittest.TestResult) -> Dict:
        """Format test results into a dictionary."""
        return {
            'total': result.testsRun,
            'passed': result.testsRun - len(result.failures) - len(result.errors),
            'failed': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'failures': [
                {
                    'test': str(test),
                    'message': err
                }
                for test, err in result.failures
            ],
            'error_details': [
                {
                    'test': str(test),
                    'message': err
                }
                for test, err in result.errors
            ]
        }
    
    def _save_test_report(self, results: Dict):
        """Save test results to a JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.results_dir / f'test_report_{timestamp}.json'
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test report saved to: {report_path}")
    
    def _print_test_summary(self, module_name: str, results: Dict):
        """Print a summary of test results."""
        logger.info("\n" + "="*50)
        logger.info(f"Test Results for {module_name}")
        logger.info("="*50)
        logger.info(f"Total Tests: {results['total']}")
        logger.info(f"Passed: {results['passed']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"Errors: {results['errors']}")
        logger.info(f"Skipped: {results['skipped']}")
        
        if results['failures']:
            logger.error("\nFailures:")
            for failure in results['failures']:
                logger.error(f"\n{failure['test']}")
                logger.error(f"{failure['message']}")
        
        if results['error_details']:
            logger.error("\nErrors:")
            for error in results['error_details']:
                logger.error(f"\n{error['test']}")
                logger.error(f"{error['message']}")
    
    def run_tests(self) -> bool:
        """
        Run all tests and generate report.
        
        Returns:
            bool: True if all tests passed, False otherwise
        """
        all_results = {}
        all_passed = True
        
        for module_name in self.test_modules:
            logger.info(f"\nRunning tests from {module_name}...")
            
            # Load tests
            test_suite = self._import_test_module(module_name)
            if test_suite is None:
                all_passed = False
                continue
            
            # Run tests
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(test_suite)
            
            # Format and store results
            formatted_results = self._format_test_results(result)
            all_results[module_name] = formatted_results
            
            # Print summary
            self._print_test_summary(module_name, formatted_results)
            
            # Update overall status
            if result.failures or result.errors:
                all_passed = False
        
        # Save full report
        self._save_test_report(all_results)
        
        # Print final summary
        total_tests = sum(r['total'] for r in all_results.values())
        total_passed = sum(r['passed'] for r in all_results.values())
        total_failed = sum(r['failed'] for r in all_results.values())
        total_errors = sum(r['errors'] for r in all_results.values())
        
        logger.info("\n" + "="*50)
        logger.info("FINAL TEST SUMMARY")
        logger.info("="*50)
        logger.info(f"Total Tests Run: {total_tests}")
        logger.info(f"Total Passed: {total_passed}")
        logger.info(f"Total Failed: {total_failed}")
        logger.info(f"Total Errors: {total_errors}")
        logger.info(f"Overall Status: {'PASSED' if all_passed else 'FAILED'}")
        
        return all_passed

def main():
    """Run all tests."""
    try:
        # Add tests directory to path
        test_dir = Path(__file__).parent
        sys.path.insert(0, str(test_dir))
        
        # Run tests
        runner = TestRunner()
        success = runner.run_tests()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
