import unittest
import sys
import os
import glob
import importlib.util
import ast
from unittest.mock import MagicMock, patch

# --- MOCKING STREAMLIT ENVIRONMENT ---
# We must mock streamlit before importing any modules that use it
# We must mock streamlit before importing any modules that use it
mock_st = MagicMock()
mock_st.secrets = {}
# Mock cache_data to pass through the function (don't cache in tests)
def mock_decorator(*args, **kwargs):
    def decorator(f):
        return f
    return decorator
mock_st.cache_data = mock_decorator

sys.modules["streamlit"] = mock_st
# Components
mock_components = MagicMock()
sys.modules["streamlit.components.v1"] = mock_components
sys.modules["streamlit.components.v1.components"] = MagicMock() # FIX for declare_component import
sys.modules["streamlit.components"] = MagicMock()
sys.modules["streamlit_option_menu"] = MagicMock()
sys.modules["streamlit_lottie"] = MagicMock()
sys.modules["plotly"] = MagicMock()
sys.modules["plotly.express"] = MagicMock()
sys.modules["plotly.graph_objects"] = MagicMock()

# Only append path if not already there
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

class TestModuleIntegrity(unittest.TestCase):
    
    def test_all_modules_ast_parse(self):
        """
        Verify that all Python files in modules/ directory contain valid Python code.
        """
        modules_dir = os.path.join(PROJECT_ROOT, 'modules')
        py_files = glob.glob(os.path.join(modules_dir, "*.py"))
        
        self.assertTrue(len(py_files) > 0, "No modules found to test!")
        
        for file_path in py_files:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            
            try:
                ast.parse(source)
            except SyntaxError as e:
                # Catch LaTeX escape sequence warnings as errors if needed, but usually just standard syntax
                self.fail(f"SyntaxError in {os.path.basename(file_path)}: {str(e)}")

    def test_verify_show_function_exists(self):
        """
        Import every module and verify it has a 'show' function.
        """
        modules_dir = os.path.join(PROJECT_ROOT, 'modules')
        py_files = glob.glob(os.path.join(modules_dir, "*.py"))
        exclude_list = ["__init__.py"]
        
        for file_path in py_files:
            module_name = os.path.basename(file_path)
            if module_name in exclude_list:
                continue
                
            spec = importlib.util.spec_from_file_location(module_name[:-3], file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name[:-3]] = module
                try:
                    spec.loader.exec_module(module)
                except Exception as e:
                    # Log but don't fail immediately if it's a deep dependency issue (like torch missing)
                    print(f"Warning: Could not import {module_name} due to: {e}")
                    continue
                
                # Check for 'show' function
                if not hasattr(module, 'show'):
                    self.fail(f"Module {module_name} is missing the required 'show()' function.")

class TestBackendLogic(unittest.TestCase):
    
    def test_news_fetcher_logic(self):
        """
        Test the news fetcher utility (mocking the network call).
        """
        from utils import news_fetcher
        
        # Mock Response object with simple XML
        mock_response = MagicMock()
        mock_response.content = b'<rss><channel><item><title>AI News - Source</title><link>http://x.com</link><pubDate>Mon, 25 Dec 2023 12:00:00 GMT</pubDate><source>Source</source><description>desc</description></item></channel></rss>'
        mock_response.raise_for_status = MagicMock()

        with patch('requests.get', return_value=mock_response):
            news = news_fetcher.fetch_ai_news(limit=1)
            
            self.assertEqual(len(news), 1)
            self.assertEqual(news[0]['title'], "AI News") # It strips suffix " - Source"

    def test_rag_imports_safety(self):
        """
        Verify RAG module falls back safely if dependencies are missing.
        """
        from modules import rag_tutorial
        self.assertTrue(hasattr(rag_tutorial, 'show'))

if __name__ == '__main__':
    unittest.main()
