#!/usr/bin/env python3
"""Simple test to verify basic functionality without external dependencies."""

import sys
from pathlib import Path
import os

def test_project_structure():
    """Test that the project structure is correct."""
    print("Testing project structure...")
    
    required_dirs = [
        "src/data",
        "src/models", 
        "src/eval",
        "src/viz",
        "configs/data",
        "configs/model",
        "demo",
        "tests",
        "assets/models",
        "assets/plots",
        "assets/maps"
    ]
    
    required_files = [
        "README.md",
        "DISCLAIMER.md",
        "requirements.txt",
        ".gitignore",
        "demo/app.py",
        "scripts/train.py",
        "test_system.py"
    ]
    
    current_dir = Path(__file__).parent
    
    # Check directories
    for dir_path in required_dirs:
        full_path = current_dir / dir_path
        if not full_path.exists():
            print(f"✗ Missing directory: {dir_path}")
            return False
        print(f"✓ Found directory: {dir_path}")
    
    # Check files
    for file_path in required_files:
        full_path = current_dir / file_path
        if not full_path.exists():
            print(f"✗ Missing file: {file_path}")
            return False
        print(f"✓ Found file: {file_path}")
    
    return True

def test_python_files():
    """Test that Python files can be parsed."""
    print("Testing Python file syntax...")
    
    current_dir = Path(__file__).parent
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(current_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    print(f"Found {len(python_files)} Python files")
    
    for py_file in python_files:
        try:
            with open(py_file, 'r') as f:
                content = f.read()
            # Basic syntax check - try to compile
            compile(content, str(py_file), 'exec')
            print(f"✓ {py_file.relative_to(current_dir)}")
        except SyntaxError as e:
            print(f"✗ Syntax error in {py_file.relative_to(current_dir)}: {e}")
            return False
        except Exception as e:
            print(f"✗ Error in {py_file.relative_to(current_dir)}: {e}")
            return False
    
    return True

def test_config_files():
    """Test that configuration files are valid."""
    print("Testing configuration files...")
    
    current_dir = Path(__file__).parent
    
    # Test YAML files
    yaml_files = [
        "configs/model/config.yaml",
        "configs/data/schema.yaml"
    ]
    
    for yaml_file in yaml_files:
        full_path = current_dir / yaml_file
        if full_path.exists():
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                # Basic YAML structure check
                if ':' in content and content.strip():
                    print(f"✓ {yaml_file}")
                else:
                    print(f"✗ Invalid YAML structure in {yaml_file}")
                    return False
            except Exception as e:
                print(f"✗ Error reading {yaml_file}: {e}")
                return False
        else:
            print(f"✗ Missing {yaml_file}")
            return False
    
    return True

def main():
    """Run all tests."""
    print("🌪️ Natural Disaster Prediction System - Structure Test")
    print("=" * 60)
    
    tests = [
        test_project_structure,
        test_python_files,
        test_config_files
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All structure tests passed! Project is properly organized.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run: streamlit run demo/app.py")
        print("3. Run: python scripts/train.py")
        print("4. Check README.md for full instructions")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
