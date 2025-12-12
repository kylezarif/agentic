"""
Setup Verification Script
=========================
Checks if all required dependencies are installed correctly.
"""

import sys

def check_import(package_name, import_name=None):
    """Try to import a package and return status."""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✓ {package_name} installed")
        return True
    except ImportError as e:
        print(f"✗ {package_name} NOT installed: {e}")
        return False

def main():
    print("="*60)
    print("DEPENDENCY CHECK")
    print("="*60)

    packages = [
        ("openai", "openai"),
        ("python-dotenv", "dotenv"),
        ("datasets", "datasets"),
        ("deepeval", "deepeval"),
        ("autogen-core", "autogen_core"),
        ("autogen-ext", "autogen_ext"),
        ("langgraph", "langgraph"),
        ("langchain", "langchain"),
        ("langchain-openai", "langchain_openai"),
    ]

    results = []
    for package_name, import_name in packages:
        results.append(check_import(package_name, import_name))

    print("\n" + "="*60)
    print(f"SUMMARY: {sum(results)}/{len(results)} packages installed")
    print("="*60)

    if all(results):
        print("✅ All dependencies installed! Ready to run experiments.")
        return 0
    else:
        print("⚠️  Some dependencies missing. Install with:")
        print("   python3 -m pip install <package-name>")
        return 1

if __name__ == "__main__":
    sys.exit(main())
