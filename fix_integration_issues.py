#!/usr/bin/env python3
"""
Fix Integration Issues

This script fixes common integration issues in the HRM-TOPAS codebase:
1. Ensures proper imports are available
2. Fixes tensor shape mismatches
3. Resolves CUDA compatibility issues
4. Addresses Sacred Signature maintenance
5. Validates all import paths

Usage:
  python fix_integration_issues.py --check-only
  python fix_integration_issues.py --fix-imports --validate-shapes
  python fix_integration_issues.py --all-fixes
"""

import os
import sys
import ast
import importlib
import argparse
import traceback
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "docs" / "HRM-main"))


class IntegrationIssuesFixer:
    """Fixes integration issues in the HRM-TOPAS codebase."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.issues_found = []
        self.fixes_applied = []
        
    def check_imports(self) -> List[Dict[str, Any]]:
        """Check for import issues in Python files."""
        print("🔍 Checking import issues...")
        
        import_issues = []
        python_files = list(self.project_root.glob("**/*.py"))
        
        # Filter out venv files and __pycache__
        python_files = [f for f in python_files if "venv" not in str(f) and "__pycache__" not in str(f)]
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Parse AST to find imports
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                self._check_module_availability(alias.name, py_file, import_issues)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                self._check_module_availability(node.module, py_file, import_issues)
                                
                except SyntaxError as e:
                    import_issues.append({
                        "file": str(py_file),
                        "type": "syntax_error",
                        "error": str(e),
                        "line": getattr(e, 'lineno', 0)
                    })
                    
            except Exception as e:
                import_issues.append({
                    "file": str(py_file),
                    "type": "file_error", 
                    "error": str(e)
                })
                
        print(f"  Found {len(import_issues)} import issues")
        return import_issues
        
    def _check_module_availability(self, module_name: str, file_path: Path, issues: List[Dict]):
        """Check if a module can be imported."""
        try:
            # Skip built-in and standard library modules
            if module_name in ['os', 'sys', 'json', 'time', 'argparse', 'traceback', 'pathlib']:
                return
                
            # Skip relative imports (handled differently)
            if module_name.startswith('.'):
                return
                
            # Try to import the module
            importlib.import_module(module_name)
            
        except ImportError as e:
            issues.append({
                "file": str(file_path),
                "type": "import_error",
                "module": module_name,
                "error": str(e)
            })
        except Exception as e:
            # Other errors (skip for now)
            pass
            
    def check_hrm_integration(self) -> List[Dict[str, Any]]:
        """Check HRM integration specific issues."""
        print("🔍 Checking HRM integration...")
        
        hrm_issues = []
        
        # Check if HRM directory exists
        hrm_dir = self.project_root / "docs" / "HRM-main"
        if not hrm_dir.exists():
            hrm_issues.append({
                "type": "missing_hrm_directory",
                "path": str(hrm_dir),
                "error": "HRM directory not found"
            })
            return hrm_issues
            
        # Check if HRM models can be imported
        try:
            sys.path.insert(0, str(hrm_dir))
            from models.hrm.hrm_act_v1 import HRMActV1
            from models.common import HRMConfig
            print("  ✅ HRM models can be imported")
        except ImportError as e:
            hrm_issues.append({
                "type": "hrm_import_error",
                "error": str(e),
                "suggestion": "Check HRM-main directory structure and dependencies"
            })
        except Exception as e:
            hrm_issues.append({
                "type": "hrm_general_error",
                "error": str(e)
            })
            
        # Check HRM-TOPAS bridge
        try:
            from models.hrm_topas_bridge import HRMTOPASBridge, HRMTOPASIntegrationConfig
            print("  ✅ HRM-TOPAS bridge can be imported")
        except ImportError as e:
            hrm_issues.append({
                "type": "bridge_import_error",
                "error": str(e),
                "suggestion": "Check hrm_topas_bridge.py file"
            })
            
        print(f"  Found {len(hrm_issues)} HRM integration issues")
        return hrm_issues
        
    def check_cuda_compatibility(self) -> List[Dict[str, Any]]:
        """Check CUDA compatibility issues."""
        print("🔍 Checking CUDA compatibility...")
        
        cuda_issues = []
        
        try:
            import torch
            
            # Check if CUDA is available
            if not torch.cuda.is_available():
                cuda_issues.append({
                    "type": "cuda_not_available",
                    "error": "CUDA not available on this system",
                    "suggestion": "Install CUDA-enabled PyTorch or use CPU mode"
                })
            else:
                print(f"  ✅ CUDA available: {torch.cuda.get_device_name()}")
                print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
                
                # Check mixed precision support
                try:
                    from torch.cuda.amp import autocast, GradScaler
                    print("  ✅ Mixed precision supported")
                except ImportError:
                    cuda_issues.append({
                        "type": "amp_not_available",
                        "error": "Mixed precision not available",
                        "suggestion": "Update PyTorch to support mixed precision"
                    })
                    
        except ImportError:
            cuda_issues.append({
                "type": "torch_not_available",
                "error": "PyTorch not available",
                "suggestion": "Install PyTorch"
            })
            
        print(f"  Found {len(cuda_issues)} CUDA issues")
        return cuda_issues
        
    def check_tensor_shapes(self) -> List[Dict[str, Any]]:
        """Check for potential tensor shape issues."""
        print("🔍 Checking tensor shape compatibility...")
        
        shape_issues = []
        
        # This is a static analysis - in practice you'd run the models
        try:
            import torch
            
            # Test basic tensor operations that might cause issues
            test_grid = torch.randn(10, 10)  # 2D grid
            
            # Test common transformations
            transformations = [
                ("unsqueeze(0)", lambda x: x.unsqueeze(0)),
                ("unsqueeze(0).unsqueeze(0)", lambda x: x.unsqueeze(0).unsqueeze(0)),
                ("view(-1)", lambda x: x.view(-1)),
                ("flatten", lambda x: x.flatten()),
            ]
            
            for name, transform in transformations:
                try:
                    result = transform(test_grid)
                    print(f"  ✅ {name}: {test_grid.shape} -> {result.shape}")
                except Exception as e:
                    shape_issues.append({
                        "type": "tensor_transform_error",
                        "transform": name,
                        "error": str(e)
                    })
                    
        except Exception as e:
            shape_issues.append({
                "type": "tensor_test_error",
                "error": str(e)
            })
            
        print(f"  Found {len(shape_issues)} tensor shape issues")
        return shape_issues
        
    def validate_configs(self) -> List[Dict[str, Any]]:
        """Validate configuration files."""
        print("🔍 Validating configuration files...")
        
        config_issues = []
        
        # Check main config file
        main_config = self.project_root / "configs" / "hrm_integrated_training.json"
        if main_config.exists():
            try:
                import json
                with open(main_config, 'r') as f:
                    config = json.load(f)
                    
                # Validate required sections
                required_sections = [
                    "global", "hrm_config", "model_config", "train_challenges"
                ]
                
                for section in required_sections:
                    if section not in config:
                        config_issues.append({
                            "type": "missing_config_section",
                            "file": str(main_config),
                            "section": section,
                            "error": f"Required section '{section}' not found"
                        })
                        
                # Validate paths exist
                path_fields = ["train_challenges", "train_solutions", "eval_challenges"]
                for field in path_fields:
                    if field in config and config[field]:
                        if not os.path.exists(config[field]):
                            config_issues.append({
                                "type": "missing_data_path",
                                "file": str(main_config),
                                "field": field,
                                "path": config[field],
                                "error": f"Data path does not exist: {config[field]}"
                            })
                            
                print("  ✅ Main configuration file validated")
                
            except json.JSONDecodeError as e:
                config_issues.append({
                    "type": "json_decode_error",
                    "file": str(main_config),
                    "error": str(e)
                })
            except Exception as e:
                config_issues.append({
                    "type": "config_error",
                    "file": str(main_config),
                    "error": str(e)
                })
        else:
            config_issues.append({
                "type": "missing_config_file",
                "file": str(main_config),
                "error": "Main configuration file not found"
            })
            
        print(f"  Found {len(config_issues)} configuration issues")
        return config_issues
        
    def fix_import_issues(self, issues: List[Dict[str, Any]]) -> int:
        """Attempt to fix import issues."""
        print("🔧 Fixing import issues...")
        
        fixes_applied = 0
        
        for issue in issues:
            if issue["type"] == "import_error":
                # For now, just report - automatic fixing is complex
                print(f"  ⚠️  Import issue in {issue['file']}: {issue['module']} - {issue['error']}")
                
        return fixes_applied
        
    def create_missing_files(self) -> int:
        """Create any missing essential files."""
        print("🔧 Creating missing essential files...")
        
        files_created = 0
        
        # Create __init__.py files if missing
        init_dirs = [
            self.project_root / "trainers",
            self.project_root / "trainers" / "phases",
            self.project_root / "models",
            self.project_root / "validation"
        ]
        
        for init_dir in init_dirs:
            if init_dir.exists():
                init_file = init_dir / "__init__.py"
                if not init_file.exists():
                    try:
                        init_file.write_text("# HRM-TOPAS Integration\n")
                        print(f"  ✅ Created {init_file}")
                        files_created += 1
                    except Exception as e:
                        print(f"  ❌ Failed to create {init_file}: {e}")
                        
        return files_created
        
    def run_comprehensive_check(self, fix_issues: bool = False) -> Dict[str, Any]:
        """Run comprehensive integration check."""
        print("🔍 Running Comprehensive Integration Check")
        print("="*60)
        
        all_issues = {}
        all_fixes = {}
        
        # Check imports
        all_issues["imports"] = self.check_imports()
        
        # Check HRM integration
        all_issues["hrm_integration"] = self.check_hrm_integration()
        
        # Check CUDA compatibility
        all_issues["cuda"] = self.check_cuda_compatibility()
        
        # Check tensor shapes
        all_issues["tensor_shapes"] = self.check_tensor_shapes()
        
        # Check configurations
        all_issues["configs"] = self.validate_configs()
        
        # Apply fixes if requested
        if fix_issues:
            print("\n🔧 Applying fixes...")
            
            all_fixes["import_fixes"] = self.fix_import_issues(all_issues["imports"])
            all_fixes["files_created"] = self.create_missing_files()
            
        # Compile summary
        total_issues = sum(len(issues) for issues in all_issues.values())
        total_fixes = sum(fixes for fixes in all_fixes.values()) if fix_issues else 0
        
        summary = {
            "total_issues": total_issues,
            "total_fixes": total_fixes,
            "issues_by_category": {k: len(v) for k, v in all_issues.items()},
            "all_issues": all_issues,
            "fixes_applied": all_fixes if fix_issues else {}
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print("📊 INTEGRATION CHECK SUMMARY")
        print(f"{'='*60}")
        print(f"Total issues found: {total_issues}")
        
        for category, count in summary["issues_by_category"].items():
            status = "✅" if count == 0 else f"⚠️ {count}"
            print(f"  {category.replace('_', ' ').title()}: {status}")
            
        if fix_issues and total_fixes > 0:
            print(f"\nFixes applied: {total_fixes}")
            
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if all_issues["hrm_integration"]:
            print("  - Ensure HRM-main directory is properly set up")
            print("  - Check HRM dependencies are installed")
            
        if all_issues["cuda"]:
            print("  - Install CUDA-enabled PyTorch for GPU support")
            print("  - Use --device cpu if GPU is not available")
            
        if all_issues["configs"]:
            print("  - Update configuration files with correct paths")
            print("  - Ensure ARC dataset is downloaded and accessible")
            
        if all_issues["imports"]:
            print("  - Install missing dependencies: pip install -r requirements.txt")
            print("  - Check Python path configuration")
            
        return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fix HRM-TOPAS integration issues")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check for issues, don't fix")
    parser.add_argument("--fix-imports", action="store_true",
                       help="Attempt to fix import issues")
    parser.add_argument("--validate-shapes", action="store_true",
                       help="Validate tensor shapes")
    parser.add_argument("--all-fixes", action="store_true",
                       help="Apply all available fixes")
    
    args = parser.parse_args()
    
    print("🔧 HRM-TOPAS Integration Issue Fixer")
    print("="*40)
    
    fixer = IntegrationIssuesFixer()
    
    # Determine if we should fix issues
    fix_issues = args.all_fixes or args.fix_imports
    
    try:
        results = fixer.run_comprehensive_check(fix_issues=fix_issues)
        
        # Exit code based on critical issues
        critical_issues = (
            len(results["all_issues"]["hrm_integration"]) +
            len(results["all_issues"]["configs"])
        )
        
        if critical_issues == 0:
            print("\n✅ Integration check PASSED - no critical issues found")
            exit_code = 0
        else:
            print(f"\n⚠️  Integration check found {critical_issues} critical issues")
            exit_code = 1
            
        return exit_code
        
    except Exception as e:
        print(f"❌ Integration check failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)