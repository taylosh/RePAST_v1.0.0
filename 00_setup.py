"""
00_setup.py - Environment Gatekeeper & Smart Initialization
@author: taylosh
Created on Mar 14 2025
Last edited on Mar 16 2026

Main initialization script for the overhauled ASR pipeline.
- Enforces execution within the 'transcriber' Conda environment.
- Builds necessary directory structures for Phase 1-3.
- Interactive Hugging Face token management.
- Hardware verification via Unified GPU Backend (AMD & NVIDIA support).
- Smart dependency installation with skip-if-exists logic.
- Generates lock file for reproducible environments.
- MFA installation guidance (installs in transcriber environment).
"""

import os
import sys
import platform
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

# Rich console integration
try:
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    class FallbackConsole:
        def print(self, *args, **kwargs): print(*args)
    console = FallbackConsole()

# ============================================================================
# SECTION 1: ENVIRONMENT ENFORCEMENT
# ============================================================================

def check_and_enforce_environment(env_name: str = "transcriber"):
    """Checks if active environment is correct; guides user if not."""
    active_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    
    if active_env == env_name:
        console.print(f"[green]Active environment verified: {env_name}[/green]")
        return True

    console.print(f"[yellow]Current environment '{active_env}' is not '{env_name}'.[/yellow]")
    
    # Check if environment exists
    result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
    
    if env_name in result.stdout:
        console.print(f"\n[red]The '{env_name}' environment exists but is NOT active.[/red]")
        console.print(f"Action Required: Run [bold]'conda activate {env_name}'[/bold] and launch setup again.")
        sys.exit(0)
    else:
        console.print(f"[yellow]Environment '{env_name}' not found. Creating it now...[/yellow]")
        try:
            subprocess.run(['conda', 'create', '-n', env_name, 'python=3.9', '-y'], check=True)
            console.print(f"\n[green]Environment '{env_name}' created successfully.[/green]")
            console.print("\n[bold]NEXT STEPS:[/bold]")
            console.print(f"  1. Activate the environment: [bold]conda activate {env_name}[/bold]")
            console.print("  2. Run this setup script again: [bold]python libs/00_setup.py[/bold]")
            console.print("\nThe environment is empty - the setup script will install all dependencies.")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Failed to create environment: {e}[/red]")
            console.print("Please create manually:")
            console.print(f"  conda create -n {env_name} python=3.9")
        sys.exit(0)

# ============================================================================
# SECTION 2: COMPONENT CHECKER
# ============================================================================

class ComponentChecker:
    """Checks if components are already installed/set up."""
    
    @staticmethod
    def package_installed(package_name: str, min_version: str = None) -> bool:
        """Check if a Python package is installed and meets version."""
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            return False
        
        if min_version:
            try:
                import pkg_resources
                installed = pkg_resources.get_distribution(package_name).version
                from packaging import version
                return version.parse(installed) >= version.parse(min_version)
            except:
                return True  # Can't check version, assume it's fine
        return True
    
    @staticmethod
    def directory_exists(path: Path) -> bool:
        """Check if directory exists."""
        return path.exists()
    
    @staticmethod
    def file_exists(path: Path) -> bool:
        """Check if file exists."""
        return path.is_file()
    
    @staticmethod
    def mfa_installed() -> bool:
        """Check if MFA is available in current environment."""
        try:
            subprocess.run(['mfa', 'version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    @staticmethod
    def c_modules_built() -> bool:
        """Check if C modules are compiled."""
        bin_dir = Path(__file__).parent / "bin"
        if not bin_dir.exists():
            return False
        
        # Look for any compiled module
        extensions = ['.so', '.pyd', '.dylib']
        for ext in extensions:
            if len(list(bin_dir.glob(f'*{ext}'))) > 0:
                return True
        return False

# ============================================================================
# SECTION 3: SETUP TRACKER
# ============================================================================

class SetupTracker:
    """Tracks what was done, what succeeded, what failed."""
    
    def __init__(self):
        self.results = {
            'directories': {'created': [], 'skipped': [], 'failed': []},
            'hf_token': {'status': None, 'message': ''},
            'hardware': {'gpu': None, 'message': ''},
            'pytorch': {'status': None, 'message': ''},
            'packages': {'installed': [], 'skipped': [], 'failed': []},
            'c_modules': {'status': None, 'message': ''},
            'mfa': {'status': None, 'message': ''}
        }
    
    def add_dir_result(self, dir_path: str, status: str):
        """status: 'created', 'skipped', 'failed'"""
        self.results['directories'][status].append(dir_path)
    
    def set_hf_token(self, status: bool, message: str = ''):
        self.results['hf_token']['status'] = status
        self.results['hf_token']['message'] = message
    
    def set_hardware(self, gpu_info: str, message: str = ''):
        self.results['hardware']['gpu'] = gpu_info
        self.results['hardware']['message'] = message
    
    def set_pytorch(self, status: bool, message: str = ''):
        self.results['pytorch']['status'] = status
        self.results['pytorch']['message'] = message
    
    def add_package_result(self, package: str, status: str):
        """status: 'installed', 'skipped', 'failed'"""
        self.results['packages'][status].append(package)
    
    def set_c_modules(self, status: bool, message: str = ''):
        self.results['c_modules']['status'] = status
        self.results['c_modules']['message'] = message
    
    def set_mfa(self, status: bool, message: str = ''):
        self.results['mfa']['status'] = status
        self.results['mfa']['message'] = message
    
    def report(self):
        """Display final report."""
        if RICH_AVAILABLE:
            console.print("\n[bold cyan]════════════════════════════════════════[/bold cyan]")
            console.print("[bold cyan]         SETUP EXECUTION SUMMARY        [/bold cyan]")
            console.print("[bold cyan]════════════════════════════════════════[/bold cyan]\n")
            
            # Directories
            dir_table = Table(title="Directory Setup", box=box.ROUNDED)
            dir_table.add_column("Status", style="bold")
            dir_table.add_column("Count", justify="right")
            dir_table.add_column("Paths", style="dim")
            
            dir_table.add_row("[green]Created[/green]", 
                            str(len(self.results['directories']['created'])),
                            ", ".join(self.results['directories']['created'][:3]) + 
                            ("..." if len(self.results['directories']['created']) > 3 else ""))
            dir_table.add_row("[yellow]Skipped (existed)[/yellow]", 
                            str(len(self.results['directories']['skipped'])),
                            ", ".join(self.results['directories']['skipped'][:3]) + 
                            ("..." if len(self.results['directories']['skipped']) > 3 else ""))
            dir_table.add_row("[red]Failed[/red]", 
                            str(len(self.results['directories']['failed'])),
                            ", ".join(self.results['directories']['failed']))
            console.print(dir_table)
            
            # Token
            token_status = "Present" if self.results['hf_token']['status'] else "Missing"
            token_color = "green" if self.results['hf_token']['status'] else "yellow"
            console.print(f"\n[{token_color}]HuggingFace Token: {token_status}[/{token_color}]")
            if self.results['hf_token']['message']:
                console.print(f"  [dim]{self.results['hf_token']['message']}[/dim]")
            
            # Hardware
            hw_color = "green" if self.results['hardware']['gpu'] else "yellow"
            hw_text = self.results['hardware']['gpu'] or "None detected (CPU mode)"
            console.print(f"\n[{hw_color}]Hardware: {hw_text}[/{hw_color}]")
            
            # PyTorch
            if self.results['pytorch']['status'] is True:
                console.print(f"[green]PyTorch: Installed ({self.results['pytorch']['message']})[/green]")
            elif self.results['pytorch']['status'] is False:
                console.print(f"[red]PyTorch: Failed - {self.results['pytorch']['message']}[/red]")
            else:
                console.print("[yellow]PyTorch: Skipped (already installed)[/yellow]")
            
            # Packages
            pkg_table = Table(title="Python Packages", box=box.ROUNDED)
            pkg_table.add_column("Status", style="bold")
            pkg_table.add_column("Count", justify="right")
            pkg_table.add_column("Packages", style="dim")
            
            pkg_table.add_row("[green]Installed[/green]", 
                            str(len(self.results['packages']['installed'])),
                            ", ".join(self.results['packages']['installed'][:5]) + 
                            ("..." if len(self.results['packages']['installed']) > 5 else ""))
            pkg_table.add_row("[yellow]Skipped (existed)[/yellow]", 
                            str(len(self.results['packages']['skipped'])),
                            ", ".join(self.results['packages']['skipped'][:5]) + 
                            ("..." if len(self.results['packages']['skipped']) > 5 else ""))
            pkg_table.add_row("[red]Failed[/red]", 
                            str(len(self.results['packages']['failed'])),
                            ", ".join(self.results['packages']['failed']))
            console.print(pkg_table)
            
            # C Modules
            c_status = "Built" if self.results['c_modules']['status'] else "Not built"
            c_color = "green" if self.results['c_modules']['status'] else "yellow"
            console.print(f"\n[{c_color}]C Modules: {c_status}[/{c_color}]")
            
            # MFA
            mfa_status = "Installed" if self.results['mfa']['status'] else "Not found"
            mfa_color = "green" if self.results['mfa']['status'] else "yellow"
            console.print(f"[{mfa_color}]MFA: {mfa_status}[/{mfa_color}]")
            if self.results['mfa']['message']:
                console.print(f"  [dim]{self.results['mfa']['message']}[/dim]")
            
        else:
            # Fallback text report
            print("\n=== SETUP SUMMARY ===")
            print(f"Directories created: {len(self.results['directories']['created'])}")
            print(f"Directories skipped: {len(self.results['directories']['skipped'])}")
            print(f"Directories failed: {len(self.results['directories']['failed'])}")
            print(f"HF Token: {'Present' if self.results['hf_token']['status'] else 'Missing'}")
            print(f"Hardware: {self.results['hardware']['gpu'] or 'CPU only'}")
            print(f"PyTorch: {self.results['pytorch']['message'] or 'Skipped'}")
            print(f"Packages installed: {len(self.results['packages']['installed'])}")
            print(f"Packages skipped: {len(self.results['packages']['skipped'])}")
            print(f"Packages failed: {len(self.results['packages']['failed'])}")
            print(f"C Modules: {'Built' if self.results['c_modules']['status'] else 'Not built'}")
            print(f"MFA: {'Installed' if self.results['mfa']['status'] else 'Not found'}")

# ============================================================================
# SECTION 4: MAIN SETUP CLASS
# ============================================================================

class PipelineSetup:
    def __init__(self):
        self.root = Path(__file__).parent
        self.tracker = SetupTracker()
        
        # REVISED: Only directories actually used by the pipeline
        self.required_dirs = [
            # Phase 1
            'original_audio',
            'preprocessed_audio',
            
            # Phase 2
            'initial_transcription/textgrids',
            'initial_transcription/transcripts',
            'initial_transcription/diarization_data',
            
            # Phase 3
            'aligned_textgrids',
            
            # Models (empty - pipeline downloads as needed)
            'models/embeddings',           # For register.py
            'models/syllable_dicts',        # For syllabify.py (optional)
            'models/speechbrain',            # For enhance.py experimental
            
            # System
            'logs',
            'bin',                           # Compiled C modules
            'config'                          # pipeline_config.json
        ]
        
        # Package groups with minimum versions
        self.package_groups = {
            'core': [
                'numpy>=1.24.0,<2.0.0',
                'scipy>=1.11.0',
                'librosa>=0.10.1',
                'soundfile>=0.12.1',
                'requests>=2.31.0',
                'rich>=13.7.1'
            ],
            'enhancement': [
                'noisereduce>=3.0.2',
                'pyloudnorm>=0.1.1',
                'speechbrain>=1.0.0'
            ],
            'transcription': [
                'openai-whisper>=20231117',
                #'pyannote.audio>=3.0.0'
            ],
            'alignment': [
                'praatio>=5.0.0'
            ],
            'linguistic': [
                'spacy>=3.7.0'
            ],
            'espnet': [  # For register.py
                'espnet<202511',
                'espnet-model-zoo'
            ],
            'utils': [
                'psutil>=5.9.0',
                's3prl>=0.4.0',
                'einops>=0.7.0',
                'packaging>=23.0'
            ]
        }
        
        # Platform-specific packages
        if platform.system() == "Windows":
            self.package_groups['windows'] = ['torch-directml']

    def run(self):
        """Main execution flow."""
        console.print("\n=== [bold cyan]PIPELINE SMART SETUP[/bold cyan] ===")
        
        # Step 1: Create directories (check existence first)
        self.create_directories()
        
        # Step 2: Setup HuggingFace token
        self.setup_hf_token()
        
        # Step 3: Verify hardware
        self.verify_hardware()
        
        # Step 4: Install PyTorch (if needed)
        self.install_pytorch()
        
        # Step 5: Install Python packages (skip existing)
        self.install_dependencies()
        
        # Step 6: Build C modules (if needed)
        self.build_c_modules()
        
        # Step 7: Install MFA (in current environment)
        self.install_mfa()
        
        # Step 8: Generate lock file if anything was installed
        if (self.tracker.results['packages']['installed'] or 
            self.tracker.results['pytorch']['status'] or
            self.tracker.results['mfa']['status']):
            self.generate_lock_file()
        
        # Step 9: Show final report
        self.tracker.report()
        
        console.print("\n[bold green]SETUP COMPLETE[/bold green]")
        
        # Final guidance based on what was installed/skipped
        if not self.tracker.results['mfa']['status']:
            console.print("\n[yellow]MFA installation was skipped or failed.[/yellow]")
            console.print("  Phase 3 (alignment) requires MFA. To install manually:")
            console.print(f"  [bold]conda activate transcriber[/bold]")
            console.print("  [bold]conda install -c conda-forge montreal-forced-aligner[/bold]")

    # ========================================================================
    # STEP 1: DIRECTORIES
    # ========================================================================
    
    def create_directories(self):
        """Create only directories that don't already exist."""
        console.print("\n[bold]Checking directories...[/bold]")
        
        for dir_path in self.required_dirs:
            full_path = self.root / dir_path
            
            if full_path.exists():
                self.tracker.add_dir_result(dir_path, 'skipped')
                console.print(f"  [yellow][/yellow] {dir_path} [dim](exists)[/dim]")
            else:
                try:
                    full_path.mkdir(parents=True, exist_ok=True)
                    self.tracker.add_dir_result(dir_path, 'created')
                    console.print(f"  [green][/green] {dir_path} [dim](created)[/dim]")
                except Exception as e:
                    self.tracker.add_dir_result(dir_path, 'failed')
                    console.print(f"  [red][/red] {dir_path} [dim](failed: {e})[/dim]")

    # ========================================================================
    # STEP 2: HUGGINGFACE TOKEN
    # ========================================================================
    
    def setup_hf_token(self):
        """Check for token, prompt if missing."""
        token_file = self.root / "HuggingFaceToken.txt"
        
        if token_file.exists():
            self.tracker.set_hf_token(True, "Token file found")
            console.print(f"\n[green]HuggingFace token found[/green]")
            return
        
        console.print("\n[bold][Hugging Face Token Required][/bold]")
        console.print("Pyannote diarization requires a HuggingFace token.")
        console.print("Get one at: https://huggingface.co/settings/tokens")
        
        if RICH_AVAILABLE:
            token = Prompt.ask("Enter your token", password=True)
        else:
            token = input("Enter your token: ").strip()
        
        if token:
            token_file.write_text(token)
            self.tracker.set_hf_token(True, "Token saved")
            console.print("[green]Token saved[/green]")
        else:
            self.tracker.set_hf_token(False, "No token provided")
            console.print("[yellow]No token provided. Diarization will not work.[/yellow]")

    # ========================================================================
    # STEP 3: HARDWARE VERIFICATION
    # ========================================================================
    
    def verify_hardware(self):
        """Check GPU availability via wrapper."""
        console.print("\n[bold]Verifying hardware...[/bold]")
        
        try:
            sys.path.insert(0, str(self.root / "libs"))
            from wrap_accel import get_c_acceleration_wrapper
            wrapper = get_c_acceleration_wrapper()
            
            if wrapper.available and wrapper.gpu_backend.gpu_is_available():
                gpu_name = wrapper.gpu_backend.get_device_name()
                self.tracker.set_hardware(gpu_name, "GPU acceleration available")
                console.print(f"  [green]GPU detected: {gpu_name}[/green]")
                
                # Store for later use
                self.gpu_wrapper = wrapper
            else:
                self.tracker.set_hardware(None, "CPU mode")
                console.print(f"  [yellow]No compatible GPU, using CPU[/yellow]")
                self.gpu_wrapper = None
                
        except Exception as e:
            self.tracker.set_hardware(None, f"Detection failed: {e}")
            console.print(f"  [yellow]Hardware detection failed: {e}[/yellow]")
            self.gpu_wrapper = None

    # ========================================================================
    # STEP 4: PYTORCH INSTALLATION
    # ========================================================================
    
    def install_pytorch(self):
        """Install PyTorch based on hardware, but only if not already installed."""
        console.print("\n[bold]Checking PyTorch...[/bold]")
        
        # Check if PyTorch is already installed
        if ComponentChecker.package_installed('torch'):
            self.tracker.set_pytorch(None, "Already installed")
            console.print(f"  [yellow]PyTorch already installed[/yellow]")
            return
        
        console.print("  Installing PyTorch...")
        
        # Determine installation command
        if hasattr(self, 'gpu_wrapper') and self.gpu_wrapper:
            try:
                gpu_name = self.gpu_wrapper.gpu_backend.get_device_name().lower()
                
                if 'nvidia' in gpu_name or 'cuda' in gpu_name:
                    cmd = ['pip', 'install', 'torch>=2.0.0', 'torchaudio>=2.0.0']
                    self.tracker.set_pytorch(True, "CUDA version")
                    console.print(f"    [dim]NVIDIA detected - installing CUDA version[/dim]")
                    
                elif 'amd' in gpu_name:
                    cmd = ['pip', 'install', 'torch>=2.0.0', 'torchaudio>=2.0.0', 
                          '--index-url', 'https://download.pytorch.org/whl/rocm5.6']
                    self.tracker.set_pytorch(True, "ROCm version")
                    console.print(f"    [dim]AMD detected - installing ROCm version[/dim]")
                else:
                    cmd = ['pip', 'install', 'torch>=2.0.0', 'torchaudio>=2.0.0', 
                          '--index-url', 'https://download.pytorch.org/whl/cpu']
                    self.tracker.set_pytorch(True, "CPU version")
                    console.print(f"    [dim]Unknown GPU - installing CPU version[/dim]")
            except:
                cmd = ['pip', 'install', 'torch>=2.0.0', 'torchaudio>=2.0.0', 
                      '--index-url', 'https://download.pytorch.org/whl/cpu']
                self.tracker.set_pytorch(True, "CPU version (fallback)")
                console.print(f"    [dim]GPU detection failed - installing CPU version[/dim]")
        else:
            cmd = ['pip', 'install', 'torch>=2.0.0', 'torchaudio>=2.0.0', 
                  '--index-url', 'https://download.pytorch.org/whl/cpu']
            self.tracker.set_pytorch(True, "CPU version")
            console.print(f"    [dim]No GPU - installing CPU version[/dim]")
        
        # Run installation
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            console.print(f"  [green]PyTorch installed[/green]")
        except subprocess.CalledProcessError as e:
            self.tracker.set_pytorch(False, f"Installation failed: {e}")
            console.print(f"  [red]PyTorch installation failed[/red]")

    # ========================================================================
    # STEP 5: PYTHON PACKAGES
    # ========================================================================
    
    def install_dependencies(self):
        """Install Python packages, skipping any that are already installed."""
        console.print("\n[bold]Checking Python packages...[/bold]")
        
        # Flatten package list
        all_packages = []
        for group, packages in self.package_groups.items():
            all_packages.extend(packages)
        
        # Separate into "to install" and "skip"
        to_install = []
        
        for pkg in all_packages:
            # Extract base package name and min version
            if '>=' in pkg:
                base_name = pkg.split('>=')[0]
                min_version = pkg.split('>=')[1].split(',')[0] if ',' in pkg else pkg.split('>=')[1]
            elif '<' in pkg:
                base_name = pkg.split('<')[0]
                min_version = None
            else:
                base_name = pkg
                min_version = None
            
            if ComponentChecker.package_installed(base_name, min_version):
                self.tracker.add_package_result(base_name, 'skipped')
                console.print(f"  [yellow][/yellow] {base_name} [dim](exists)[/dim]")
            else:
                to_install.append(pkg)
                console.print(f"  [dim]→ {pkg} [dim](needed)[/dim]")
        
        if not to_install:
            console.print("  [green]All packages already installed[/green]")
            return
        
        # Ask for confirmation
        if RICH_AVAILABLE:
            if not Confirm.ask(f"Install {len(to_install)} missing packages?"):
                console.print("[yellow]Package installation skipped[/yellow]")
                for pkg in to_install:
                    base_name = pkg.split('>=')[0].split('<')[0]
                    self.tracker.add_package_result(base_name, 'failed')
                return
        
        # Install in batches to avoid command line length limits
        batch_size = 20
        for i in range(0, len(to_install), batch_size):
            batch = to_install[i:i+batch_size]
            console.print(f"  Installing batch {i//batch_size + 1}/{(len(to_install)-1)//batch_size + 1}...")
            
            try:
                subprocess.run(['pip', 'install'] + batch, check=True, capture_output=True)
                for pkg in batch:
                    base_name = pkg.split('>=')[0].split('<')[0]
                    self.tracker.add_package_result(base_name, 'installed')
                console.print(f"    [green]Batch complete[/green]")
            except subprocess.CalledProcessError as e:
                console.print(f"    [red]Batch failed[/red]")
                for pkg in batch:
                    base_name = pkg.split('>=')[0].split('<')[0]
                    self.tracker.add_package_result(base_name, 'failed')
        
        # Install spaCy model separately
        if ComponentChecker.package_installed('en_core_web_sm'):
            console.print(f"  [yellow]en_core_web_sm [dim](exists)[/dim]")
        else:
            console.print(f"  Installing spaCy English model...")
            try:
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                             check=True, capture_output=True)
                console.print(f"    [green]Installed[/green]")
            except:
                console.print(f"    [red]Failed[/red]")

    # ========================================================================
    # STEP 6: C MODULES
    # ========================================================================
    
    def build_c_modules(self):
        """Build C modules if not already built."""
        console.print("\n[bold]Checking C modules...[/bold]")
        
        if ComponentChecker.c_modules_built():
            self.tracker.set_c_modules(True, "Already built")
            console.print(f"  [yellow]C modules already built[/yellow]")
            return
        
        build_script = self.root / "libs" / "build_accel.py"
        if not build_script.exists():
            self.tracker.set_c_modules(False, "build_accel.py not found")
            console.print(f"  [red]build_accel.py not found[/red]")
            return
        
        console.print(f"  Building C modules...")
        try:
            subprocess.run([sys.executable, str(build_script)], check=True, capture_output=True)
            
            if ComponentChecker.c_modules_built():
                self.tracker.set_c_modules(True, "Built successfully")
                console.print(f"    [green]Build complete[/green]")
            else:
                self.tracker.set_c_modules(False, "Build completed but modules not found")
                console.print(f"    [yellow]Build completed but modules not found[/yellow]")
        except subprocess.CalledProcessError as e:
            self.tracker.set_c_modules(False, f"Build failed: {e}")
            console.print(f"    [red]Build failed[/red]")

    # ========================================================================
    # STEP 7: MFA INSTALLATION (IN CURRENT ENVIRONMENT)
    # ========================================================================
    
    def install_mfa(self):
        """Install Montreal Forced Aligner via conda-forge in current environment."""
        console.print("\n[bold]Checking Montreal Forced Aligner...[/bold]")
        
        if ComponentChecker.mfa_installed():
            self.tracker.set_mfa(True, "MFA available")
            console.print(f"  [green]MFA found[/green]")
            return
        
        console.print("  [yellow]MFA not found[/yellow]")
        
        # Confirm installation
        if RICH_AVAILABLE:
            if not Confirm.ask("Install Montreal Forced Aligner via conda-forge?"):
                self.tracker.set_mfa(False, "Installation skipped by user")
                console.print("  [yellow]MFA installation skipped[/yellow]")
                return
        else:
            response = input("Install Montreal Forced Aligner? (y/n): ").strip().lower()
            if response != 'y':
                self.tracker.set_mfa(False, "Installation skipped by user")
                print("MFA installation skipped")
                return
        
        # Get current environment name
        current_env = os.environ.get("CONDA_DEFAULT_ENV", "transcriber")
        
        console.print(f"  Installing MFA in environment '{current_env}'...")
        
        try:
            # Install MFA from conda-forge
            subprocess.run([
                'conda', 'install', '-c', 'conda-forge', 
                'montreal-forced-aligner', '-y'
            ], check=True, capture_output=True)
            
            # Verify installation
            if ComponentChecker.mfa_installed():
                self.tracker.set_mfa(True, "Installed successfully")
                console.print(f"    [green]MFA installed[/green]")
            else:
                self.tracker.set_mfa(False, "Installation completed but verification failed")
                console.print(f"    [yellow]Installation completed but 'mfa' command not found[/yellow]")
                
        except subprocess.CalledProcessError as e:
            self.tracker.set_mfa(False, f"Installation failed: {e}")
            console.print(f"    [red]MFA installation failed[/red]")
            console.print("      Try installing manually:")
            console.print(f"      [bold]conda activate {current_env}[/bold]")
            console.print("      [bold]conda install -c conda-forge montreal-forced-aligner[/bold]")

    # ========================================================================
    # STEP 8: LOCK FILE
    # ========================================================================
    
    def generate_lock_file(self):
        """Generate requirements-lock.txt with exact versions."""
        console.print("\n[bold]Generating lock file...[/bold]")
        
        try:
            result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True, check=True)
            lock_path = self.root / "requirements-lock.txt"
            
            with open(lock_path, 'w') as f:
                f.write("# Generated by 00_setup.py\n")
                f.write(f"# Date: {__import__('datetime').datetime.now()}\n")
                f.write("# Exact versions for reproducibility\n\n")
                f.write(result.stdout)
            
            console.print(f"  [green]Lock file saved to {lock_path}[/green]")
        except Exception as e:
            console.print(f"  [yellow]Failed to generate lock file: {e}[/yellow]")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Step 0: Enforce environment
    check_and_enforce_environment("transcriber")
    
    # Run setup
    setup = PipelineSetup()
    setup.run()
