"""
Device Manager - Detects and manages GPU/CPU compute devices.
Supports NVIDIA CUDA, AMD ROCm/HIP (including shared memory APUs),
Apple Metal (MPS), and CPU fallback.
"""
import os
import sys
import logging
import platform

logger = logging.getLogger("DeviceManager")


class DeviceManager:
    """
    Detects available compute devices and provides the optimal
    device configuration for training and inference.
    Priority: NVIDIA CUDA/ROCm > Apple MPS > CPU
    """

    # AMD APUs with shared memory architecture (user-configurable VRAM)
    AMD_SHARED_MEMORY_GPUS = [
        'gfx1151',  # Radeon 8060S / Strix Point
        'gfx1150',  # Radeon 8050S / Strix Point
        'gfx1103',  # Radeon 780M / Phoenix
        'gfx1102',  # Radeon 760M / Phoenix
        'gfx1035',  # Radeon 680M / Rembrandt
        'gfx1034',  # Radeon 660M / Rembrandt
    ]

    # Known keywords identifying shared memory AMD APUs
    AMD_SHARED_MEMORY_KEYWORDS = [
        '8060s', '8050s', '780m', '760m', '680m', '660m',
        'radeon(tm) 8060', 'radeon(tm) 8050',
        'radeon(tm) 780', 'radeon(tm) 760',
    ]

    def __init__(self):
        self.device_type = "cpu"
        self.device_name = "CPU"
        self.gpu_memory_gb = 0.0
        self.system_memory_gb = 0.0
        self.total_physical_memory_gb = 0.0
        self.is_amd_hip = False
        self.is_shared_memory = False
        self.supported_backends = []
        self._detect_device()

    def _detect_device(self):
        """Auto-detect the best available compute device."""
        import torch

        self.supported_backends = ["cpu"]
        self._detect_system_memory()
        self.total_physical_memory_gb = self._get_actual_physical_ram()

        # Check CUDA (covers both NVIDIA and AMD ROCm/HIP)
        if torch.cuda.is_available():
            self.device_type = "cuda"
            self.device_name = torch.cuda.get_device_name(0)
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            # Detect if this is AMD HIP
            self.is_amd_hip = self._check_amd_hip()

            if self.is_amd_hip:
                self.supported_backends.append("rocm")
                self.is_shared_memory = self._check_shared_memory()

                if self.is_shared_memory:
                    # Correct for torch reporting GTT+VRAM on AMD shared memory
                    # torch may report combined GTT + VRAM, not just the driver VRAM setting
                    total_calculated = self.gpu_memory_gb + self.system_memory_gb
                    if self.total_physical_memory_gb > 0 and total_calculated > self.total_physical_memory_gb * 1.1:
                        # torch is reporting more than physical RAM allows
                        corrected_vram = self.total_physical_memory_gb - self.system_memory_gb
                        if corrected_vram > 0:
                            logger.info(
                                f"  Correcting VRAM: torch reported {self.gpu_memory_gb:.1f} GB "
                                f"(includes GTT), actual VRAM allocation: {corrected_vram:.1f} GB"
                            )
                            self.gpu_memory_gb = corrected_vram

                    total_physical = self._get_total_physical_memory()
                    logger.info(
                        f"AMD APU detected (shared memory): {self.device_name}"
                    )
                    logger.info(
                        f"  Total physical RAM: {total_physical:.1f} GB"
                    )
                    logger.info(
                        f"  VRAM allocation: {self.gpu_memory_gb:.1f} GB "
                        f"(user-configurable via AMD driver)"
                    )
                    logger.info(
                        f"  Remaining system RAM: {self.system_memory_gb:.1f} GB"
                    )
                else:
                    logger.info(
                        f"AMD discrete GPU detected (ROCm/HIP): {self.device_name} "
                        f"({self.gpu_memory_gb:.1f} GB VRAM)"
                    )
            else:
                self.supported_backends.append("cuda")
                logger.info(
                    f"NVIDIA GPU detected: {self.device_name} "
                    f"({self.gpu_memory_gb:.1f} GB VRAM)"
                )
            return

        # Check Apple Metal (MPS)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device_type = "mps"
            self.device_name = f"Apple {platform.processor()} (Metal)"
            self.is_shared_memory = True
            try:
                import psutil
                total = psutil.virtual_memory().total / (1024**3)
                self.gpu_memory_gb = total * 0.75
            except ImportError:
                self.gpu_memory_gb = 8.0
            self.supported_backends.append("mps")
            logger.info(f"Apple Metal detected: {self.device_name}")
            return

        # CPU fallback
        self.device_type = "cpu"
        self.device_name = platform.processor() or "CPU"
        self.gpu_memory_gb = 0
        logger.info(f"Using CPU: {self.device_name}")

    def _detect_system_memory(self):
        """Detect total system RAM (as seen by OS, may exclude VRAM on shared systems)."""
        try:
            import psutil
            self.system_memory_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            self.system_memory_gb = 0.0

    def _get_actual_physical_ram(self) -> float:
        """
        Get actual total physical RAM from hardware.
        On Windows, queries physical DIMM chips via PowerShell to get true
        installed memory, regardless of VRAM carve-out on shared memory systems.
        On Linux/macOS uses system tools.
        """
        try:
            if platform.system() == 'Windows':
                import subprocess
                result = subprocess.run(
                    [
                        'powershell', '-NoProfile', '-Command',
                        '(Get-CimInstance Win32_PhysicalMemory | '
                        'Measure-Object -Property Capacity -Sum).Sum / 1GB'
                    ],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    return float(result.stdout.strip())

            elif platform.system() == 'Linux':
                # Try dmidecode first (needs root), fall back to /proc/meminfo
                try:
                    import subprocess
                    result = subprocess.run(
                        ['dmidecode', '-t', 'memory'],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        total = 0
                        for line in result.stdout.split('\n'):
                            if 'Size:' in line and 'No Module' not in line:
                                parts = line.strip().split()
                                if len(parts) >= 2:
                                    try:
                                        size = int(parts[1])
                                        if 'GB' in line:
                                            total += size
                                        elif 'MB' in line:
                                            total += size / 1024
                                    except ValueError:
                                        pass
                        if total > 0:
                            return total
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass

                # Fallback: /proc/meminfo
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            kb = int(line.split()[1])
                            return kb / (1024**2)

            elif platform.system() == 'Darwin':
                import subprocess
                result = subprocess.run(
                    ['sysctl', '-n', 'hw.memsize'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    return int(result.stdout.strip()) / (1024**3)

        except Exception as e:
            logger.debug(f"Could not get physical RAM: {e}")

        return 0.0

    def _get_total_physical_memory(self) -> float:
        """
        Get total physical memory pool.
        Uses actual physical RAM if available, otherwise calculates from
        OS-visible RAM + GPU VRAM allocation.
        """
        if self.total_physical_memory_gb > 0:
            return self.total_physical_memory_gb
        if self.is_shared_memory and self.system_memory_gb > 0:
            return self.system_memory_gb + self.gpu_memory_gb
        if self.system_memory_gb > 0:
            return self.system_memory_gb
        return self.gpu_memory_gb

    def _get_available_system_ram(self) -> float:
        """Get system RAM remaining after VRAM allocation (shared memory systems)."""
        if self.is_shared_memory:
            # On shared memory systems, psutil already reports RAM minus VRAM
            # So system_memory_gb IS the remaining RAM
            return self.system_memory_gb
        return self.system_memory_gb

    def _check_amd_hip(self) -> bool:
        """Detect if PyTorch is using AMD HIP backend."""
        import torch

        # Method 1: Check torch version string
        torch_version = torch.__version__.lower()
        if 'hip' in torch_version or 'rocm' in torch_version:
            return True

        # Method 2: Check device name for AMD identifiers
        name_lower = self.device_name.lower()
        amd_keywords = ['amd', 'radeon', 'instinct', 'navi', 'vega', 'rdna', 'cdna']
        if any(kw in name_lower for kw in amd_keywords):
            return True

        # Method 3: Check for ROCm environment
        if os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH'):
            return True

        # Method 4: Check HIP runtime
        if hasattr(torch, '_C') and hasattr(torch._C, '_hip_getDeviceCount'):
            return True

        return False

    def _check_shared_memory(self) -> bool:
        """
        Detect if this AMD GPU uses shared memory (APU/unified memory).
        These GPUs carve VRAM from system RAM, configurable via AMD driver.
        Valid VRAM settings: 0.5, 1, 2, 4, 8, 16, 32, 64, 96 GB
        """
        import torch

        name_lower = self.device_name.lower()

        # Check against known shared memory GPU keywords
        for keyword in self.AMD_SHARED_MEMORY_KEYWORDS:
            if keyword in name_lower:
                return True

        # Check GFX architecture ID if available
        try:
            arch = torch.cuda.get_device_properties(0).gcnArchName
            if arch and arch.lower() in [g.lower() for g in self.AMD_SHARED_MEMORY_GPUS]:
                return True
        except (AttributeError, RuntimeError):
            pass

        # Heuristic: if VRAM is suspiciously large relative to system RAM
        shared_memory_tiers = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 96.0]
        if self.system_memory_gb > 0:
            vram_to_ram_ratio = self.gpu_memory_gb / self.system_memory_gb
            # If VRAM is > 40% of system RAM, it's likely shared
            if vram_to_ram_ratio > 0.4:
                for tier in shared_memory_tiers:
                    if abs(self.gpu_memory_gb - tier) < 2.0:
                        return True
                # Also check if total (VRAM + system RAM) matches a known physical size
                total = self.gpu_memory_gb + self.system_memory_gb
                known_physical = [32, 64, 96, 128, 192, 256]
                for phys in known_physical:
                    if abs(total - phys) < 5.0:
                        return True

        return False

    def get_torch_device(self):
        """Get the torch device string."""
        import torch
        if self.device_type == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif self.device_type == "mps":
            return torch.device("mps")
        return torch.device("cpu")

    def get_max_batch_size(self, model_size_gb: float = 0.5) -> int:
        """
        Estimate max batch size based on available VRAM.
        For shared memory systems, be conservative to leave RAM for the OS.
        """
        available_vram = self.gpu_memory_gb

        if self.is_shared_memory:
            # Reserve memory for OS + other processes
            # Don't use more than 80% of allocated VRAM on shared systems
            available_vram *= 0.8
            remaining_system_ram = self._get_available_system_ram()
            if remaining_system_ram < 8.0:
                logger.warning(
                    f"Low system RAM ({remaining_system_ram:.1f} GB remaining). "
                    f"Consider reducing VRAM allocation in AMD driver."
                )
                available_vram *= 0.6

        # Rough estimate: (available VRAM - model size) / ~0.5GB per batch item
        usable = max(0, available_vram - model_size_gb)
        batch_size = max(1, int(usable / 0.5))

        return min(batch_size, 64)  # cap at 64

    def get_training_args(self) -> dict:
        """Get device-specific training arguments."""
        import torch
        args = {}

        if self.device_type == "cuda":
            if self.is_amd_hip:
                args['fp16'] = True
                args['bf16'] = False
                if self.is_shared_memory:
                    args['gradient_checkpointing'] = True
            else:
                args['fp16'] = True
                args['bf16'] = False
                if torch.cuda.is_available():
                    cap = torch.cuda.get_device_capability()
                    if cap[0] >= 8:
                        args['fp16'] = False
                        args['bf16'] = True

        elif self.device_type == "mps":
            args['fp16'] = False
            args['bf16'] = False
            args['use_mps_device'] = True

        else:
            args['fp16'] = False
            args['bf16'] = False
            args['no_cuda'] = True

        return args

    def get_device_info(self) -> dict:
        """Return device information dictionary."""
        info = {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "gpu_memory_gb": round(self.gpu_memory_gb, 2),
            "system_memory_gb": round(self.system_memory_gb, 2),
            "is_amd_hip": self.is_amd_hip,
            "is_shared_memory": self.is_shared_memory,
            "supported_backends": self.supported_backends,
        }

        if self.is_shared_memory:
            total_physical = self._get_total_physical_memory()
            info["total_physical_memory_gb"] = round(total_physical, 2)
            info["shared_memory_note"] = (
                f"VRAM is allocated from system RAM. "
                f"Total physical: {total_physical:.0f} GB, "
                f"VRAM: {self.gpu_memory_gb:.1f} GB, "
                f"System RAM: {self.system_memory_gb:.1f} GB. "
                f"Adjustable via AMD Adrenalin driver settings."
            )
            info["vram_tiers_gb"] = [0.5, 1, 2, 4, 8, 16, 32, 64, 96]

        return info