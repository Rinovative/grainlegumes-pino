"""
Import-free experiment command modules.

Modules:
- cli_build_artifacts: validate or build artifacts for completed runs
- cli_device: define the shared strict runtime-device option
- cli_optuna: validate or execute additional Optuna trials
- cli_train: execute a fresh or explicit-resume training run

The initializer deliberately imports and exports no command module, keeping
package import free of parser construction and optional runtime dependencies.
"""

__all__: list[str] = []
