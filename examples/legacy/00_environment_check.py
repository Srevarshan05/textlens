"""Check the TextLens runtime before downloading or loading a model."""

from textlens import check_dependencies, get_hardware_info, print_hardware_status


def main() -> None:
    print_hardware_status()
    report = check_dependencies()
    print(f"Dependencies satisfied: {report.all_satisfied}")
    if report.missing:
        print("Missing modules:", ", ".join(report.missing))
        print("Install with:", report.install_command)

    hardware = get_hardware_info()
    print(f"CUDA available: {hardware.gpu_available}")
    print(f"Suggested PyTorch command: {hardware.recommended_torch_cmd}")


if __name__ == "__main__":
    main()
