def determine_batch_size(device: str) -> int:
    """
    Determines the batch size dynamically based on the available device and resources.

    Args:
        device (str): The computation device ("cuda" or "cpu").

    Returns:
        int: Recommended batch size for the device.
    """
    if device == "cuda":
        # Check GPU properties
        import torch
        gpu_properties = torch.cuda.get_device_properties(0)
        vram_gb = gpu_properties.total_memory / \
            (1024 ** 3)  # Convert VRAM to GB

        # Set batch size based on VRAM
        if vram_gb >= 16:  # High-end GPU
            return 256
        elif vram_gb >= 8:  # Mid-range GPU
            return 128
        else:  # Low-end GPU
            return 64
    else:
        # CPU-based batch size (depends on core count)
        core_count = os.cpu_count()
        if core_count >= 16:
            return 32
        elif core_count >= 8:
            return 16
        else:
            return 8
