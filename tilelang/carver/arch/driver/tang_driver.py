from __future__ import annotations
import ctypes


class tangDeviceProp(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("name", ctypes.c_char * 256),  # Device name.
        ("uuid", ctypes.c_byte * 16),  # a 16-byte unique identifier
        ("totalGlobalMem", ctypes.c_uint64),  # size of global memory region (in bytes).
        ("sharedMemPerBlock", ctypes.c_int),  # the maximum amount of shared memory available to a thread block in bytes.
        ("regsPerBlock", ctypes.c_int),  # the maximum number of 32-bit registers available to a thread block.
        ("warpSize", ctypes.c_int),  # the warp size in threads.
        ("memPitch", ctypes.c_int),  # the maximum pitch in bytes allowed by the memory copy functions
        ("maxThreadsPerBlock", ctypes.c_int),  # the maximum number of threads per block.
        ("maxThreadsDim", ctypes.c_int * 3),  # Max number of threads in each dimension (XYZ) of a block.
        ("maxGridSize", ctypes.c_int * 3),  # Max grid dimensions (XYZ).
        ("clockRate", ctypes.c_int),  # Max clock frequency of the multiProcessors in khz.
        ("totalConstMem", ctypes.c_int),  # the total amount of constant memory available on the device in bytes.
        ("multiProcessorCount", ctypes.c_int),  # Number of multi-processors (compute units).
        ("maxBlocksPerMultiProcessor", ctypes.c_int),  # the number of multiprocessors on the device
        ("asyncEngineCount", ctypes.c_int),
        ("memoryClockRate", ctypes.c_int),  # Max global memory clock frequency in khz
        ("memoryBusWidth", ctypes.c_int),  # Global memory bus width in bits.
        ("l2CacheSize", ctypes.c_int),  # L2 cache size.
        ("maxThreadsPerMultiProcessor", ctypes.c_int),  # Maximum resident threads per multi-processor.
        ("globalL1CacheSupported", ctypes.c_int),  # whether the device supports caching of globals in L1 cache
        ("localL1CacheSupported", ctypes.c_int),  # whether the device supports caching of locals in L1 cache
        ("sharedMemPerMultiprocessor", ctypes.c_int),  # Maximum Shared Memory Per Multiprocessor.
        ("regsPerMultiprocessor", ctypes.c_int),  # the maximum amount of shared memory available to a multiprocessor in bytes
        ("streamPrioritiesSupported", ctypes.c_int),  # whether the device supports stream priorities
        ("concurrentKernels", ctypes.c_int),  # Device can possibly execute multiple kernels concurrently.
        ("computePreemptionSupported", ctypes.c_int),  # whether the device supports Compute Preemption
        ("kernelExecTimeoutEnabled", ctypes.c_int),  # Run time limit for kernels executed on the device
        ("ECCEnabled", ctypes.c_int),  # Device has ECC support enabled
        ("accessPolicyMaxWindowSize", ctypes.c_int),  # the maximum value of tangAccessPolicyWindow::num_bytes
        ("tccDriver", ctypes.c_int),  # whether device is a Tesla device using TCC driver
        ("singleToDoublePrecisionPerfRatio", ctypes.c_int),  # the ratio of single precision  performance (in floating-point
        # operations per second) to double
        # precision performance
        ("cooperativeLaunch", ctypes.c_int),  # whether the device supports launching cooperative kernels via tangLaunchCooperativeKernel
        ("cooperativeMultiDeviceLaunch", ctypes.c_int),  # whether the device supports launching
        # cooperative kernels via
        # tangLaunchCooperativeKernelMultiDevice
        ("persistingL2CacheMaxSize", ctypes.c_int),  # L2 cache's maximum persisting lines size in bytes
        ("canMapHostMemory", ctypes.c_int),  # Check whether TANG can map host memory
        ("unifiedAddressing", ctypes.c_int),  # whether the device shares a unified address space
        # with the host and 0 otherwise
        ("managedMemory", ctypes.c_int),  # whether the device supports allocating managed memory
        # on this system, or 0 if it is not supported
        ("concurrentManagedAccess", ctypes.c_int),  # whether the device can coherently access
        # managed memory concurrently with the CPU
        ("directManagedMemAccessFromHost", ctypes.c_int),  # whether the host can directly access
        # managed memory on the device without
        # migration
        ("pageableMemoryAccess", ctypes.c_int),  # whether the device supports coherently
        # accessing pageable memory without calling
        # tangHostRegister on it
        ("pageableMemoryAccessUsesHostPageTables", ctypes.c_int),  # whether the device accesses
        # pageable memory via the
        # host's page tables
        ("canUseHostPointerForRegisteredMem", ctypes.c_int),  # whether the device can access
        # host registered memory at the
        # same virtual address as the CPU
        ("hostNativeAtomicSupported", ctypes.c_int),  # Link between the device and the host
        # supports native atomic operations
        ("canFlushRemoteWrites", ctypes.c_int),  # Device supports flushing of outstanding remote writes
        ("gpuOverlap", ctypes.c_int),  # Device can possibly copy memory and execute a kernel concurrently
        ("integrated", ctypes.c_int),  # Device is integrated with host memory
        ("maxSharedMemoryPerBlockOptin", ctypes.c_int),  # The maximum optin shared memory per
        # block. This value may vary by chip.
        # See ::tangFuncSetAttribute
        ("gpuDirectRDMASupported", ctypes.c_int),  # Device supports GPUDirect RDMA APIs
        ("gpuDirectRDMAFlushWritesOptions", ctypes.c_int),  # The returned attribute shall be
        # interpreted as a bitmask, where the
        # individual bits are listed in the
        # ::tangFlushGPUDirectRDMAWritesOptions
        # enum
        ("gpuDirectRDMAWritesOrdering", ctypes.c_int),  # GPUDirect RDMA writes to the device do
        # not need to be flushed for consumers
        # within the scope indicated by the
        # returned attribute. See
        # ::tangGPUDirectRDMAWritesOrdering for
        # the numerical values returned here.
        ("major", ctypes.c_int),  # the major revision numbers defining the device's compute capability
        ("minor", ctypes.c_int),  # the minor revision numbers defining the device's compute capability
        ("pciBusID", ctypes.c_int),  # PCI Bus ID.
        ("pciDeviceID", ctypes.c_int),  # PCI Device ID.
        ("pciDomainID", ctypes.c_int),  # PCI Domain ID
        ("isMultiGpuBoard", ctypes.c_int),  # whether device is on a multi-GPU board.
        ("multiGpuBoardGroupID", ctypes.c_int),  # a unique identifier for a group of devices associated with the same board
        ("computeMode", ctypes.c_int),  # the compute mode that the device is currently in
        ("reservedSharedMemoryPerBlock", ctypes.c_int),  # Shared memory reserved by TANG driver per block in bytes
        ("sparseTangArraySupported", ctypes.c_int),  # Device supports sparse arrays and sparse  mipmapped arrays
        ("hostRegisterSupported", ctypes.c_int),  # Device supports host memory registration via ::tangHostRegister
        ("hostRegisterReadOnlySupported", ctypes.c_int),  # Device supports using the
        # ::tangHostRegister flag
        # tangHostRegisterReadOnly to register
        # memory that must be mapped as
        # read-only to the GPU
        ("memoryPoolsSupported", ctypes.c_int),  # Device supports using the ::tangMallocAsync and ::tangMemPool family of APIs
        ("memoryPoolSupportedHandleTypes", ctypes.c_int),  # Handle types supported with mempool based IPC
    ]


def get_tang_device_properties(device_id: int = 0) -> tangDeviceProp | None:
    libtangrt = ctypes.cdll.LoadLibrary("libtangrt_shared.so")

    prop = tangDeviceProp()
    tangGetDeviceProperties = libtangrt.tangGetDeviceProperties
    tangGetDeviceProperties.argtypes = [ctypes.POINTER(tangDeviceProp), ctypes.c_int]
    tangGetDeviceProperties.restype = ctypes.c_int
    ret = tangGetDeviceProperties(ctypes.byref(prop), device_id)
    if ret == 0:
        return prop
    else:
        raise RuntimeError(f"tangGetDeviceProperties failed with error {ret}")


def get_device_name(device_id: int = 0) -> str | None:
    prop = get_tang_device_properties(device_id)
    if prop:
        return prop.name.decode()
    else:
        raise RuntimeError("Failed to get device properties.")


def get_shared_memory_per_block(device_id: int = 0, format: str = "bytes") -> int | None:
    assert format in ["bytes", "kb", "mb"], "Invalid format. Must be one of: bytes, kb, mb"
    prop = get_tang_device_properties(device_id)
    if prop:
        # Convert size_t to int to avoid overflow issues
        shared_mem = int(prop.sharedMemPerBlock)
        if format == "bytes":
            return shared_mem
        elif format == "kb":
            return shared_mem // 1024
        elif format == "mb":
            return shared_mem // (1024 * 1024)
        else:
            raise RuntimeError("Invalid format. Must be one of: bytes, kb, mb")
    else:
        raise RuntimeError("Failed to get device properties.")


def get_device_attribute(attr: int, device_id: int = 0) -> int:
    try:
        libtangrt = ctypes.cdll.LoadLibrary("libtangrt_shared.so")

        value = ctypes.c_int()
        tangDeviceGetAttribute = libtangrt.tangDeviceGetAttribute
        tangDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        tangDeviceGetAttribute.restype = ctypes.c_int

        ret = tangDeviceGetAttribute(ctypes.byref(value), attr, device_id)
        if ret != 0:
            raise RuntimeError(f"tangDeviceGetAttribute failed with error {ret}")

        return value.value
    except Exception as e:
        print(f"Error getting device attribute: {str(e)}")
        return None


def get_max_dynamic_shared_size_bytes(device_id: int = 0, format: str = "bytes") -> int | None:
    """
    Get the maximum dynamic shared memory size in bytes, kilobytes, or megabytes.
    """
    assert format in ["bytes", "kb", "mb"], "Invalid format. Must be one of: bytes, kb, mb"
    prop = get_tang_device_properties(device_id)
    if prop:
        # Convert size_t to int to avoid overflow issues
        shared_mem = int(prop.sharedMemPerMultiprocessor)
        if format == "bytes":
            return shared_mem
        elif format == "kb":
            return shared_mem // 1024
        elif format == "mb":
            return shared_mem // (1024 * 1024)
        else:
            raise RuntimeError("Invalid format. Must be one of: bytes, kb, mb")
    else:
        raise RuntimeError("Failed to get device properties.")


def get_persisting_l2_cache_max_size(device_id: int = 0) -> int:
    prop = get_tang_device_properties(device_id)
    if prop:
        return prop.persistingL2CacheMaxSize
    else:
        raise RuntimeError("Failed to get device properties for persisting L2 cache max size.")


def get_num_sms(device_id: int = 0) -> int:
    """
    Get the number of streaming multiprocessors (SMs) on the TANG device.

    Args:
        device_id (int, optional): The TANG device ID. Defaults to 0.

    Returns:
        int: The number of SMs on the device.

    Raises:
        RuntimeError: If unable to get the device properties.
    """
    prop = get_tang_device_properties(device_id)
    if prop is None:
        raise RuntimeError("Failed to get device properties.")
    return prop.multiProcessorCount


def get_registers_per_block(device_id: int = 0) -> int:
    """
    Get the maximum number of 32-bit registers available per block.
    """
    prop = get_tang_device_properties(device_id)
    if prop:
        return prop.regsPerBlock
    else:
        raise RuntimeError("Failed to get device properties.")
