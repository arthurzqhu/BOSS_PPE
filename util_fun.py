import psutil
import os
import gc
import sys

def detailed_memory_analysis():
    """Analyze memory usage across different spaces"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    print("=== Memory Analysis ===")
    print(f"RSS (Resident Set Size): {mem_info.rss / 1024**3:.2f} GB")
    print(f"VMS (Virtual Memory Size): {mem_info.vms / 1024**3:.2f} GB")
    print(f"Shared Memory: {mem_info.shared / 1024**3:.2f} GB")
    print(f"Text Segment: {mem_info.text / 1024**3:.2f} GB")
    print(f"Data Segment: {mem_info.data / 1024**3:.2f} GB")
    
    # # Check if there's a large gap between RSS and VMS
    # if mem_info.vms > mem_info.rss * 1.5:
    #     print(f"\n⚠️  Large VMS-RSS gap: {(mem_info.vms - mem_info.rss) / 1024**3:.2f} GB")
    #     print("This might indicate memory fragmentation or external libraries")

def get_top_memory_vars(local_vars, top_n=10, l_return=True):
    var_mem = []
    for k, v in local_vars.items():
        try:
            size = sys.getsizeof(v)
        except Exception:
            size = 0
        var_mem.append((k, size, type(v)))
    var_mem.sort(key=lambda x: x[1], reverse=True)
    print(f"{'Variable':<30} {'Size (MB)':>12} {'Type':>30}")
    print("-" * 75)
    for name, size, typ in var_mem[:top_n]:
        print(f"{name:<30} {size/1024/1024:12.2f} {str(typ):>30}")
    if l_return:
        return var_mem

def free_iterated_vars(local_vars, verbose=True):
    """
    Identify and delete variables that are likely to be iterated/temporary variables,
    such as those with names like 'i', 'j', 'k', 'tmp', or starting with '_'.
    """
    iter_var_names = []
    for k in list(local_vars.keys()):
        if (
            (len(k) == 1 and k in "ijklmnpq") or
            # k.startswith('_') or
            k.startswith('tmp') or
            k.startswith('temp') or
            k.endswith('_tmp') or
            k.endswith('_temp')
        ):
            iter_var_names.append(k)
    for k in iter_var_names:
        if verbose:
            print(f"Deleting variable: {k}")
        try:
            del local_vars[k]
        except Exception:
            pass
    gc.collect()
    if verbose:
        print(f"Deleted {len(iter_var_names)} likely iterated/temporary variables.")

def cleanup_large_variables(local_vars, var_names_to_keep=None):
    """
    Clean up large variables from the local namespace to free memory.
    
    Args:
        local_vars: locals() dictionary
        var_names_to_keep: list of variable names to keep (e.g., ['nc_dict', 'ncvars'])
    """
    if var_names_to_keep is None:
        var_names_to_keep = []
    
    # Common large variable names to clean up
    large_var_patterns = [
        'nc_dict', 'ncvars', 'data_range', 'all_combos', 'chunk_',
        'temp_', 'tmp_', '_tmp', '_temp', 'large_', 'big_'
    ]
    
    cleaned_vars = []
    for var_name in list(local_vars.keys()):
        # Skip variables we want to keep
        if var_name in var_names_to_keep:
            continue
            
        # Check if variable name matches patterns for large variables
        should_clean = any(pattern in var_name for pattern in large_var_patterns)
        
        # Also clean variables that are likely to be large based on type
        var_value = local_vars[var_name]
        if hasattr(var_value, '__len__') and len(var_value) > 1000:
            should_clean = True
        
        if should_clean:
            try:
                del local_vars[var_name]
                cleaned_vars.append(var_name)
            except (KeyError, TypeError):
                pass
    
    # Force garbage collection
    gc.collect()
    
    print(f"Cleaned up {len(cleaned_vars)} variables: {cleaned_vars}")
    return cleaned_vars

def monitor_memory_usage(func):
    """
    Decorator to monitor memory usage before and after function execution.
    """
    def wrapper(*args, **kwargs):
        print(f"Memory before {func.__name__}:")
        detailed_memory_analysis()
        
        result = func(*args, **kwargs)
        
        print(f"Memory after {func.__name__}:")
        detailed_memory_analysis()
        
        return result
    return wrapper

def optimize_numpy_arrays():
    """
    Optimize NumPy arrays to use less memory.
    """
    import numpy as np
    
    # Set NumPy to use less memory
    np.set_printoptions(precision=6, suppress=True)
    
    # Use float32 instead of float64 where possible
    # This can be applied to arrays that don't need double precision
    
    print("NumPy memory optimization applied")

def get_memory_usage_by_type():
    """
    Get memory usage broken down by object type.
    """
    import sys
    import gc
    
    type_sizes = {}
    type_counts = {}
    
    for obj in gc.get_objects():
        obj_type = type(obj).__name__
        try:
            size = sys.getsizeof(obj)
            if obj_type not in type_sizes:
                type_sizes[obj_type] = 0
                type_counts[obj_type] = 0
            type_sizes[obj_type] += size
            type_counts[obj_type] += 1
        except:
            pass
    
    # Sort by size
    sorted_types = sorted(type_sizes.items(), key=lambda x: x[1], reverse=True)
    
    print("Memory usage by type:")
    print(f"{'Type':<20} {'Count':<10} {'Size (MB)':<12}")
    print("-" * 45)
    
    for obj_type, size in sorted_types[:20]:
        count = type_counts[obj_type]
        size_mb = size / 1024 / 1024
        print(f"{obj_type:<20} {count:<10} {size_mb:<12.2f}")
    
    return type_sizes, type_counts
