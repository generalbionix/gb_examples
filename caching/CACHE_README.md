# Caching for GeneralBionix API Calls

This project includes intelligent caching using [joblib.Memory](https://joblib.readthedocs.io/en/latest/memory.html) to avoid repeated expensive API calls during development and testing.

## Benefits

- **Faster Development**: Identical API calls return cached results instantly
- **Cost Savings**: Avoid redundant API calls during testing 
- **Robust**: Uses industry-standard joblib caching designed for scientific computing
- **Automatic**: No manual cache key management needed
- **Persistent**: Cache survives between runs

## How It Works

The `CachedGeneralBionixClient` wraps the original `GeneralBionixClient` and automatically caches:

- **Point Cloud Cropping**: Same point cloud + click coordinates → cached cropped result
- **Grasp Prediction**: Same cropped point cloud → cached grasp predictions  
- **Grasp Filtering**: Same grasp list → cached filtering results

Cache keys are automatically generated from input data using joblib's robust hashing.

## File Structure

```
caching/
├── __init__.py          # Package initialization (exports CachedGeneralBionixClient)
├── cache_client.py      # Main cached client implementation
├── cache_utils.py       # Internal cache utilities (use manage_cache.py instead)
└── CACHE_README.md      # This documentation

manage_cache.py          # Command-line cache management script (in root directory)
```

## Usage

### Basic Usage

```python
from caching import CachedGeneralBionixClient

# Replace GeneralBionixClient with CachedGeneralBionixClient
client = CachedGeneralBionixClient(
    api_key="your-api-key",
    enable_cache=True,  # Set to False to disable caching
    verbose=1          # 0=silent, 1=normal, 2=verbose
)

# Use exactly like the original client
result = client.crop_point_cloud(pcd, x, y)  # First call hits API
result = client.crop_point_cloud(pcd, x, y)  # Second call uses cache
```

### Configuration Options

```python
client = CachedGeneralBionixClient(
    api_key="your-api-key",
    cache_dir="./cache",     # Directory for cache files  
    enable_cache=True,       # Enable/disable caching
    verbose=1               # Verbosity: 0=silent, 1=normal, 2=verbose
)
```

### Cache Management

```python
# View cache statistics
stats = client.cache_stats()
print(f"Cache has {stats['total_files']} files, {stats['total_size_mb']} MB")

# Clear all cache
client.clear_cache()

# Clear cache for specific method
client.clear_cache("crop")     # Clear point cloud cropping cache
client.clear_cache("predict")  # Clear grasp prediction cache  
client.clear_cache("filter")   # Clear grasp filtering cache
```

### Command Line Cache Management

Use the provided cache management script in the root directory:

```bash
# Show cache statistics
python manage_cache.py stats

# Clear all cache
python manage_cache.py clear

# Clear specific method cache
python manage_cache.py clear --method crop
python manage_cache.py clear --method predict
python manage_cache.py clear --method filter

# Use custom cache directory
python manage_cache.py --cache-dir /path/to/cache stats
```

## Cache Behavior

- **Automatic Detection**: Identical inputs automatically return cached results
- **Persistent**: Cache survives between program runs
- **Smart Hashing**: Uses joblib's sophisticated hashing for complex data structures
- **Safe**: Cache misses gracefully fall back to API calls
- **Space Efficient**: Binary serialization for efficient storage

## Examples

The main example files (`grasp_example.py` and `agent_example.py`) have been updated to use caching by default. You can see the caching in action:

1. **First Run**: All API calls hit the services (you'll see "📡 Calling..." messages)
2. **Second Run**: Identical operations use cached results (much faster)
3. **Cache Stats**: See cache statistics displayed at startup

## When Cache Hits/Misses

**Cache HIT** (fast):
- Same point cloud data + same click coordinates
- Same cropped point cloud for grasp prediction
- Same list of grasps for filtering

**Cache MISS** (API call):
- Different point cloud (different scene/objects)
- Different click coordinates
- Different grasp poses

## Troubleshooting

### Cache Not Working?

1. Check that `enable_cache=True` in the client initialization
2. Verify cache directory permissions 
3. Use `verbose=2` to see detailed cache behavior
4. Check cache stats with `client.cache_stats()`

### Clear Cache If Needed

```python
# If you want fresh API calls (e.g., testing API changes)
client.clear_cache()
```

### Disable Caching

```python
# For debugging or when you always want fresh API calls
client = CachedGeneralBionixClient(api_key="key", enable_cache=False)
```

## Integration

The cached client is a drop-in replacement for `GeneralBionixClient`:

```python
# Before
from client import GeneralBionixClient
client = GeneralBionixClient(api_key)

# After  
from caching import CachedGeneralBionixClient
client = CachedGeneralBionixClient(api_key)
```

All existing code continues to work unchanged while gaining automatic caching benefits. 