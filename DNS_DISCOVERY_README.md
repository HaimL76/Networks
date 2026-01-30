# DNS Server Discovery Tool

A comprehensive Python program to discover DNS servers on the internet, starting from Google's 8.8.8.8 DNS server.

## Features

- **Multiple Discovery Methods**: 
  - Validates known public DNS servers
  - Discovers servers from NS records of popular domains
  - Scans IP ranges where DNS servers are commonly found
  - Tests both UDP and TCP DNS functionality

- **Performance Testing**: Measures response times of discovered servers

- **Concurrent Processing**: Uses multi-threading for efficient scanning

- **Export Results**: Save discovered servers to JSON format

## Installation

Install the required dependency:
```bash
pip install dnspython
```

## Usage

### Basic Usage
```bash
python dns_servers.py
```

### Command Line Options
- `--timeout TIMEOUT`: Connection timeout in seconds (default: 3)
- `--threads THREADS`: Number of concurrent threads (default: 50)
- `--no-ranges`: Skip IP range scanning
- `--no-ns`: Skip NS record discovery  
- `--save FILENAME`: Save results to specified file
- `--performance`: Test performance of discovered servers

### Examples

1. **Quick discovery** (skip range scanning):
   ```bash
   python dns_servers.py --no-ranges
   ```

2. **Full discovery with performance testing**:
   ```bash
   python dns_servers.py --performance --save results.json
   ```

3. **Conservative scan** (fewer threads, longer timeout):
   ```bash
   python dns_servers.py --threads 20 --timeout 5
   ```

## Discovery Methods Explained

### 1. Known Public DNS Servers
The tool starts by validating a list of well-known public DNS servers:
- Google DNS (8.8.8.8, 8.8.4.4)
- Cloudflare (1.1.1.1, 1.0.0.1)
- OpenDNS (208.67.222.222, 208.67.220.220)
- Quad9 (9.9.9.9, 149.112.112.112)
- And more...

### 2. NS Record Discovery
Queries NS records for popular domains to find authoritative name servers:
- google.com, facebook.com, amazon.com
- wikipedia.org, github.com, microsoft.com
- And other high-traffic domains

### 3. IP Range Scanning
Scans common IP ranges where DNS servers are typically located:
- Google DNS ranges (8.8.8.0/24)
- Cloudflare ranges (1.1.1.0/24)
- Major ISP DNS server ranges
- Public DNS provider ranges

## How It Works

1. **Port Detection**: Checks if port 53 (DNS) is open
2. **DNS Validation**: Sends actual DNS queries to verify functionality
3. **Dual Protocol Testing**: Tests both UDP and TCP DNS queries
4. **Response Validation**: Ensures servers return valid DNS responses

## Sample Output

```
============================================================
DNS Server Discovery Tool
============================================================
Starting discovery from Google DNS (8.8.8.8)
Timeout: 3s, Threads: 50
============================================================
Validating known public DNS servers...
✓ Validated known server: 8.8.8.8
✓ Validated known server: 1.1.1.1
...

Discovering DNS servers from NS records...
✓ Found DNS server from NS record: 216.239.32.10 (ns1.google.com)
...

Scanning IP range: 8.8.8.0/24
✓ Found DNS server: 8.8.8.8
...

============================================================
DISCOVERY COMPLETE
============================================================
Time taken: 45.23 seconds
Total DNS servers discovered: 127

Discovered DNS servers:
  1. 1.0.0.1
  2. 1.1.1.1
  3. 8.8.4.4
  4. 8.8.8.8
  ...
```

## Performance Testing Output

When using `--performance` flag:

```
Testing performance of 10 servers...
  8.8.8.8: 25.34ms
  1.1.1.1: 18.76ms
  208.67.222.222: 45.12ms
  ...

Fastest DNS servers:
  1. 1.1.1.1 - 18.76ms
  2. 8.8.8.8 - 25.34ms
  3. 8.8.4.4 - 28.91ms
  ...
```

## Important Notes

- **Ethical Usage**: This tool should be used responsibly and in compliance with network policies
- **Network Impact**: Use reasonable thread counts to avoid overwhelming networks
- **Firewall Considerations**: Some networks may block DNS queries or port scanning
- **Rate Limiting**: Some DNS servers may rate limit or block excessive queries

## Technical Details

- **DNS Query Type**: Performs A record queries for 'google.com' to test servers
- **Protocols**: Tests both UDP (standard) and TCP (for larger responses)
- **Threading**: Uses ThreadPoolExecutor for concurrent operations
- **Timeout Handling**: Configurable timeouts prevent hanging on unresponsive servers
- **Error Handling**: Graceful handling of network errors and invalid responses

## Customization

You can modify the `DNSDiscovery` class to:
- Add more known DNS servers to the initial list
- Include additional domains for NS record queries
- Modify IP ranges to scan
- Change the DNS query types or test domains
- Adjust threading and timeout parameters

This tool provides a comprehensive approach to DNS server discovery and can be extended for more specialized network reconnaissance tasks.