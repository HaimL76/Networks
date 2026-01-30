#!/usr/bin/env python3
"""
Simple test of the DNS discovery functionality
"""

from dns_servers import DNSDiscovery
import time

def main():
    print("Simple DNS Discovery Test")
    print("=" * 40)
    
    # Create a discovery instance with shorter timeout for testing
    discovery = DNSDiscovery(timeout=2, threads=10)
    
    print("Testing basic functionality...")
    
    # Test known servers only
    discovery.validate_known_servers()
    
    print(f"\nFound {len(discovery.discovered_servers)} DNS servers:")
    sorted_servers = sorted(discovery.discovered_servers)
    for i, server in enumerate(sorted_servers, 1):
        print(f"  {i}. {server}")
    
    # Test performance of first 5 servers
    if len(discovery.discovered_servers) > 0:
        print(f"\nTesting performance of first 5 servers...")
        servers_to_test = list(discovery.discovered_servers)[:5]
        discovery.test_server_performance(servers_to_test)

if __name__ == "__main__":
    main()