#!/usr/bin/env python3
"""
Test the discovery chain functionality
"""

from dns_servers import DNSDiscovery
import json

def main():
    print("Testing DNS Discovery Chain Functionality")
    print("=" * 50)
    
    # Create a discovery instance with shorter timeout for testing
    discovery = DNSDiscovery(timeout=2, threads=10)
    
    print("Step 1: Validating known servers...")
    discovery.validate_known_servers()
    
    print(f"\nStep 2: Discovering from NS records...")
    # Test with a smaller set of domains
    test_domains = ['google.com', 'github.com']
    discovery.discover_from_ns_records(test_domains, '8.8.8.8')
    
    print(f"\nStep 3: Show discovery relationships...")
    discovery.show_discovery_chain()
    
    # Show statistics
    stats = discovery.get_discovery_statistics()
    print(f"\nDiscovery Statistics:")
    for method, data in stats.items():
        method_name = method.replace('_', ' ').title()
        print(f"  {method_name}: {data['count']} servers from {len(data['sources'])} sources")
    
    # Save detailed results
    results = {
        'total_servers': len(discovery.discovered_servers),
        'servers': sorted(list(discovery.discovered_servers)),
        'discovery_chain': {server: {'method': method, 'source': source, 'info': info} 
                          for server, (method, source, info) in discovery.discovery_chain.items()}
    }
    
    with open('test_discovery_chain.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to test_discovery_chain.json")

if __name__ == "__main__":
    main()