#!/usr/bin/env python3
"""
Visualize DNS Server Discovery Relationships
"""

import json

def visualize_discovery_chain(json_file):
    """Read and visualize the DNS discovery chain from JSON file"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    discovery_chain = data.get('discovery_chain', {})
    
    print("🔍 DNS SERVER DISCOVERY RELATIONSHIPS")
    print("=" * 60)
    
    # Group by discovery method and source
    relationships = {}
    
    for server_ip, details in discovery_chain.items():
        method = details['method']
        source = details['source']
        info = details['info']
        
        if method not in relationships:
            relationships[method] = {}
        
        if source not in relationships[method]:
            relationships[method][source] = []
        
        relationships[method][source].append((server_ip, info))
    
    # Display the relationships
    for method, sources in relationships.items():
        method_title = method.replace('_', ' ').title()
        print(f"\n📋 {method_title}:")
        print("-" * (len(method_title) + 3))
        
        for source, servers in sources.items():
            if method == 'known_server':
                print(f"  📚 Built-in DNS server list: {len(servers)} servers")
                for server_ip, info in sorted(servers):
                    print(f"     • {server_ip}")
            
            elif method == 'ns_record':
                print(f"  🔗 Queried via {source}:")
                # Group by domain
                by_domain = {}
                for server_ip, info in servers:
                    domain = info.split(' for ')[-1] if ' for ' in info else 'unknown'
                    if domain not in by_domain:
                        by_domain[domain] = []
                    by_domain[domain].append((server_ip, info.split(' for ')[0]))
                
                for domain, domain_servers in by_domain.items():
                    print(f"       📍 {domain}:")
                    for server_ip, ns_name in sorted(domain_servers):
                        print(f"         → {server_ip} ({ns_name})")
            
            elif method == 'range_scan':
                print(f"  🔍 Scanned IP range {source}:")
                for server_ip, info in sorted(servers):
                    print(f"     → {server_ip}")
    
    # Summary statistics
    print(f"\n📊 DISCOVERY SUMMARY:")
    print(f"   Total DNS servers found: {data['total_servers']}")
    
    method_counts = {}
    for details in discovery_chain.values():
        method = details['method']
        method_counts[method] = method_counts.get(method, 0) + 1
    
    for method, count in method_counts.items():
        method_name = method.replace('_', ' ').title()
        print(f"   {method_name}: {count} servers")
    
    print("\n💡 RELATIONSHIP EXPLANATION:")
    print("   • Known servers are pre-configured public DNS servers")
    print("   • NS Record servers were discovered by querying other DNS servers")
    print("   • Range scan servers were found by scanning IP ranges")
    print("   • The 'source' shows which DNS server was used to discover others")

def main():
    try:
        visualize_discovery_chain('test_discovery_chain.json')
    except FileNotFoundError:
        print("❌ test_discovery_chain.json not found. Run test_discovery_chain.py first.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()