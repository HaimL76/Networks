#!/usr/bin/env python3
"""
DNS Server Discovery Tool
Discovers DNS servers on the internet starting from Google's 8.8.8.8
"""

import socket
import threading
import time
import ipaddress
import dns.resolver
import dns.query
import dns.message
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
from datetime import datetime

class DNSDiscovery:
    def __init__(self, timeout=3, threads=50):
        self.timeout = timeout
        self.threads = threads
        self.discovered_servers = set()
        self.discovery_chain = {}  # Maps server -> (source_type, source_server, additional_info)
        self.lock = threading.Lock()
        
        # Known public DNS servers to start with
        self.known_servers = [
            '8.8.8.8',          # Google Primary
            '8.8.4.4',          # Google Secondary
            '1.1.1.1',          # Cloudflare Primary
            '1.0.0.1',          # Cloudflare Secondary
            '208.67.222.222',   # OpenDNS Primary
            '208.67.220.220',   # OpenDNS Secondary
            '9.9.9.9',          # Quad9 Primary
            '149.112.112.112',  # Quad9 Secondary
            '64.6.64.6',        # Verisign Primary
            '64.6.65.6',        # Verisign Secondary
        ]

    def test_dns_server(self, ip, port=53):
        """Test if an IP address is a functioning DNS server"""
        try:
            # Test UDP DNS query
            if self.test_udp_dns(ip, port):
                return True
            # Test TCP DNS query
            if self.test_tcp_dns(ip, port):
                return True
        except Exception:
            pass
        return False

    def test_udp_dns(self, ip, port=53):
        """Test DNS server using UDP"""
        try:
            # Create a DNS query for google.com A record
            query = dns.message.make_query('google.com', dns.rdatatype.A)
            response = dns.query.udp(query, ip, timeout=self.timeout, port=port)
            
            # Check if we got a valid response
            if response and len(response.answer) > 0:
                return True
        except Exception:
            pass
        return False

    def test_tcp_dns(self, ip, port=53):
        """Test DNS server using TCP"""
        try:
            # Create a DNS query for google.com A record
            query = dns.message.make_query('google.com', dns.rdatatype.A)
            response = dns.query.tcp(query, ip, timeout=self.timeout, port=port)
            
            # Check if we got a valid response
            if response and len(response.answer) > 0:
                return True
        except Exception:
            pass
        return False

    def scan_port_53(self, ip, source_type="range_scan", source_info=""):
        """Check if port 53 is open and if it's a DNS server"""
        try:
            # Quick port scan
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            result = sock.connect_ex((ip, 53))
            sock.close()
            
            if result == 0:  # Port is open
                # Verify it's actually a DNS server
                if self.test_dns_server(ip):
                    with self.lock:
                        if ip not in self.discovered_servers:
                            self.discovered_servers.add(ip)
                            self.discovery_chain[ip] = (source_type, source_info, "Port scan + DNS test")
                            print(f"✓ Found DNS server: {ip} (via {source_type}: {source_info})")
                    return True
        except Exception:
            pass
        return False

    def scan_ip_range(self, ip_range):
        """Scan an IP range for DNS servers"""
        print(f"Scanning IP range: {ip_range}")
        
        try:
            network = ipaddress.IPv4Network(ip_range, strict=False)
            
            with ThreadPoolExecutor(max_workers=self.threads) as executor:
                futures = []
                
                for ip in network:
                    # Skip network and broadcast addresses
                    if ip == network.network_address or ip == network.broadcast_address:
                        continue
                    
                    future = executor.submit(self.scan_port_53, str(ip), "range_scan", ip_range)
                    futures.append(future)
                
                # Process completed futures
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        pass
                        
        except Exception as e:
            print(f"Error scanning range {ip_range}: {e}")

    def discover_from_ns_records(self, domain_list, query_server='8.8.8.8'):
        """Discover DNS servers by querying NS records of popular domains"""
        print(f"Discovering DNS servers from NS records (using {query_server})...")
        
        for domain in domain_list:
            try:
                # Query NS records
                resolver = dns.resolver.Resolver()
                resolver.nameservers = [query_server]  # Use specified DNS for queries
                answers = resolver.resolve(domain, 'NS')
                
                for ns in answers:
                    ns_name = str(ns.target).rstrip('.')
                    try:
                        # Resolve the NS name to IP
                        ip_answers = resolver.resolve(ns_name, 'A')
                        for ip_answer in ip_answers:
                            ip = str(ip_answer)
                            if self.test_dns_server(ip):
                                with self.lock:
                                    if ip not in self.discovered_servers:
                                        self.discovered_servers.add(ip)
                                        self.discovery_chain[ip] = ("ns_record", query_server, f"{ns_name} for {domain}")
                                        print(f"✓ Found DNS server: {ip} (NS: {ns_name} for {domain}, discovered via {query_server})")
                    except Exception:
                        pass
                        
            except Exception as e:
                print(f"Error querying NS records for {domain}: {e}")

    def get_common_ranges(self):
        """Get common IP ranges where DNS servers might be found"""
        ranges = [
            # Google DNS
            '8.8.8.0/24',
            '8.8.4.0/24',
            
            # Cloudflare
            '1.1.1.0/24',
            '1.0.0.0/24',
            
            # OpenDNS
            '208.67.222.0/24',
            '208.67.220.0/24',
            
            # Quad9
            '9.9.9.0/24',
            '149.112.112.0/24',
            
            # Comodo Secure DNS
            '8.26.56.0/24',
            '8.20.247.0/24',
            
            # Level3
            '209.244.0.0/24',
            '4.2.2.0/24',
            
            # Small sample ranges from major ISPs
            '4.4.4.0/24',
            '4.4.8.0/24',
        ]
        return ranges

    def validate_known_servers(self):
        """Validate and add known public DNS servers"""
        print("Validating known public DNS servers...")
        
        for server in self.known_servers:
            if self.test_dns_server(server):
                with self.lock:
                    if server not in self.discovered_servers:
                        self.discovered_servers.add(server)
                        self.discovery_chain[server] = ("known_server", "built-in_list", "Pre-configured public DNS")
                    print(f"✓ Validated known server: {server}")
            else:
                print(f"✗ Failed to validate: {server}")

    def run_discovery(self, scan_ranges=True, use_ns_records=True):
        """Run the complete DNS server discovery process"""
        print("=" * 60)
        print("DNS Server Discovery Tool")
        print("=" * 60)
        print(f"Starting discovery from Google DNS (8.8.8.8)")
        print(f"Timeout: {self.timeout}s, Threads: {self.threads}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Validate known servers
        self.validate_known_servers()
        
        # Step 2: Discover from NS records
        if use_ns_records:
            popular_domains = [
                'google.com', 'facebook.com', 'youtube.com', 'amazon.com',
                'wikipedia.org', 'twitter.com', 'instagram.com', 'linkedin.com',
                'github.com', 'stackoverflow.com', 'reddit.com', 'microsoft.com'
            ]
            # Use different DNS servers for queries to show relationships
            query_servers = ['8.8.8.8', '1.1.1.1', '208.67.222.222']
            for i, query_server in enumerate(query_servers):
                if query_server in self.discovered_servers:  # Only use validated servers
                    domain_subset = popular_domains[i*4:(i+1)*4]  # Split domains among servers
                    if domain_subset:
                        self.discover_from_ns_records(domain_subset, query_server)
        
        # Step 3: Scan common IP ranges
        if scan_ranges:
            ranges = self.get_common_ranges()
            for ip_range in ranges:
                self.scan_ip_range(ip_range)
        
        end_time = time.time()
        
        # Results
        print("=" * 60)
        print("DISCOVERY COMPLETE")
        print("=" * 60)
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Total DNS servers discovered: {len(self.discovered_servers)}")
        print("\nDiscovered DNS servers:")
        
        sorted_servers = sorted(self.discovered_servers, key=lambda x: ipaddress.IPv4Address(x))
        for i, server in enumerate(sorted_servers, 1):
            print(f"{i:3d}. {server}")
        
        # Show discovery chain
        self.show_discovery_chain()
        
        # Show statistics
        stats = self.get_discovery_statistics()
        print(f"\nDiscovery Statistics:")
        for method, data in stats.items():
            method_name = method.replace('_', ' ').title()
            print(f"  {method_name}: {data['count']} servers from {len(data['sources'])} sources")
        
        return sorted_servers

    def save_results(self, filename=None):
        """Save discovered servers to a file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"discovered_dns_servers_{timestamp}.json"
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_servers': len(self.discovered_servers),
            'servers': sorted(list(self.discovered_servers), key=lambda x: ipaddress.IPv4Address(x)),
            'discovery_chain': {server: {'method': method, 'source': source, 'info': info} 
                              for server, (method, source, info) in self.discovery_chain.items()},
            'statistics': self.get_discovery_statistics()
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
        return filename

    def test_server_performance(self, servers_to_test=None):
        """Test performance of discovered DNS servers"""
        if servers_to_test is None:
            servers_to_test = list(self.discovered_servers)[:10]  # Test top 10
        
        print(f"\nTesting performance of {len(servers_to_test)} servers...")
        performance_results = []
        
        for server in servers_to_test:
            try:
                start_time = time.time()
                query = dns.message.make_query('google.com', dns.rdatatype.A)
                response = dns.query.udp(query, server, timeout=5)
                end_time = time.time()
                
                response_time = (end_time - start_time) * 1000  # Convert to ms
                performance_results.append((server, response_time))
                print(f"  {server}: {response_time:.2f}ms")
                
            except Exception:
                print(f"  {server}: Failed")
        
        # Sort by response time
        performance_results.sort(key=lambda x: x[1])
        print("\nFastest DNS servers:")
        for i, (server, response_time) in enumerate(performance_results[:5], 1):
            print(f"  {i}. {server} - {response_time:.2f}ms")
        
        return performance_results

    def show_discovery_chain(self):
        """Display how each DNS server was discovered"""
        print("\n" + "=" * 80)
        print("DNS SERVER DISCOVERY CHAIN")
        print("=" * 80)
        
        # Group servers by discovery method
        by_method = {}
        for server, (method, source, info) in self.discovery_chain.items():
            if method not in by_method:
                by_method[method] = []
            by_method[method].append((server, source, info))
        
        # Display each method group
        method_names = {
            'known_server': 'Known Public DNS Servers',
            'ns_record': 'Discovered via NS Records',
            'range_scan': 'Found via IP Range Scanning'
        }
        
        for method, servers in by_method.items():
            method_title = method_names.get(method, method.replace('_', ' ').title())
            print(f"\n{method_title}:")
            print("-" * len(method_title))
            
            if method == 'known_server':
                for server, source, info in sorted(servers):
                    print(f"  {server} - {info}")
            
            elif method == 'ns_record':
                # Group by source server
                by_source = {}
                for server, source, info in servers:
                    if source not in by_source:
                        by_source[source] = []
                    by_source[source].append((server, info))
                
                for source_server, discovered_servers in by_source.items():
                    print(f"  Via {source_server}:")
                    for server, info in sorted(discovered_servers):
                        print(f"    → {server} ({info})")
            
            elif method == 'range_scan':
                # Group by IP range
                by_range = {}
                for server, source, info in servers:
                    if source not in by_range:
                        by_range[source] = []
                    by_range[source].append(server)
                
                for ip_range, servers_in_range in by_range.items():
                    print(f"  From range {ip_range}:")
                    for server in sorted(servers_in_range, key=lambda x: ipaddress.IPv4Address(x)):
                        print(f"    → {server}")

    def get_discovery_statistics(self):
        """Get statistics about discovery methods"""
        stats = {}
        for server, (method, source, info) in self.discovery_chain.items():
            if method not in stats:
                stats[method] = {'count': 0, 'sources': set()}
            stats[method]['count'] += 1
            stats[method]['sources'].add(source)
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='Discover DNS servers on the internet')
    parser.add_argument('--timeout', type=int, default=3, help='Connection timeout in seconds')
    parser.add_argument('--threads', type=int, default=50, help='Number of concurrent threads')
    parser.add_argument('--no-ranges', action='store_true', help='Skip IP range scanning')
    parser.add_argument('--no-ns', action='store_true', help='Skip NS record discovery')
    parser.add_argument('--save', type=str, help='Save results to specified file')
    parser.add_argument('--performance', action='store_true', help='Test performance of discovered servers')
    
    args = parser.parse_args()
    
    try:
        # Check if required packages are available
        import dns.resolver
        import dns.query
        import dns.message
    except ImportError:
        print("Error: dnspython package is required.")
        print("Install it with: pip install dnspython")
        return 1
    
    # Create discovery instance
    discovery = DNSDiscovery(timeout=args.timeout, threads=args.threads)
    
    # Run discovery
    servers = discovery.run_discovery(
        scan_ranges=not args.no_ranges,
        use_ns_records=not args.no_ns
    )
    
    # Save results if requested
    if args.save:
        discovery.save_results(args.save)
    
    # Test performance if requested
    if args.performance:
        discovery.test_server_performance()
    
    return 0


if __name__ == "__main__":
    exit(main())
