import socket
import subprocess
import threading
import time
import ipaddress
import json
import csv
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional imports with fallbacks
try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("Warning: folium not available. Geographic maps will be disabled.")

class InternetMapper:
    def __init__(self):
        self.network_graph = nx.Graph()
        self.discovered_hosts = set()
        self.route_cache = {}
        self.geo_data = {}
        self.as_data = {}
        self.results = {
            'nodes': [],
            'edges': [],
            'routes': [],
            'topology': {}
        }
    
    def ping_host(self, host, timeout=3):
        """Check if a host is reachable via ping"""
        try:
            if isinstance(host, str):
                # For domain names, resolve to IP first
                try:
                    ip = socket.gethostbyname(host)
                except socket.gaierror:
                    return False, None
            else:
                ip = str(host)
            
            # Use subprocess for cross-platform ping
            import platform

            platform_system: str = platform.system().lower()

            is_windows: bool = platform_system == "windows"

            param = "-n" if is_windows else "-c"
            command = ["ping", param, "1", "-w" if is_windows else "-W", str(timeout * 1000), ip]

            #command = ["ping", param, "1", ip]
            
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout + 1)
            return result.returncode == 0, ip
        except Exception as e:
            print(f"Error pinging {host}: {e}")
            return False, None
    
    def traceroute(self, target, max_hops=30):
        """Perform traceroute to discover network path"""
        route = []
        try:
            import platform

            platform_system: str = platform.system().lower()

            is_windows: bool = platform_system == "windows"

            max_hops = 1

            if is_windows:
                cmd = ["tracert", "-h", str(max_hops), "-w", "3000", target]
            else:
                cmd = ["traceroute", "-m", str(max_hops), "-w", "3", target]

            #print(cmd)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Parse traceroute output
            lines = result.stdout.split('\n')
            for line in lines:
                if is_windows:
                    # Windows tracert format
                    match = re.search(r'\d+\s+(?:\d+\s*ms\s+)*(?:\d+\s*ms\s+)*(?:\d+\s*ms\s+)?([^\s]+)', line)
                    if match:
                        hop = match.group(1)
                        if hop != '*' and not hop.startswith('Request'):
                            route.append(hop)
                else:
                    # Unix traceroute format
                    match = re.search(r'\d+\s+([^\s\(]+)', line)
                    if match:
                        hop = match.group(1)
                        if hop != '*':
                            route.append(hop)
        
        except Exception as e:
            print(f"Traceroute error for {target}: {e}")
        
        return route
    
    def scan_network_range(self, network_range, max_workers=100):
        """Scan a network range for active hosts"""
        print(f"Scanning network range: {network_range}")
        network = ipaddress.ip_network(network_range, strict=False)
        active_hosts = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.ping_host, str(ip)): ip for ip in network.hosts()}
            
            for future in as_completed(futures):
                ip = futures[future]
                try:
                    is_active, resolved_ip = future.result()
                    if is_active:
                        active_hosts.append(resolved_ip or str(ip))

                        for host in active_hosts:
                            print(f"  Active host found: {host}")

                        self.discovered_hosts.add(resolved_ip or str(ip))
                except Exception as e:
                    print(f"Error scanning {ip}: {e}")
        
        return active_hosts
    
    def get_public_ip_ranges(self):
        """Get common public IP ranges for scanning"""
        # Major cloud providers and common ranges
        ranges = [
            "8.8.8.0/24",      # Google DNS
            "1.1.1.0/24",      # Cloudflare DNS
            "208.67.222.0/24", # OpenDNS
            "4.2.2.0/24",      # Level3
            "208.80.152.0/24", # Wikipedia
            "151.101.0.0/24",  # Reddit/Fastly
            "172.217.0.0/24",  # Google services
            "52.0.0.0/8",      # AWS (sample)
            "20.0.0.0/8",      # Microsoft Azure (sample)
        ]
        return ranges
    
    def discover_topology(self, seed_hosts=None, max_depth=3):
        """Discover network topology starting from seed hosts"""
        if seed_hosts is None:
            seed_hosts = [
                "8.8.8.8",
                "1.1.1.1",
                "google.com",
                "wikipedia.org",
                "github.com",
                "stackoverflow.com"
            ]
        
        visited = set()
        queue = deque([(host, 0) for host in seed_hosts])
        
        while queue:
            current_host, depth = queue.popleft()
            
            if depth > max_depth or current_host in visited:
                continue
            
            visited.add(current_host)
            #print(f"Exploring {current_host} at depth {depth}")
            
            print(f"[{depth}] ping {current_host}")

            # Ping test
            is_active, ip = self.ping_host(current_host)

            if not is_active:
                continue
            
            self.discovered_hosts.add(ip)
            self.network_graph.add_node(ip, hostname=current_host, depth=depth)
            
            print(f"[{depth}] traceroute {current_host}")

            public_ranges = self.get_public_ip_ranges()
            
            for ip_range in public_ranges[:3]:  # Limit to avoid overwhelming
                try:
                    active_hosts = self.scan_network_range(ip_range, max_workers=50)
                    print(f"Found {len(active_hosts)} active hosts in {ip_range}")
                except Exception as e:
                    print(f"Error scanning {ip_range}: {e}")

            # Traceroute to discover path
            routes = None#self.traceroute(current_host)

            routes_valid: bool = routes is not None and len(routes) > 1

            if not routes_valid:
                continue

            for route in routes[1:]:
                print(f"[{depth}] found hop {route}")
                
                queue.append((route, depth + 1))

                self.discovered_hosts.add(route)

            continue

            if route:
                self.route_cache[ip] = route
                
                # Add route nodes and edges
                prev_node = None
                for i, hop in enumerate(route):
                    if hop and hop != '*':
                        self.network_graph.add_node(hop, hop_number=i+1)
                        if prev_node:
                            self.network_graph.add_edge(prev_node, hop)
                        prev_node = hop
                
                # Add discovered IPs to queue for further exploration
                for hop in route[-3:]:  # Only explore last few hops to avoid infinite expansion
                    if hop and hop != '*' and hop not in visited:
                        queue.append((hop, depth + 1))
    
    def get_geolocation_data(self, ip):
        """Get geolocation data for an IP address"""
        try:
            # Using ip-api.com for geolocation (free tier)
            response = requests.get(f"http://ip-api.com/json/{ip}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    return {
                        'country': data.get('country', 'Unknown'),
                        'region': data.get('regionName', 'Unknown'),
                        'city': data.get('city', 'Unknown'),
                        'lat': data.get('lat', 0),
                        'lon': data.get('lon', 0),
                        'isp': data.get('isp', 'Unknown'),
                        'as': data.get('as', 'Unknown')
                    }
        except Exception as e:
            print(f"Error getting geolocation for {ip}: {e}")
        return None
    
    def enrich_with_metadata(self):
        """Enrich discovered hosts with geolocation and AS information"""
        print("Enriching hosts with metadata...")
        
        for host in list(self.discovered_hosts):
            geo_data = self.get_geolocation_data(host)
            if geo_data:
                self.geo_data[host] = geo_data
                # Update graph node with metadata
                if self.network_graph.has_node(host):
                    self.network_graph.nodes[host].update(geo_data)
            
            time.sleep(0.1)  # Rate limiting
    
    def analyze_topology(self):
        """Analyze the discovered network topology"""
        analysis = {
            'total_nodes': self.network_graph.number_of_nodes(),
            'total_edges': self.network_graph.number_of_edges(),
            'connected_components': nx.number_connected_components(self.network_graph),
            'average_degree': sum(dict(self.network_graph.degree()).values()) / self.network_graph.number_of_nodes() if self.network_graph.number_of_nodes() > 0 else 0,
            'diameter': 0,
            'clustering_coefficient': 0,
            'top_degree_nodes': []
        }
        
        if self.network_graph.number_of_nodes() > 0:
            # Get largest connected component for diameter calculation
            if nx.is_connected(self.network_graph):
                analysis['diameter'] = nx.diameter(self.network_graph)
            else:
                largest_cc = max(nx.connected_components(self.network_graph), key=len)
                subgraph = self.network_graph.subgraph(largest_cc)
                if len(largest_cc) > 1:
                    analysis['diameter'] = nx.diameter(subgraph)
            
            # Clustering coefficient
            analysis['clustering_coefficient'] = nx.average_clustering(self.network_graph)
            
            # Top degree nodes
            degrees = dict(self.network_graph.degree())
            analysis['top_degree_nodes'] = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return analysis
    
    def visualize_network(self, output_file="network_topology.png"):
        """Create network visualization"""
        plt.figure(figsize=(20, 16))
        
        if self.network_graph.number_of_nodes() == 0:
            plt.text(0.5, 0.5, 'No network data to visualize', ha='center', va='center')
            plt.savefig(output_file)
            return
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(self.network_graph, k=1, iterations=50)
        
        # Draw nodes with different colors based on properties
        node_colors = []
        node_sizes = []
        
        for node in self.network_graph.nodes():
            degree = self.network_graph.degree(node)
            node_sizes.append(100 + degree * 20)
            
            # Color based on geolocation if available
            if node in self.geo_data:
                country = self.geo_data[node].get('country', 'Unknown')
                if 'United States' in country:
                    node_colors.append('red')
                elif 'Germany' in country:
                    node_colors.append('blue')
                elif 'United Kingdom' in country:
                    node_colors.append('green')
                else:
                    node_colors.append('orange')
            else:
                node_colors.append('gray')
        
        nx.draw_networkx_nodes(self.network_graph, pos, 
                             node_color=node_colors, 
                             node_size=node_sizes,
                             alpha=0.7)
        
        nx.draw_networkx_edges(self.network_graph, pos, 
                             alpha=0.3, 
                             edge_color='gray')
        
        # Add labels for high-degree nodes
        high_degree_nodes = {node: node for node, degree in self.network_graph.degree() if degree > 2}
        nx.draw_networkx_labels(self.network_graph, pos, 
                               labels=high_degree_nodes,
                               font_size=8)
        
        plt.title("Internet Network Topology Map", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Network visualization saved to {output_file}")
    
    def create_geographic_map(self, output_file="geographic_network_map.html"):
        """Create geographic visualization of network nodes"""
        if not FOLIUM_AVAILABLE:
            print("Folium not available. Skipping geographic map creation.")
            return
            
        # Create base map
        map_center = [40.0, 0.0]  # Center on Europe/Atlantic
        network_map = folium.Map(location=map_center, zoom_start=2)
        
        # Add nodes with geolocation data
        for host, geo_data in self.geo_data.items():
            if geo_data and geo_data['lat'] != 0 and geo_data['lon'] != 0:
                folium.Marker(
                    location=[geo_data['lat'], geo_data['lon']],
                    popup=f"IP: {host}<br>ISP: {geo_data.get('isp', 'Unknown')}<br>City: {geo_data.get('city', 'Unknown')}",
                    tooltip=f"{host} ({geo_data.get('city', 'Unknown')})",
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(network_map)
        
        # Add connections between geolocated nodes
        for edge in self.network_graph.edges():
            node1, node2 = edge
            if node1 in self.geo_data and node2 in self.geo_data:
                geo1, geo2 = self.geo_data[node1], self.geo_data[node2]
                if all([geo1.get('lat'), geo1.get('lon'), geo2.get('lat'), geo2.get('lon')]):
                    folium.PolyLine(
                        locations=[[geo1['lat'], geo1['lon']], [geo2['lat'], geo2['lon']]],
                        color='blue',
                        weight=2,
                        opacity=0.6
                    ).add_to(network_map)
        
        network_map.save(output_file)
        print(f"Geographic map saved to {output_file}")
    
    def export_results(self, output_prefix="internet_map"):
        """Export results to various formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export nodes
        nodes_file = f"{output_prefix}_nodes_{timestamp}.csv"
        with open(nodes_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['IP', 'Hostname', 'Country', 'City', 'ISP', 'AS', 'Degree', 'Latitude', 'Longitude'])
            
            for node in self.network_graph.nodes():
                geo = self.geo_data.get(node, {})
                hostname = self.network_graph.nodes[node].get('hostname', node)
                degree = self.network_graph.degree(node)
                
                writer.writerow([
                    node,
                    hostname,
                    geo.get('country', ''),
                    geo.get('city', ''),
                    geo.get('isp', ''),
                    geo.get('as', ''),
                    degree,
                    geo.get('lat', ''),
                    geo.get('lon', '')
                ])
        
        # Export edges
        edges_file = f"{output_prefix}_edges_{timestamp}.csv"
        with open(edges_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Source', 'Target'])
            for edge in self.network_graph.edges():
                writer.writerow(edge)
        
        # Export routes
        routes_file = f"{output_prefix}_routes_{timestamp}.json"
        with open(routes_file, 'w') as f:
            json.dump(self.route_cache, f, indent=2)
        
        print(f"Results exported:")
        print(f"  - Nodes: {nodes_file}")
        print(f"  - Edges: {edges_file}")
        print(f"  - Routes: {routes_file}")
    
    def run_full_scan(self):
        """Run complete internet mapping process"""
        print("Starting Internet Mapping Process...")
        print("=" * 50)
        
        # Step 1: Discover topology
        print("Step 1: Discovering network topology...")
        self.discover_topology()
        
        # Step 2: Scan additional ranges
        print("\nStep 2: Scanning additional network ranges...")
        public_ranges = self.get_public_ip_ranges()
        for ip_range in public_ranges[:3]:  # Limit to avoid overwhelming
            try:
                active_hosts = self.scan_network_range(ip_range, max_workers=50)
                print(f"Found {len(active_hosts)} active hosts in {ip_range}")
            except Exception as e:
                print(f"Error scanning {ip_range}: {e}")
        
        # Step 3: Enrich with metadata
        print("\nStep 3: Enriching with geolocation data...")
        self.enrich_with_metadata()
        
        # Step 4: Analyze topology
        print("\nStep 4: Analyzing network topology...")
        analysis = self.analyze_topology()
        
        print(f"\nTopology Analysis:")
        print(f"  Total nodes: {analysis['total_nodes']}")
        print(f"  Total edges: {analysis['total_edges']}")
        print(f"  Connected components: {analysis['connected_components']}")
        print(f"  Average degree: {analysis['average_degree']:.2f}")
        print(f"  Network diameter: {analysis['diameter']}")
        print(f"  Clustering coefficient: {analysis['clustering_coefficient']:.3f}")
        
        print("\nTop 5 nodes by degree:")
        for node, degree in analysis['top_degree_nodes'][:55555]:
            print(f"  {node}: {degree} connections")
        
        # Step 5: Create visualizations
        print("\nStep 5: Creating visualizations...")
        self.visualize_network()
        if FOLIUM_AVAILABLE:
            try:
                self.create_geographic_map()
            except Exception as e:
                print(f"Could not create geographic map: {e}")
        else:
            print("Skipping geographic map (folium not available)")
        
        # Step 6: Export results
        print("\nStep 6: Exporting results...")
        self.export_results()
        
        print("\nInternet mapping complete!")
        return analysis


def main():
    """Main function to run internet mapping"""
    mapper = InternetMapper()
    
    print("Internet Topology Mapper")
    print("========================")
    print("This program will attempt to map network topology and routes.")
    print("Note: This is for educational purposes and should be used responsibly.")
    print()
    
    try:
        # Run the mapping process
        analysis = mapper.run_full_scan()
        
        print("\nMapping session completed successfully!")
        print("Check the generated files for detailed results.")
        
    except KeyboardInterrupt:
        print("\nMapping interrupted by user.")
    except Exception as e:
        print(f"Error during mapping: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()