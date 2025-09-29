import json
import sys
import argparse
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Any


class ArxivHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.papers_data = self._load_json(os.path.join(current_dir, 'sample_data', 'papers.json'))
        self.corpus_data = self._load_json(os.path.join(current_dir, 'sample_data', 'corpus_analysis.json'))
        super().__init__(*args, **kwargs)
    
    def _load_json(self, filepath: str) -> Any:
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    def _send_json_response(self, data: Any, status_code: int = 200):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response_json = json.dumps(data, indent=2)
        self.wfile.write(response_json.encode('utf-8'))
    
    def _send_error_response(self, status_code: int, message: str):
        error_data = {"error": message}
        self._send_json_response(error_data, status_code)
    
    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query_params = parse_qs(parsed_url.query)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            if path == '/papers':
                self._handle_papers_list()
                result_count = len(self.papers_data) if self.papers_data else 0
                print(f"[{timestamp}] GET /papers - 200 OK ({result_count} results)")
            elif path.startswith('/papers/'):
                arxiv_id = path.split('/')[-1]
                self._handle_paper_detail(arxiv_id)
                print(f"[{timestamp}] GET /papers/{arxiv_id} - 200 OK")
            elif path == '/search':
                query = query_params.get('q', [''])[0]
                if not query:
                    self._send_error_response(400, "Missing query parameter 'q'")
                    print(f"[{timestamp}] GET /search - 400 Bad Request")
                    return
                self._handle_search(query)
                print(f"[{timestamp}] GET /search?q={query} - 200 OK")
            elif path == '/stats':
                self._handle_stats()
                print(f"[{timestamp}] GET /stats - 200 OK")
            else:
                self._send_error_response(404, "Endpoint not found")
                print(f"[{timestamp}] GET {path} - 404 Not Found")
        
        except Exception as e:
            self._send_error_response(500, f"Server error: {str(e)}")
            print(f"[{timestamp}] GET {path} - 500 Server Error")
    
    def _handle_papers_list(self):
        if not self.papers_data:
            self._send_error_response(500, "Papers data not available")
            return
        
        papers_list = []
        for paper in self.papers_data:
            papers_list.append({
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "authors": paper["authors"],
                "categories": paper["categories"]
            })
        
        self._send_json_response(papers_list)
    
    def _handle_paper_detail(self, arxiv_id: str):
        if not self.papers_data:
            self._send_error_response(500, "Papers data not available")
            return
        
        paper = None
        for p in self.papers_data:
            if p["arxiv_id"] == arxiv_id:
                paper = p
                break
        
        if not paper:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] GET /papers/{arxiv_id} - 404 Not Found")
            self._send_error_response(404, f"Paper with ID '{arxiv_id}' not found")
            return
        
        self._send_json_response(paper)
    
    def _handle_search(self, query: str):
        if not self.papers_data:
            self._send_error_response(500, "Papers data not available")
            return
        
        query_terms = query.lower().split()
        results = []
        
        for paper in self.papers_data:
            title_lower = paper["title"].lower()
            abstract_lower = paper["abstract"].lower()
            
            matches_in = []

            title_matches = 0
            abstract_matches = 0
            
            for term in query_terms:
                title_count = title_lower.count(term)
                if title_count > 0:
                    title_matches += title_count
                    if "title" not in matches_in:
                        matches_in.append("title")
                
                abstract_count = abstract_lower.count(term)
                if abstract_count > 0:
                    abstract_matches += abstract_count
                    if "abstract" not in matches_in:
                        matches_in.append("abstract")
            
            match_score = title_matches * 2 + abstract_matches
            
            if matches_in:
                results.append({
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper["title"],
                    "match_score": match_score,
                    "matches_in": matches_in
                })
        
        results.sort(key=lambda x: x["match_score"], reverse=True)
        
        response = {
            "query": query,
            "results": results
        }
        
        self._send_json_response(response)
    
    def _handle_stats(self):
        if not self.corpus_data:
            self._send_error_response(500, "Corpus data not available")
            return
        
        stats_response = {
            "total_papers": self.corpus_data["papers_processed"],
            "total_words": self.corpus_data["corpus_stats"]["total_words"],
            "unique_words": self.corpus_data["corpus_stats"]["unique_words_global"],
            "top_10_words": self.corpus_data["top_50_words"][:10],  # Get top 10
            "category_distribution": self.corpus_data["category_distribution"]
        }
        
        self._send_json_response(stats_response)


def run_server(port: int = 8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, ArxivHandler)
    
    print(f"Starting ArXiv API server on port {port}")
    print(f"Access at: http://localhost:{port}")
    print("")
    print("Available endpoints:")
    print("  GET /papers")
    print("  GET /papers/{arxiv_id}")
    print("  GET /search?q={query}")
    print("  GET /stats")
    print("")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()


def main():
    parser = argparse.ArgumentParser(description='ArXiv Papers HTTP Server')
    parser.add_argument('port', nargs='?', type=int, default=8080,
                       help='Port number to run the server on (default: 8080)')
    
    args = parser.parse_args()
    
    if args.port < 1024 or args.port > 65535:
        print("Error: Port must be between 1024 and 65535")
        sys.exit(1)
    
    run_server(args.port)


if __name__ == "__main__":
    main()
