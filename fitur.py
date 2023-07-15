import re

class FeatureExtraction:
    def __init__(self, url):
        self.url = url
    
    def get_features_list(self):
        features = []
        
        # Fitur 1: Menggunakan IP address sebagai domain
        features.append(1 if re.match(r'\d+\.\d+\.\d+\.\d+', self.url.split("//")[-1].split("/")[0]) else -1)
        
        # Fitur 2: Panjang URL yang melebihi ambang batas
        threshold_length = 75
        features.append(1 if len(self.url) > threshold_length else -1)
        
        # Fitur 3: Menggunakan URL pendek (shortened URL)
        features.append(1 if re.search(r'(bit\.ly|goo\.gl|t\.co)', self.url) else -1)
        
        # Fitur 4: Mengandung simbol @ dalam URL
        features.append(1 if "@" in self.url else -1)
        
        # Fitur 5: Redirecting menggunakan //
        features.append(1 if "//" in self.url else -1)
        
        # Fitur 6: Jumlah subdomain
        subdomains = self.url.split("//")[-1].split(".")
        features.append(len(subdomains) - 2)  # Mengurangi 2 untuk menghilangkan domain utama dan TLD
        
        # Fitur 7: Menggunakan protokol HTTPS
        features.append(1 if self.url.startswith("https://") else -1)
        
        # Fitur 8: Panjang domain registration
        domain = re.findall(r'(?:[a-z]+\.[a-z]+\.[a-z]+|[a-z]+\.[a-z]+)', self.url, re.IGNORECASE)
        if domain:
            domain_length = len(domain[0])
            features.append(1 if domain_length > 10 else -1)
        else:
            features.append(-1)
        
        # # Fitur 9: Ada favicon pada halaman
        # features.append(1 if has_favicon(self.url) else -1)
        # Fitur 10: Menggunakan port non-standar
        features.append(1 if re.match(r':\d+', self.url.split("//")[-1].split("/")[0]) else -1)
        
        # Fitur 11: URL domain HTTPS
        features.append(1 if "https" in self.url.split("//")[1] else -1)
        
        # Fitur 12: URL mengandung request URL
        features.append(1 if "request" in self.url.lower() else -1)
        
        # Fitur 13: URL mengandung anchor URL
        features.append(1 if "#" in self.url else -1)
        
        # Fitur 14: Jumlah links dalam script tags
        script_tags = re.findall(r'<script.*?</script>', self.url, re.IGNORECASE)
        num_links_in_script_tags = sum([1 for script in script_tags if "http" in script])
        features.append(num_links_in_script_tags)
        
        # Fitur 15: Server menggunakan form handler
        features.append(1 if "form" in self.url.lower() else -1)
        
        # Fitur 16: URL mengandung abnormal URL
        features.append(1 if "abnormal" in self.url.lower() else -1)
        
        # Fitur 17: URL melakukan website forwarding
        features.append(1 if "forwarding" in self.url.lower() else -1)
        
        # Fitur 18: Menggunakan custom status bar
        features.append(1 if "status" in self.url.lower() else -1)
        
        # Fitur 19: Disable right click pada halaman
        features.append(1 if "rightclick" in self.url.lower() else -1)
        
        # Fitur 20: Menggunakan popup window
        features.append(1 if "popup" in self.url.lower() else -1)
        
        # Fitur 21: Menggunakan iframe redirection
        features.append(1 if "iframe" in self.url.lower() else -1)
        
        # Fitur 22: URL mengandung DNS recording
        features.append(1 if "dns" in self.url.lower() else -1)
        
        # Fitur 23: Traffic pada website
        features.append(1 if "traffic" in self.url.lower() else -1)
        
        # Fitur 24: PageRank website
        features.append(1 if "pagerank" in self.url.lower() else -1)
        
        # Fitur 25: Google Index website
        features.append(1 if "google" in self.url.lower() else -1)
        return FeatureExtraction

