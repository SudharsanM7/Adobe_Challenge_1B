import fitz
import re
import json
import time
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class PersonaDrivenDocumentAnalyzer:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=1
        )
        
        # Domain-specific keywords for different personas
        self.domain_keywords = {
            'travel_planner': {
                'high_priority': ['attractions', 'hotels', 'restaurants', 'activities', 'transportation', 
                                'itinerary', 'booking', 'accommodation', 'dining', 'sightseeing', 'budget',
                                'group', 'friends', 'young', 'college', 'entertainment', 'nightlife'],
                'locations': ['cities', 'beaches', 'coastal', 'downtown', 'center', 'district'],
                'practical': ['tips', 'tricks', 'packing', 'planning', 'advice', 'guide']
            },
            'researcher': {
                'high_priority': ['methodology', 'results', 'analysis', 'data', 'experiment', 'study'],
                'academic': ['abstract', 'introduction', 'conclusion', 'discussion', 'literature']
            },
            'business_analyst': {
                'high_priority': ['revenue', 'financial', 'market', 'strategy', 'investment', 'growth'],
                'metrics': ['performance', 'roi', 'profit', 'cost', 'budget', 'forecast']
            }
        }
        
    def extract_document_content(self, pdf_path):
        """Extract content from PDF with enhanced section detection"""
        doc = fitz.open(pdf_path)
        content = {
            'title': '',
            'sections': [],
            'full_text': '',
            'metadata': {
                'filename': pdf_path.name,
                'total_pages': len(doc)
            }
        }
        
        all_text = []
        current_section = None
        page_content = {}
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = []
            
            # Extract text with formatting
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    if not line["spans"]:
                        continue
                    
                    line_text = "".join(span["text"] for span in line["spans"]).strip()
                    if len(line_text) < 3:
                        continue
                    
                    # Get main span for formatting
                    main_span = max(line["spans"], key=lambda s: len(s["text"]))
                    is_heading = self._is_heading(line_text, main_span, page_num == 0)
                    
                    if is_heading:
                        # Save previous section
                        if current_section and current_section['content'].strip():
                            content['sections'].append(current_section)
                        
                        # Start new section
                        current_section = {
                            'title': line_text.strip(),
                            'content': '',
                            'page_start': page_num + 1,
                            'page_end': page_num + 1,
                            'importance_score': 0.0
                        }
                        
                        # Set document title from first major heading
                        if not content['title'] and page_num <= 1:
                            content['title'] = line_text.strip()
                    else:
                        # Add to current section
                        if current_section is None:
                            # Create default section if no heading found
                            section_title = "Introduction" if page_num == 0 else f"Content Page {page_num + 1}"
                            current_section = {
                                'title': section_title,
                                'content': '',
                                'page_start': page_num + 1,
                                'page_end': page_num + 1,
                                'importance_score': 0.0
                            }
                        
                        current_section['content'] += line_text + ' '
                        current_section['page_end'] = page_num + 1
                    
                    page_text.append(line_text)
                    all_text.append(line_text)
            
            page_content[page_num + 1] = ' '.join(page_text)
        
        # Add final section
        if current_section and current_section['content'].strip():
            content['sections'].append(current_section)
        
        # If no sections found, create sections from page content
        if not content['sections']:
            for page_num, text in page_content.items():
                if text.strip():
                    content['sections'].append({
                        'title': f"Page {page_num} Content",
                        'content': text,
                        'page_start': page_num,
                        'page_end': page_num,
                        'importance_score': 0.0
                    })
        
        content['full_text'] = '\n'.join(all_text)
        doc.close()
        
        return content
    
    def _is_heading(self, text, span, is_first_page):
        """Enhanced heading detection"""
        # Skip very long text (likely paragraphs)
        if len(text) > 100:
            return False
        
        # Numbered patterns
        numbered_patterns = [
            r'^(\d+\.?\s+)',  # 1. or 1
            r'^([IVXLCDM]+\.?\s+)',  # Roman numerals
            r'^([A-Z]\.?\s+)',  # A. or A
            r'^(Chapter\s+\d+)',  # Chapter 1
            r'^(Section\s+\d+)',  # Section 1
        ]
        
        for pattern in numbered_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        # Common section headers
        section_headers = {
            'introduction', 'overview', 'summary', 'conclusion', 'abstract',
            'methodology', 'methods', 'results', 'discussion', 'references',
            'acknowledgments', 'appendix', 'bibliography', 'index',
            # Travel-specific headers
            'cities', 'attractions', 'restaurants', 'hotels', 'cuisine',
            'activities', 'nightlife', 'transportation', 'tips', 'tricks',
            'culture', 'traditions', 'history', 'things to do', 'guide',
            'coastal adventures', 'culinary experiences', 'entertainment',
            'packing tips', 'general packing tips and tricks'
        }
        
        if any(header in text.lower() for header in section_headers):
            return True
        
        # Font-based detection
        size = span.get('size', 12)
        bold = bool(span.get('flags', 0) & 16)
        
        # Larger or bold text likely to be headings
        if bold and size > 12:
            return True
        
        if size > 14:
            return True
        
        # All caps (but not too long)
        if text.isupper() and 3 <= len(text) <= 50:
            return True
        
        return False
    
    def analyze_persona_job(self, persona_role, job_task):
        """Enhanced persona and job analysis"""
        combined_text = f"{persona_role} {job_task}".lower()
        
        # Extract key terms
        words = word_tokenize(combined_text)
        filtered_words = [
            self.stemmer.stem(word) for word in words 
            if word.isalnum() and word not in self.stop_words and len(word) > 2
        ]
        
        # Get most important terms
        word_freq = Counter(filtered_words)
        key_terms = [term for term, freq in word_freq.most_common(30)]
        
        # Determine domain based on persona
        domain = 'travel_planner'  # Default
        if 'research' in combined_text or 'study' in combined_text or 'analysis' in combined_text:
            domain = 'researcher'
        elif 'business' in combined_text or 'financial' in combined_text or 'investment' in combined_text:
            domain = 'business_analyst'
        elif 'travel' in combined_text or 'trip' in combined_text or 'planner' in combined_text:
            domain = 'travel_planner'
        
        # Extract domain-specific focus areas
        focus_keywords = self.domain_keywords.get(domain, {})
        relevant_terms = []
        
        for category, terms in focus_keywords.items():
            for term in terms:
                if term in combined_text:
                    relevant_terms.append(term)
        
        return {
            'key_terms': key_terms,
            'domain': domain,
            'focus_keywords': relevant_terms,
            'raw_text': combined_text,
            'persona_role': persona_role,
            'job_task': job_task
        }
    
    def score_section_relevance(self, section, persona_analysis, all_sections_text):
        """Enhanced relevance scoring"""
        section_text = f"{section['title']} {section['content']}".lower()
        
        # TF-IDF similarity
        try:
            corpus = [persona_analysis['raw_text']] + [section_text] + all_sections_text
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            
            persona_vector = tfidf_matrix[0]
            section_vector = tfidf_matrix[1]
            
            similarity = cosine_similarity(persona_vector, section_vector)[0][0]
        except:
            similarity = 0.0
        
        # Domain-specific keyword matching
        domain_score = 0.0
        domain_keywords = self.domain_keywords.get(persona_analysis['domain'], {})
        
        for category, keywords in domain_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in section_text)
            if category == 'high_priority':
                domain_score += matches * 0.3
            else:
                domain_score += matches * 0.1
        
        # Title-based scoring
        title_score = 0.0
        section_title = section['title'].lower()
        
        # Higher score for titles that match job requirements
        job_keywords = persona_analysis['job_task'].lower().split()
        title_matches = sum(1 for keyword in job_keywords if keyword in section_title)
        title_score = min(title_matches * 0.2, 0.6)
        
        # Section type importance
        important_sections = {
            'cities': 0.4, 'attractions': 0.4, 'restaurants': 0.3, 'hotels': 0.3,
            'things to do': 0.4, 'activities': 0.4, 'nightlife': 0.3, 'entertainment': 0.3,
            'tips': 0.3, 'tricks': 0.3, 'cuisine': 0.3, 'culinary': 0.3,
            'coastal': 0.3, 'adventures': 0.3, 'packing': 0.2, 'guide': 0.2
        }
        
        section_type_score = 0.0
        for section_type, score in important_sections.items():
            if section_type in section_title:
                section_type_score = max(section_type_score, score)
        
        # Length penalty for very short sections
        length_penalty = 0.0
        if len(section['content']) < 100:
            length_penalty = -0.1
        
        # Combined score
        final_score = (
            similarity * 0.3 +
            domain_score * 0.25 +
            title_score * 0.25 +
            section_type_score * 0.15 +
            length_penalty + 0.05  # Base score
        )
        
        return min(max(final_score, 0), 1.0)
    
    def extract_subsections(self, section, persona_analysis, max_length=500):
        """Extract relevant subsections from content"""
        content = section['content'].strip()
        if not content:
            return []
        
        # Split content into meaningful chunks
        sentences = sent_tokenize(content)
        if not sentences:
            return []
        
        # If content is short, return as single subsection
        if len(content) <= max_length:
            return [{
                'document': section.get('document', ''),
                'refined_text': content,
                'page_number': section['page_start']
            }]
        
        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Score and select best chunks
        scored_chunks = []
        for chunk in chunks:
            # Score based on keyword relevance
            chunk_lower = chunk.lower()
            score = 0
            
            # Domain keyword scoring
            domain_keywords = self.domain_keywords.get(persona_analysis['domain'], {})
            for category, keywords in domain_keywords.items():
                matches = sum(1 for keyword in keywords if keyword in chunk_lower)
                if category == 'high_priority':
                    score += matches * 2
                else:
                    score += matches
            
            # Job-specific keyword scoring
            job_words = persona_analysis['job_task'].lower().split()
            score += sum(1 for word in job_words if word in chunk_lower)
            
            scored_chunks.append((score, chunk))
        
        # Sort by score and return top chunks
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Return top 3 chunks or all if fewer
        result = []
        for score, chunk in scored_chunks[:3]:
            if chunk.strip():
                result.append({
                    'document': section.get('document', ''),
                    'refined_text': chunk.strip(),
                    'page_number': section['page_start']
                })
        
        return result if result else [{
            'document': section.get('document', ''),
            'refined_text': content[:max_length] + ('...' if len(content) > max_length else ''),
            'page_number': section['page_start']
        }]
    
    def process_document_collection(self, input_data):
        """Main processing function matching expected format"""
        start_time = time.time()
        
        # Parse input data structure
        documents = input_data['documents']
        persona_info = input_data['persona']
        job_info = input_data['job_to_be_done']
        
        # Extract persona and job details
        persona_role = persona_info.get('role', 'General User')
        job_task = job_info.get('task', 'General task')
        
        # Analyze persona and job
        persona_analysis = self.analyze_persona_job(persona_role, job_task)
        
        # Process all documents
        all_sections = []
        all_sections_text = []
        document_filenames = []
        
        for doc_info in documents:
            filename = doc_info['filename']
            document_filenames.append(filename)
            
            doc_path = Path("/app/input") / filename
            
            if not doc_path.exists():
                print(f"Warning: Document {filename} not found")
                continue
            
            print(f"Processing {filename}...")
            content = self.extract_document_content(doc_path)
            
            for section in content['sections']:
                section['document'] = filename
                all_sections.append(section)
                all_sections_text.append(f"{section['title']} {section['content']}")
        
        # Score all sections for relevance
        for section in all_sections:
            section['importance_score'] = self.score_section_relevance(
                section, persona_analysis, all_sections_text
            )
        
        # Sort sections by importance
        all_sections.sort(key=lambda x: x['importance_score'], reverse=True)
        
        # Prepare extracted sections (top 5-10)
        extracted_sections = []
        subsection_analysis = []
        
        top_sections = all_sections[:10]  # Take top 10 sections
        
        for rank, section in enumerate(top_sections, 1):
            extracted_sections.append({
                'document': section['document'],
                'section_title': section['title'],
                'importance_rank': rank,
                'page_number': section['page_start']
            })
            
            # Extract subsections for top 5 sections
            if rank <= 5:
                subsections = self.extract_subsections(section, persona_analysis)
                subsection_analysis.extend(subsections)
        
        # Prepare final output matching expected format
        processing_time = time.time() - start_time
        
        result = {
            'metadata': {
                'input_documents': document_filenames,
                'persona': persona_role,
                'job_to_be_done': job_task,
                'processing_timestamp': datetime.now().isoformat()
            },
            'extracted_sections': extracted_sections[:5],  # Limit to top 5 as in sample
            'subsection_analysis': subsection_analysis
        }
        
        return result

def main():
    """Main processing function"""
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load input configuration
    config_file = input_dir / "input.json"
    if not config_file.exists():
        print("Error: input.json not found in input directory")
        return
    
    with open(config_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    print("Initializing Persona-Driven Document Analyzer...")
    analyzer = PersonaDrivenDocumentAnalyzer()
    
    print("Processing document collection...")
    result = analyzer.process_document_collection(input_data)
    
    # Save output
    output_file = output_dir / "challenge1b_output.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete!")
    print(f"Extracted {len(result['extracted_sections'])} sections")
    print(f"Generated {len(result['subsection_analysis'])} subsections")

if __name__ == "__main__":
    main()
