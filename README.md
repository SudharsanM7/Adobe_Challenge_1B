# Challenge 1B: Persona-Driven Document Intelligence

## Overview
This solution extracts and prioritizes relevant content from document collections based on specific user personas and their job requirements. It processes 3-10 PDFs to identify the most important sections and subsections for targeted users.

## Features
- **Multi-domain support**: Travel planning, academic research, business analysis
- **Intelligent section ranking**: Combines semantic similarity with domain expertise
- **Granular content extraction**: Provides detailed subsection analysis
- **Fast processing**: Completes analysis within 60 seconds
- **CPU-optimized**: No GPU requirements, runs on standard hardware

## Architecture
The system implements a four-stage pipeline:
1. **Document Processing**: PDF parsing with structure detection
2. **Persona Analysis**: Role and job requirement understanding
3. **Relevance Scoring**: Multi-factor content evaluation
4. **Content Extraction**: Section ranking and subsection analysis

## Technical Stack
- **PDF Processing**: PyMuPDF for robust text extraction
- **NLP**: NLTK for text processing and tokenization
- **ML**: Scikit-learn for TF-IDF vectorization and similarity
- **Performance**: Lightweight algorithms optimized for CPU execution

## Input Requirements
- **Documents**: 3-10 related PDF files
- **Persona**: User role description with expertise areas
- **Job**: Specific task the persona needs to accomplish
- **Format**: JSON configuration with document references

## Output Format
{
"metadata": {
"input_documents": ["doc1.pdf", "doc2.pdf"],
"persona": "Travel Planner",
"job_to_be_done": "Plan a 4-day trip",
"processing_timestamp": "2025-01-28T..."
},
"extracted_sections": [
{
"document": "doc1.pdf",
"section_title": "Section Name",
"importance_rank": 1,
"page_number": 5
}
],"subsection_analysis": [
{
"document": "doc1.pdf",
"refined_text": "Extracted content...",
"page_number": 5
}
]
}

## Usage Examples

### Travel Planning
- **Persona**: Travel Planner
- **Job**: Plan group trips, find accommodations and activities
- **Output**: Prioritizes cities, attractions, restaurants, practical tips

### Academic Research
- **Persona**: PhD Researcher
- **Job**: Literature review and methodology analysis
- **Output**: Focuses on methods, results, datasets, benchmarks

### Business Analysis
- **Persona**: Investment Analyst
- **Job**: Market analysis and financial trends
- **Output**: Emphasizes revenue data, market positioning, R&D investments

## Performance Characteristics
- **Processing Time**: ≤60 seconds for 3-5 documents
- **Memory Usage**: <1GB RAM
- **Model Size**: Lightweight NLP models only
- **Scalability**: Handles up to 10 documents efficiently

## Quality Metrics
- **Section Relevance**: High precision targeting persona needs
- **Content Coverage**: Comprehensive analysis across document types
- **Processing Efficiency**: Optimized for competition constraints
- **Domain Adaptability**: Generic framework supporting diverse use cases

## Build and Run
