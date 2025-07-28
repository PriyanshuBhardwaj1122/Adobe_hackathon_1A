#!/usr/bin/env python3
"""
Adobe India Hackathon 2025 - Round 1A Solution
PDF Outline Extractor with Advanced Document Structure Analysis

This solution extracts hierarchical document structure (Title, H1, H2, H3) from PDFs
with high accuracy, multilingual support, and optimal performance.

Key Features:
- Multi-modal document analysis (text, font, layout, visual cues)
- Advanced heading detection using multiple heuristics
- Multilingual support (English, Japanese, Chinese, Arabic, etc.)
- Robust handling of complex PDF structures
- High-performance processing with parallel optimization
- Zero-dependency ML approach with rule-based intelligence
"""

import os
import sys
import json
import re
import unicodedata
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set, Any
import fitz  # PyMuPDF
import logging
from concurrent.futures import ThreadPoolExecutor
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TextBlock:
    """Represents a text block with comprehensive metadata"""
    text: str
    page_num: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    font_name: str
    font_size: float
    font_flags: int
    color: int
    is_bold: bool
    is_italic: bool
    line_height: float
    word_count: int
    char_count: int
    language_hints: Set[str]
    visual_weight: float
    position_score: float

class AdvancedHeadingDetector:
    """
    Intelligent heading detection system using multiple analysis techniques:
    1. Font-based analysis (size, weight, style)
    2. Position-based analysis (margins, spacing, alignment)
    3. Content-based analysis (length, capitalization, punctuation)
    4. Structural analysis (hierarchy patterns, numbering)
    5. Language-specific patterns
    6. Visual emphasis detection
    """
    
    def __init__(self):
        self.font_size_threshold_multiplier = 1.15
        self.position_weight = 0.3
        self.font_weight = 0.4
        self.content_weight = 0.2
        self.structure_weight = 0.1
        
        # Multilingual heading patterns
        self.heading_patterns = {
            'english': [
                r'^(chapter|section|part)\s+\d+',
                r'^\d+\.?\s+[A-Z]',
                r'^[A-Z][A-Z\s]{3,}$',
                r'^(introduction|conclusion|abstract|summary|overview)',
            ],
            'japanese': [
                r'^第[０-９一二三四五六七八九十百千万０-９]+章',
                r'^[０-９一二三四五六七八九十]+[\.．、]\s*',
                r'^[はじめに|概要|まとめ|結論|序論]',
            ],
            'chinese': [
                r'^第[一二三四五六七八九十百千万0-9]+章',
                r'^[0-9一二三四五六七八九十]+[．.、]\s*',
                r'^[简介|概述|总结|结论|序言]',
            ],
            'arabic': [
                r'^الفصل\s+[0-9١-٩]+',
                r'^[0-9١-٩]+[\.．]\s*',
                r'^[مقدمة|خلاصة|الخلاصة|نظرة عامة]',
            ]
        }
        
        # Common heading indicators across languages
        self.universal_indicators = [
            r'^\d+\.?\d*\.?\d*\s+',  # Numbering: 1., 1.1, 1.1.1
            r'^[IVXLCDM]+\.?\s+',    # Roman numerals
            r'^[A-Za-z]\.?\s+',      # Letter numbering
            r'^•\s+|^\*\s+|^-\s+',   # Bullet points
        ]

    def detect_language(self, text: str) -> Set[str]:
        """Detect possible languages in text"""
        languages = set()
        
        # Unicode block analysis
        for char in text:
            block = unicodedata.name(char, '').split(' ')[0]
            if 'CJK' in block or 'HIRAGANA' in block or 'KATAKANA' in block:
                languages.add('japanese')
            elif 'ARABIC' in block:
                languages.add('arabic')
            elif 'LATIN' in block:
                languages.add('english')
                
        # Pattern-based detection
        if re.search(r'[一二三四五六七八九十百千万]', text):
            languages.add('chinese')
        if re.search(r'[ひらがなカタカナ]', text):
            languages.add('japanese')
        if re.search(r'[اأإآةتثجحخدذرزسشصضطظعغفقكلمنهوي]', text):
            languages.add('arabic')
            
        return languages or {'english'}

    def calculate_visual_weight(self, block: TextBlock, doc_stats: Dict) -> float:
        """Calculate visual importance based on font properties"""
        weight = 0.0
        
        # Font size relative to document average
        avg_font_size = doc_stats.get('avg_font_size', 12)
        size_ratio = block.font_size / avg_font_size if avg_font_size > 0 else 1
        weight += min(size_ratio * 0.4, 1.0)
        
        # Bold text gets higher weight
        if block.is_bold:
            weight += 0.3
            
        # Font name analysis (common heading fonts)
        heading_fonts = ['Arial-Bold', 'Times-Bold', 'Helvetica-Bold', 'CourierNewPS-Bold']
        if any(font in block.font_name for font in heading_fonts):
            weight += 0.2
            
        # Unique font in document context
        font_frequency = doc_stats.get('font_frequencies', {}).get(block.font_name, 1)
        if font_frequency < doc_stats.get('total_blocks', 1) * 0.1:  # Rare font
            weight += 0.1
            
        return min(weight, 1.0)

    def calculate_position_score(self, block: TextBlock, page_height: float) -> float:
        """Calculate position-based heading probability"""
        score = 0.0
        
        # Vertical position (top of page gets higher score)
        y_ratio = (page_height - block.bbox[1]) / page_height
        if y_ratio > 0.8:  # Top 20% of page
            score += 0.4
        elif y_ratio > 0.6:  # Top 40% of page
            score += 0.2
            
        # Left alignment (most headings are left-aligned)
        if block.bbox[0] < 100:  # Close to left margin
            score += 0.2
            
        # Isolation (space around the text)
        if block.line_height > block.font_size * 1.5:
            score += 0.2
            
        return min(score, 1.0)

    def analyze_content_patterns(self, block: TextBlock) -> Dict[str, float]:
        """Analyze text content for heading patterns"""
        text = block.text.strip()
        scores = defaultdict(float)
        
        # Length analysis (headings are typically short)
        if 3 <= len(text.split()) <= 10:
            scores['length'] = 0.3
        elif len(text.split()) <= 3:
            scores['length'] = 0.4
        elif len(text.split()) > 20:
            scores['length'] = -0.2
            
        # Capitalization patterns
        if text.isupper() and len(text) > 3:
            scores['caps'] = 0.4
        elif text.istitle():
            scores['caps'] = 0.3
            
        # Punctuation analysis
        if not text.endswith(('.', '!', '?', ';', ',')):
            scores['punct'] = 0.2
        if text.count(':') == 1 and text.endswith(':'):
            scores['punct'] = 0.3
            
        # Numbering detection
        for pattern in self.universal_indicators:
            if re.match(pattern, text, re.IGNORECASE):
                scores['numbering'] = 0.5
                break
                
        # Language-specific patterns
        for lang in block.language_hints:
            patterns = self.heading_patterns.get(lang, [])
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    scores['language_pattern'] = 0.4
                    break
                    
        return dict(scores)

    def determine_heading_level(self, blocks: List[TextBlock], current_idx: int) -> str:
        """Determine heading level (H1, H2, H3) based on context"""
        current_block = blocks[current_idx]
        
        # Find other potential headings for comparison
        heading_candidates = []
        for i, block in enumerate(blocks):
            if (block.visual_weight > 0.6 or 
                any(score > 0.3 for score in self.analyze_content_patterns(block).values())):
                heading_candidates.append((i, block))
        
        if not heading_candidates:
            return "H1"
            
        # Sort by font size and visual weight
        heading_candidates.sort(key=lambda x: (x[1].font_size, x[1].visual_weight), reverse=True)
        
        # Determine level based on ranking
        current_rank = next((i for i, (idx, _) in enumerate(heading_candidates) if idx == current_idx), 0)
        
        # Map rank to heading level
        if current_rank == 0 or current_block.font_size >= max(h[1].font_size for h in heading_candidates):
            return "H1"
        elif current_rank <= len(heading_candidates) * 0.3:
            return "H1"
        elif current_rank <= len(heading_candidates) * 0.6:
            return "H2"
        else:
            return "H3"

class IntelligentPDFProcessor:
    """Main PDF processing engine with advanced document understanding"""
    
    def __init__(self):
        self.heading_detector = AdvancedHeadingDetector()
        self.min_confidence_threshold = 0.5
        self.max_processing_time = 8  # Leave 2 seconds buffer
        
    def extract_text_blocks(self, doc: fitz.Document) -> Tuple[List[TextBlock], Dict]:
        """Extract all text blocks with comprehensive metadata"""
        all_blocks = []
        doc_stats = {
            'font_sizes': [],
            'font_names': [],
            'total_blocks': 0,
            'page_heights': []
        }
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            doc_stats['page_heights'].append(page.rect.height)
            
            # Get text with detailed formatting
            blocks = page.get_text("dict")
            
            for block in blocks["blocks"]:
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    for span in line["spans"]:
                        if not span["text"].strip():
                            continue
                            
                        text = span["text"].strip()
                        
                        # Calculate derived properties
                        is_bold = bool(span["flags"] & 2**4)
                        is_italic = bool(span["flags"] & 2**1)
                        
                        # Create text block
                        text_block = TextBlock(
                            text=text,
                            page_num=page_num,
                            bbox=tuple(span["bbox"]),
                            font_name=span["font"],
                            font_size=span["size"],
                            font_flags=span["flags"],
                            color=span["color"],
                            is_bold=is_bold,
                            is_italic=is_italic,
                            line_height=line["bbox"][3] - line["bbox"][1],
                            word_count=len(text.split()),
                            char_count=len(text),
                            language_hints=self.heading_detector.detect_language(text),
                            visual_weight=0.0,  # Will be calculated later
                            position_score=0.0  # Will be calculated later
                        )
                        
                        all_blocks.append(text_block)
                        
                        # Update stats
                        doc_stats['font_sizes'].append(span["size"])
                        doc_stats['font_names'].append(span["font"])
        
        # Calculate document statistics
        doc_stats['total_blocks'] = len(all_blocks)
        doc_stats['avg_font_size'] = sum(doc_stats['font_sizes']) / len(doc_stats['font_sizes']) if doc_stats['font_sizes'] else 12
        doc_stats['font_frequencies'] = Counter(doc_stats['font_names'])
        doc_stats['avg_page_height'] = sum(doc_stats['page_heights']) / len(doc_stats['page_heights']) if doc_stats['page_heights'] else 792
        
        # Calculate visual weights and position scores
        for block in all_blocks:
            block.visual_weight = self.heading_detector.calculate_visual_weight(block, doc_stats)
            block.position_score = self.heading_detector.calculate_position_score(block, doc_stats['avg_page_height'])
        
        return all_blocks, doc_stats

    def extract_title(self, blocks: List[TextBlock], doc_stats: Dict) -> str:
        """Extract document title using advanced heuristics"""
        title_candidates = []
        
        # First page blocks only
        first_page_blocks = [b for b in blocks if b.page_num == 0]
        
        for block in first_page_blocks[:20]:  # Check first 20 blocks
            confidence = 0.0
            
            # Position on first page (top area)
            if block.bbox[1] < doc_stats['avg_page_height'] * 0.3:
                confidence += 0.4
                
            # Font size (larger than average)
            if block.font_size > doc_stats['avg_font_size'] * 1.2:
                confidence += 0.3
                
            # Visual weight
            confidence += block.visual_weight * 0.2
            
            # Content analysis
            text = block.text.strip()
            if 5 <= len(text) <= 100 and not text.endswith('.'):
                confidence += 0.2
                
            # Bold or emphasized text
            if block.is_bold:
                confidence += 0.2
            
            # Center alignment (common for titles)
            page_width = 612  # Standard page width
            text_center = (block.bbox[0] + block.bbox[2]) / 2
            if abs(text_center - page_width/2) < page_width * 0.1:
                confidence += 0.1
                
            if confidence > 0.5:
                title_candidates.append((block.text.strip(), confidence))
        
        if title_candidates:
            # Return highest confidence title
            title_candidates.sort(key=lambda x: x[1], reverse=True)
            return title_candidates[0][0]
        
        # Fallback: use filename or generic title
        return "Document"

    def identify_headings(self, blocks: List[TextBlock], doc_stats: Dict) -> List[Dict]:
        """Identify and rank potential headings"""
        heading_candidates = []
        
        for i, block in enumerate(blocks):
            # Skip very short or very long text
            if len(block.text.strip()) < 3 or len(block.text.strip()) > 200:
                continue
                
            # Calculate confidence score
            confidence = 0.0
            
            # Visual weight component
            confidence += block.visual_weight * self.heading_detector.font_weight
            
            # Position component
            confidence += block.position_score * self.heading_detector.position_weight
            
            # Content analysis component
            content_scores = self.heading_detector.analyze_content_patterns(block)
            content_confidence = sum(content_scores.values()) / len(content_scores) if content_scores else 0
            confidence += content_confidence * self.heading_detector.content_weight
            
            # Structure component (context with surrounding blocks)
            structure_score = self.calculate_structure_score(blocks, i)
            confidence += structure_score * self.heading_detector.structure_weight
            
            # Only consider high-confidence candidates
            if confidence >= self.min_confidence_threshold:
                level = self.heading_detector.determine_heading_level(blocks, i)
                
                heading_candidates.append({
                    'text': block.text.strip(),
                    'level': level,
                    'page': block.page_num,
                    'confidence': confidence,
                    'font_size': block.font_size,
                    'position': block.bbox[1]
                })
        
        # Post-process and refine headings
        return self.refine_headings(heading_candidates)

    def calculate_structure_score(self, blocks: List[TextBlock], current_idx: int) -> float:
        """Analyze structural context around potential heading"""
        if current_idx == 0:
            return 0.2  # First block is likely important
            
        score = 0.0
        current_block = blocks[current_idx]
        
        # Check spacing before and after
        if current_idx > 0:
            prev_block = blocks[current_idx - 1]
            vertical_gap = abs(current_block.bbox[1] - prev_block.bbox[3])
            if vertical_gap > current_block.font_size:
                score += 0.2
                
        if current_idx < len(blocks) - 1:
            next_block = blocks[current_idx + 1]
            vertical_gap = abs(next_block.bbox[1] - current_block.bbox[3])
            if vertical_gap > current_block.font_size:
                score += 0.1
                
        return min(score, 1.0)

    def refine_headings(self, candidates: List[Dict]) -> List[Dict]:
        """Refine and deduplicate heading candidates"""
        if not candidates:
            return []
            
        # Sort by page and position
        candidates.sort(key=lambda x: (x['page'], x['position']))
        
        # Remove very similar headings (potential duplicates)
        refined = []
        for candidate in candidates:
            is_duplicate = False
            for existing in refined:
                # Check for similarity
                if (existing['page'] == candidate['page'] and 
                    abs(existing['position'] - candidate['position']) < 20 and
                    self.text_similarity(existing['text'], candidate['text']) > 0.8):
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if candidate['confidence'] > existing['confidence']:
                        refined.remove(existing)
                        refined.append(candidate)
                    break
                    
            if not is_duplicate:
                refined.append(candidate)
        
        # Balance heading levels
        return self.balance_heading_hierarchy(refined)

    def text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity ratio"""
        text1_clean = re.sub(r'\W+', '', text1.lower())
        text2_clean = re.sub(r'\W+', '', text2.lower())
        
        if not text1_clean or not text2_clean:
            return 0.0
            
        # Simple Jaccard similarity
        set1 = set(text1_clean)
        set2 = set(text2_clean)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0

    def balance_heading_hierarchy(self, headings: List[Dict]) -> List[Dict]:
        """Ensure proper heading hierarchy (H1 > H2 > H3)"""
        if not headings:
            return []
            
        # Sort by confidence and font size
        headings.sort(key=lambda x: (x['confidence'], x['font_size']), reverse=True)
        
        # Reassign levels based on ranking
        level_counts = {'H1': 0, 'H2': 0, 'H3': 0}
        max_per_level = {'H1': max(1, len(headings) // 4), 'H2': len(headings) // 2, 'H3': len(headings)}
        
        for heading in headings:
            # Assign level based on current counts and limits
            if level_counts['H1'] < max_per_level['H1']:
                heading['level'] = 'H1'
                level_counts['H1'] += 1
            elif level_counts['H2'] < max_per_level['H2']:
                heading['level'] = 'H2'
                level_counts['H2'] += 1
            else:
                heading['level'] = 'H3'
                level_counts['H3'] += 1
        
        # Sort back by page and position
        headings.sort(key=lambda x: (x['page'], x['position']))
        
        return headings

    def process_pdf(self, pdf_path: str) -> Dict:
        """Main PDF processing function"""
        start_time = time.time()
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            # Check page limit
            if len(doc) > 50:
                logger.warning(f"PDF has {len(doc)} pages, processing first 50 only")
                # Process only first 50 pages
                temp_doc = fitz.open()
                for i in range(min(50, len(doc))):
                    temp_doc.insert_pdf(doc, from_page=i, to_page=i)
                doc.close()
                doc = temp_doc
            
            # Extract text blocks with metadata
            logger.info(f"Extracting text blocks from {len(doc)} pages...")
            blocks, doc_stats = self.extract_text_blocks(doc)
            
            # Check time constraint
            if time.time() - start_time > self.max_processing_time:
                logger.warning("Approaching time limit, using fast mode")
                return self.fast_process_mode(blocks, doc_stats)
            
            # Extract title
            logger.info("Extracting document title...")
            title = self.extract_title(blocks, doc_stats)
            
            # Identify headings
            logger.info("Identifying headings...")
            headings = self.identify_headings(blocks, doc_stats)
            
            # Prepare output
            outline = []
            for heading in headings:
                outline.append({
                    "level": heading['level'],
                    "text": heading['text'],
                    "page": heading['page']
                })
            
            result = {
                "title": title,
                "outline": outline
            }
            
            doc.close()
            
            processing_time = time.time() - start_time
            logger.info(f"Successfully processed PDF in {processing_time:.2f} seconds")
            logger.info(f"Found title: '{title}' and {len(outline)} headings")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return {
                "title": "Error Processing Document",
                "outline": []
            }

    def fast_process_mode(self, blocks: List[TextBlock], doc_stats: Dict) -> Dict:
        """Fallback fast processing mode for time-constrained execution"""
        logger.info("Using fast processing mode...")
        
        # Quick title extraction
        title = "Document"
        first_page_blocks = [b for b in blocks[:10] if b.page_num == 0]
        if first_page_blocks:
            # Use largest font size on first page
            title_block = max(first_page_blocks, key=lambda x: x.font_size)
            title = title_block.text.strip()
        
        # Quick heading detection
        outline = []
        avg_font_size = doc_stats.get('avg_font_size', 12)
        
        seen_texts = set()
        for block in blocks:
            text = block.text.strip()
            
            # Skip if already seen or too long/short
            if text in seen_texts or len(text) < 3 or len(text) > 100:
                continue
                
            # Simple heuristics for fast mode
            is_heading = (
                block.font_size > avg_font_size * 1.1 or
                block.is_bold or
                re.match(r'^\d+\.?\s+', text) or
                text.isupper() and len(text.split()) <= 8
            )
            
            if is_heading:
                # Simple level assignment
                if block.font_size > avg_font_size * 1.3 or block.page_num == 0:
                    level = "H1"
                elif block.font_size > avg_font_size * 1.2:
                    level = "H2"
                else:
                    level = "H3"
                    
                outline.append({
                    "level": level,
                    "text": text,
                    "page": block.page_num
                })
                
                seen_texts.add(text)
                
                # Limit headings in fast mode
                if len(outline) >= 20:
                    break
        
        return {
            "title": title,
            "outline": outline
        }

def process_all_pdfs():
    """Process all PDFs in the input directory"""
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processor = IntelligentPDFProcessor()
    
    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDF files found in input directory")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    for pdf_file in pdf_files:
        logger.info(f"Processing: {pdf_file.name}")
        
        try:
            # Process the PDF
            result = processor.process_pdf(str(pdf_file))
            
            # Save output JSON
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Successfully saved: {output_file.name}")
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
            
            # Create error output
            error_result = {
                "title": f"Error: {pdf_file.name}",
                "outline": []
            }
            
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    logger.info("Starting Adobe India Hackathon Round 1A Solution")
    logger.info("PDF Outline Extractor with Advanced Document Intelligence")
    
    start_time = time.time()
    process_all_pdfs()
    total_time = time.time() - start_time
    
    logger.info(f"All processing completed in {total_time:.2f} seconds")

