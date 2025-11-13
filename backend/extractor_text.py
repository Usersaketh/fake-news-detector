"""
Text extraction module for URL fetching, HTML parsing, and OCR from images.
Supports three input methods: direct text, URL, and image.
"""

import os
import requests
from typing import Dict, Optional, Tuple
from bs4 import BeautifulSoup
from readability import Document
from requests_html import HTMLSession
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

# Try to import pytesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not available. Image OCR will be limited.")


class TextExtractor:
    """Extract text from various input sources."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def extract_from_text(self, headline: str, body: str) -> Dict:
        """
        Extract from direct text input.
        
        Args:
            headline: Article headline
            body: Article body text
            
        Returns:
            Dict with headline, body, and metadata
        """
        return {
            "headline": headline.strip(),
            "body": body.strip(),
            "full_text": f"{headline}\n\n{body}",
            "metadata": {
                "source": "text_input",
                "extraction_method": "direct"
            }
        }
    
    def extract_from_url(self, url: str) -> Dict:
        """
        Extract text from URL using multiple fallback methods.
        
        Args:
            url: URL to fetch
            
        Returns:
            Dict with headline, body, full_text, and metadata
        """
        logger.info(f"Fetching URL: {url}")
        
        try:
            # Try standard requests first
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            html = response.text
            
            # Try readability-lxml first (best for articles)
            try:
                doc = Document(html)
                headline = doc.title()
                body_html = doc.summary()
                
                # Parse body HTML to extract text
                soup = BeautifulSoup(body_html, 'lxml')
                body = soup.get_text(separator='\n', strip=True)
                
                # Also extract metadata
                soup_full = BeautifulSoup(html, 'lxml')
                metadata = self._extract_metadata(soup_full, url)
                
                return {
                    "headline": headline,
                    "body": body,
                    "full_text": f"{headline}\n\n{body}",
                    "metadata": metadata
                }
            except Exception as e:
                logger.warning(f"Readability failed: {e}, trying BeautifulSoup")
                
                # Fallback to BeautifulSoup
                soup = BeautifulSoup(html, 'lxml')
                
                # Try to find headline
                headline = self._find_headline(soup)
                
                # Try to find article body
                body = self._find_article_body(soup)
                
                metadata = self._extract_metadata(soup, url)
                
                return {
                    "headline": headline,
                    "body": body,
                    "full_text": f"{headline}\n\n{body}",
                    "metadata": metadata
                }
                
        except Exception as e:
            logger.error(f"Standard fetch failed: {e}, trying requests-html")
            
            # Try requests-html for JS-heavy sites
            try:
                return self._extract_with_js_rendering(url)
            except Exception as e2:
                logger.error(f"JS rendering also failed: {e2}")
                raise Exception(f"Failed to extract content from URL: {str(e)}")
    
    def _extract_with_js_rendering(self, url: str) -> Dict:
        """
        Extract content from JS-heavy sites using requests-html.
        
        Args:
            url: URL to fetch
            
        Returns:
            Extracted content dict
        """
        session = HTMLSession()
        response = session.get(url, timeout=15)
        
        # Render JavaScript
        response.html.render(timeout=20, sleep=2)
        
        # Parse rendered HTML
        soup = BeautifulSoup(response.html.html, 'lxml')
        
        headline = self._find_headline(soup)
        body = self._find_article_body(soup)
        metadata = self._extract_metadata(soup, url)
        metadata['extraction_method'] = 'js_rendered'
        
        return {
            "headline": headline,
            "body": body,
            "full_text": f"{headline}\n\n{body}",
            "metadata": metadata
        }
    
    def _find_headline(self, soup: BeautifulSoup) -> str:
        """Extract headline from HTML soup."""
        # Try common headline tags
        for selector in ['h1', 'h2', '.headline', '.article-title', 'title']:
            element = soup.select_one(selector)
            if element and element.get_text(strip=True):
                return element.get_text(strip=True)
        
        # Fallback to page title
        title = soup.find('title')
        if title:
            return title.get_text(strip=True)
        
        return "No headline found"
    
    def _find_article_body(self, soup: BeautifulSoup) -> str:
        """Extract article body from HTML soup."""
        # Try common article body selectors
        selectors = [
            'article',
            '.article-body',
            '.article-content',
            '.post-content',
            '.entry-content',
            'main'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                # Remove script and style tags
                for tag in element(['script', 'style', 'nav', 'header', 'footer']):
                    tag.decompose()
                
                text = element.get_text(separator='\n', strip=True)
                if len(text) > 100:  # Ensure substantial content
                    return text
        
        # Fallback: find all paragraphs
        paragraphs = soup.find_all('p')
        text = '\n\n'.join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50)
        
        return text if text else "No body content found"
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract metadata from HTML."""
        metadata = {
            "source": url,
            "extraction_method": "html_parse"
        }
        
        # Try to find author
        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta:
            metadata['author'] = author_meta.get('content', '')
        
        # Try to find date
        date_meta = soup.find('meta', attrs={'property': 'article:published_time'})
        if not date_meta:
            date_meta = soup.find('meta', attrs={'name': 'date'})
        if date_meta:
            metadata['date'] = date_meta.get('content', '')
        
        # Try to find description
        desc_meta = soup.find('meta', attrs={'name': 'description'})
        if not desc_meta:
            desc_meta = soup.find('meta', attrs={'property': 'og:description'})
        if desc_meta:
            metadata['description'] = desc_meta.get('content', '')
        
        return metadata
    
    def extract_from_image(self, image_file) -> Dict:
        """
        Extract text from image using OCR.
        
        Args:
            image_file: Image file object or bytes
            
        Returns:
            Dict with extracted text and metadata
        """
        if not TESSERACT_AVAILABLE:
            return {
                "headline": "OCR not available",
                "body": "Please install pytesseract and Tesseract OCR to extract text from images.",
                "full_text": "",
                "metadata": {
                    "source": "image_input",
                    "extraction_method": "ocr_unavailable"
                }
            }
        
        try:
            # Load image
            if isinstance(image_file, bytes):
                image = Image.open(io.BytesIO(image_file))
            else:
                image = Image.open(image_file)
            
            # Perform OCR
            logger.info("Performing OCR on image")
            text = pytesseract.image_to_string(image)
            
            # Clean and structure the text
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            if not lines:
                return {
                    "headline": "No text detected in image",
                    "body": "",
                    "full_text": "",
                    "metadata": {
                        "source": "image_input",
                        "extraction_method": "ocr",
                        "status": "no_text_found"
                    }
                }
            
            # Try to identify headline (usually first substantial line)
            headline = lines[0] if lines else "No headline detected"
            
            # Remaining lines as body
            body = '\n'.join(lines[1:]) if len(lines) > 1 else ""
            
            full_text = '\n'.join(lines)
            
            return {
                "headline": headline,
                "body": body,
                "full_text": full_text,
                "metadata": {
                    "source": "image_input",
                    "extraction_method": "ocr",
                    "lines_extracted": len(lines)
                }
            }
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {
                "headline": "Image extraction failed",
                "body": f"Error: {str(e)}",
                "full_text": "",
                "metadata": {
                    "source": "image_input",
                    "extraction_method": "ocr",
                    "status": "error",
                    "error": str(e)
                }
            }


def extract_text(input_type: str, **kwargs) -> Dict:
    """
    Main extraction function that routes to appropriate extractor.
    
    Args:
        input_type: One of 'text', 'url', or 'image'
        **kwargs: Arguments specific to input type
            - For 'text': headline, body
            - For 'url': url
            - For 'image': image_file
            
    Returns:
        Dict with extracted content
    """
    extractor = TextExtractor()
    
    if input_type == 'text':
        return extractor.extract_from_text(
            kwargs.get('headline', ''),
            kwargs.get('body', '')
        )
    elif input_type == 'url':
        return extractor.extract_from_url(kwargs['url'])
    elif input_type == 'image':
        return extractor.extract_from_image(kwargs['image_file'])
    else:
        raise ValueError(f"Unknown input type: {input_type}")
