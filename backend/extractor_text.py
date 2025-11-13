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
import base64
from openai import OpenAI

logger = logging.getLogger(__name__)


class TextExtractor:
    """Extract text from various input sources."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        # Initialize OpenAI client for vision-based OCR
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if self.openrouter_api_key:
            self.vision_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key
            )
        else:
            self.vision_client = None
    
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
        Extract text from image using Qwen vision model via OpenRouter.
        
        Args:
            image_file: Image file object or bytes
            
        Returns:
            Dict with extracted text and metadata
        """
        if not self.vision_client:
            error_msg = (
                "OpenRouter API key not found.\n\n"
                "Please add OPENROUTER_API_KEY to your .env file:\n"
                "Get your API key from https://openrouter.ai/\n\n"
                "This feature uses Qwen vision model for intelligent text extraction."
            )
            logger.error("OpenRouter API key not configured")
            return {
                "headline": "❌ Vision Model Not Available",
                "body": error_msg,
                "full_text": "",
                "metadata": {
                    "source": "image_input",
                    "extraction_method": "vision_unavailable",
                    "error": "api_key_missing"
                }
            }
        
        try:
            # Load and convert image to base64
            logger.info("Processing image with Qwen vision model")
            
            if isinstance(image_file, bytes):
                image_bytes = image_file
            else:
                # Read from file-like object
                image_file.seek(0)
                image_bytes = image_file.read()
            
            # Convert to base64 for API
            import base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            image_url = f"data:image/jpeg;base64,{base64_image}"
            
            # Call Qwen vision model
            completion = self.vision_client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/fake-news-detector",
                    "X-Title": "Fake News Detector",
                },
                model="qwen/qwen2.5-vl-32b-instruct:free",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Extract ALL text from this image. "
                                    "If this appears to be a news article, social media post, or meme with text, "
                                    "identify the HEADLINE (main title/claim) and BODY (supporting text/details). \n\n"
                                    "Format your response as:\n"
                                    "HEADLINE: [main title or claim]\n"
                                    "BODY: [all other text content]\n\n"
                                    "If there's only one piece of text, put it in HEADLINE and leave BODY empty."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse the response
            response_text = completion.choices[0].message.content.strip()
            logger.info(f"Vision model response (first 200 chars): {response_text[:200]}")
            
            # Parse HEADLINE and BODY from response
            headline = ""
            body = ""
            
            if "HEADLINE:" in response_text and "BODY:" in response_text:
                parts = response_text.split("BODY:", 1)
                headline_part = parts[0].replace("HEADLINE:", "").strip()
                body_part = parts[1].strip() if len(parts) > 1 else ""
                
                headline = headline_part
                body = body_part
            else:
                # Fallback: use first line as headline, rest as body
                lines = [line.strip() for line in response_text.split('\n') if line.strip()]
                headline = lines[0] if lines else "No text detected"
                body = '\n'.join(lines[1:]) if len(lines) > 1 else ""
            
            full_text = f"{headline}\n\n{body}" if body else headline
            
            if not headline and not body:
                return {
                    "headline": "No text detected in image",
                    "body": "",
                    "full_text": "",
                    "metadata": {
                        "source": "image_input",
                        "extraction_method": "vision_model",
                        "model": "qwen/qwen2.5-vl-32b-instruct",
                        "status": "no_text_found"
                    }
                }
            
            return {
                "headline": headline,
                "body": body,
                "full_text": full_text,
                "metadata": {
                    "source": "image_input",
                    "extraction_method": "vision_model",
                    "model": "qwen/qwen2.5-vl-32b-instruct",
                    "status": "success"
                }
            }
            
        except Exception as e:
            logger.error(f"Vision model extraction failed: {e}")
            return {
                "headline": "Image extraction failed",
                "body": f"Error: {str(e)}",
                "full_text": "",
                "metadata": {
                    "source": "image_input",
                    "extraction_method": "vision_model",
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
