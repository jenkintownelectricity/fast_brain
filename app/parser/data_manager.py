"""
Data Extraction Manager - LLM-powered extraction and storage
"""

import os
import re
import json
import hashlib
import tempfile
import base64
from datetime import datetime
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, asdict, field

import httpx
from werkzeug.utils import secure_filename

from .universal_parser import UniversalParser, ALLOWED_EXTENSIONS

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
TEMP_DIR = tempfile.gettempdir()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractedData:
    """A single piece of extracted data."""
    id: str
    skill_id: str
    content_type: str
    user_input: str
    assistant_response: str
    raw_content: str
    source_filename: str
    source_type: str
    category: str
    tags: List[str] = field(default_factory=list)
    importance_score: float = 50.0
    confidence: float = 0.8
    tokens: int = 0
    is_approved: bool = False
    is_archived: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA EXTRACTION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class DataExtractionManager:
    """
    Manages the full pipeline:
    1. Parse files
    2. Extract training data with LLM
    3. Store in database
    4. DELETE original files
    """
    
    def __init__(self):
        self.parser = UniversalParser()
        self.http_client = None
    
    async def _get_client(self):
        if not self.http_client:
            self.http_client = httpx.AsyncClient(timeout=120.0)
        return self.http_client
    
    async def close(self):
        if self.http_client:
            await self.http_client.aclose()
    
    async def process_files(
        self,
        files: List[Tuple[str, str, bytes]],
        skill_id: str,
        skill_context: Dict,
        auto_delete: bool = True
    ) -> Dict:
        """
        Process uploaded files:
        1. Save temporarily
        2. Parse each file
        3. Extract training data
        4. Store in database
        5. Delete temp files
        """
        results = {
            'files_processed': 0,
            'files_failed': 0,
            'items_extracted': 0,
            'items_stored': 0,
            'errors': [],
            'extracted_data': []
        }
        
        for filename, content_type, data in files:
            temp_path = None
            
            try:
                # Save to temp file
                safe_name = secure_filename(filename)
                temp_path = os.path.join(TEMP_DIR, f"{hashlib.md5(f'{filename}{datetime.now()}'.encode()).hexdigest()[:12]}_{safe_name}")
                
                with open(temp_path, 'wb') as f:
                    f.write(data)
                
                # Parse the file
                parse_result = self.parser.parse(temp_path, filename)
                
                if parse_result['errors']:
                    results['errors'].extend([f"{filename}: {e}" for e in parse_result['errors']])
                
                if not parse_result['text'] and not parse_result.get('metadata', {}).get('needs_ocr') and not parse_result.get('metadata', {}).get('needs_transcription'):
                    results['files_failed'] += 1
                    continue
                
                # Handle OCR for images
                if parse_result.get('metadata', {}).get('needs_ocr'):
                    parse_result = await self._ocr_image(temp_path, filename)
                
                # Handle transcription for audio/video
                if parse_result.get('metadata', {}).get('needs_transcription'):
                    parse_result = await self._transcribe_audio(temp_path, filename)
                
                if not parse_result['text']:
                    results['files_failed'] += 1
                    continue
                
                # Extract training data using LLM
                extracted = await self._extract_training_data(
                    text=parse_result['text'],
                    skill_context=skill_context,
                    source_file=filename,
                    source_type=parse_result.get('source_type', 'unknown')
                )
                
                # Create ExtractedData objects
                for item in extracted:
                    data_item = ExtractedData(
                        id=hashlib.md5(f"{skill_id}_{filename}_{item['user_input'][:50]}_{datetime.now()}".encode()).hexdigest()[:16],
                        skill_id=skill_id,
                        content_type=item.get('type', 'qa_pair'),
                        user_input=item['user_input'],
                        assistant_response=item['assistant_response'],
                        raw_content=item.get('raw', '')[:500],
                        source_filename=filename,
                        source_type=parse_result.get('source_type', 'unknown'),
                        category=item.get('category', 'general'),
                        tags=item.get('tags', []),
                        importance_score=item.get('importance', 50),
                        confidence=item.get('confidence', 0.8),
                        tokens=len(item['user_input'].split()) + len(item['assistant_response'].split())
                    )
                    results['extracted_data'].append(data_item.to_dict())
                
                results['files_processed'] += 1
                results['items_extracted'] += len(extracted)
                
            except Exception as e:
                results['errors'].append(f"{filename}: {str(e)}")
                results['files_failed'] += 1
            
            finally:
                # DELETE temp file immediately
                if auto_delete and temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
        
        # Score importance of all items
        if results['extracted_data']:
            results['extracted_data'] = await self._score_importance(results['extracted_data'], skill_context)
        
        return results
    
    async def _ocr_image(self, file_path: str, filename: str) -> Dict:
        """Extract text from images using vision model."""
        try:
            with open(file_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            ext = filename.rsplit('.', 1)[-1].lower()
            media_types = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'gif': 'image/gif', 'webp': 'image/webp'}
            media_type = media_types.get(ext, 'image/png')
            
            client = await self._get_client()
            response = await client.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.2-90b-vision-preview",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract ALL text from this image. Include everything you can read."},
                            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_data}"}}
                        ]
                    }],
                    "max_tokens": 4000
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result['choices'][0]['message']['content']
                return {'text': text, 'pages': 1, 'errors': [], 'metadata': {'ocr': True}, 'source_type': 'image'}
            else:
                return {'text': '', 'pages': 0, 'errors': [f'Vision API error: {response.status_code}'], 'metadata': {}}
                
        except Exception as e:
            return {'text': '', 'pages': 0, 'errors': [str(e)], 'metadata': {}}
    
    async def _transcribe_audio(self, file_path: str, filename: str) -> Dict:
        """Transcribe audio using Whisper API."""
        try:
            with open(file_path, 'rb') as f:
                audio_data = f.read()
            
            client = await self._get_client()
            
            # Use OpenAI Whisper API
            files = {"file": (filename, audio_data)}
            data = {"model": "whisper-1", "response_format": "text"}
            
            response = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                text = response.text
                return {'text': text, 'pages': 1, 'errors': [], 'metadata': {'transcribed': True}, 'source_type': 'audio'}
            else:
                return {'text': '', 'pages': 0, 'errors': [f'Whisper API error: {response.status_code}'], 'metadata': {}}
                
        except Exception as e:
            return {'text': '', 'pages': 0, 'errors': [str(e)], 'metadata': {}}
    
    async def _extract_training_data(
        self,
        text: str,
        skill_context: Dict,
        source_file: str,
        source_type: str
    ) -> List[Dict]:
        """Use LLM to extract Q&A pairs from text."""
        
        chunks = self._split_text(text, max_chars=10000)
        all_items = []
        
        for chunk in chunks:
            prompt = f"""You are an expert at extracting training data for AI voice assistants.

SKILL: {skill_context.get('name', 'AI Assistant')}
DESCRIPTION: {skill_context.get('system_prompt', '')[:400]}
SOURCE TYPE: {source_type}
SOURCE FILE: {source_file}

DOCUMENT CONTENT:
{chunk}

TASK: Extract valuable question-answer pairs for training this AI assistant.

For each item, identify:
1. user_input: A realistic customer question
2. assistant_response: An ideal, detailed response (100+ words when appropriate)
3. category: greeting, pricing, technical, scheduling, objection, closing, faq, procedure, policy, general
4. type: qa_pair, faq, procedure, fact, conversation
5. confidence: 0.0-1.0
6. tags: relevant keywords

GUIDELINES:
- Extract REAL information from the document
- Make responses detailed and natural
- Focus on what's relevant to the skill's purpose

Return JSON array:
[{{"user_input": "...", "assistant_response": "...", "category": "...", "type": "...", "confidence": 0.9, "tags": ["tag1"]}}]

Extract up to 15 high-quality pairs. Return ONLY valid JSON."""

            try:
                client = await self._get_client()
                response = await client.post(
                    GROQ_API_URL,
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 4000
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    items = self._parse_json_response(content)
                    
                    for item in items:
                        if item.get('user_input') and item.get('assistant_response'):
                            if len(item['user_input']) > 5 and len(item['assistant_response']) > 10:
                                all_items.append(item)
                
            except Exception as e:
                print(f"Extraction error: {e}")
        
        return all_items
    
    async def _score_importance(self, items: List[Dict], skill_context: Dict) -> List[Dict]:
        """Score and sort items by importance."""
        
        batch_size = 20
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            
            items_summary = json.dumps([{
                'id': item['id'],
                'user': item['user_input'][:100],
                'category': item['category']
            } for item in batch])
            
            prompt = f"""Score these training examples for "{skill_context.get('name', 'AI')}" from 0-100.

Consider: Relevance (40%), Quality (30%), Uniqueness (20%), Clarity (10%)

Items: {items_summary}

Return JSON mapping ID to score: {{"id1": 85, "id2": 72}}"""

            try:
                client = await self._get_client()
                response = await client.post(
                    GROQ_API_URL,
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 500
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    scores = self._parse_json_response(result['choices'][0]['message']['content'])
                    
                    if isinstance(scores, dict):
                        for item in batch:
                            item['importance_score'] = scores.get(item['id'], 50)
                
            except Exception as e:
                print(f"Scoring error: {e}")
        
        items.sort(key=lambda x: x.get('importance_score', 0), reverse=True)
        return items
    
    def _split_text(self, text: str, max_chars: int = 10000) -> List[str]:
        """Split text into chunks."""
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        paragraphs = text.split('\n\n')
        current = ""
        
        for para in paragraphs:
            if len(current) + len(para) < max_chars:
                current += para + "\n\n"
            else:
                if current:
                    chunks.append(current.strip())
                current = para + "\n\n"
        
        if current:
            chunks.append(current.strip())
        
        return chunks or [text[:max_chars]]
    
    def _parse_json_response(self, content: str) -> Any:
        """Parse JSON from LLM response."""
        content = content.strip()
        
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0]
        elif '```' in content:
            content = content.split('```')[1].split('```')[0]
        
        try:
            return json.loads(content)
        except:
            match = re.search(r'[\[{][\s\S]*[\]}]', content)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
        return []
