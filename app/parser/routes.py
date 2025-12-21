"""
Parser API Routes
"""

import os
import hashlib
from datetime import datetime
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from .universal_parser import FILE_TYPES, ALLOWED_EXTENSIONS
from .data_manager import DataExtractionManager

# ═══════════════════════════════════════════════════════════════════════════════
# BLUEPRINT
# ═══════════════════════════════════════════════════════════════════════════════

parser_bp = Blueprint('parser', __name__)

# In-memory storage (use Redis/DB in production)
extraction_jobs = {}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_db_connection():
    """Get database connection."""
    import psycopg2
    return psycopg2.connect(os.environ.get('DATABASE_URL'))


def get_skill_context(skill_id: str) -> dict:
    """Get skill information."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, system_prompt FROM fb_skills WHERE id = %s", [skill_id])
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if row:
            return {'id': row[0], 'name': row[1], 'system_prompt': row[2] or ''}
    except:
        pass
    
    return {'id': skill_id, 'name': 'AI Assistant', 'system_prompt': ''}


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@parser_bp.route('/api/parser/supported-types', methods=['GET'])
def get_supported_types():
    """Get list of all supported file types."""
    types_by_category = {}
    
    for ext, info in FILE_TYPES.items():
        category = info['category']
        if category not in types_by_category:
            types_by_category[category] = []
        types_by_category[category].append({
            'extension': ext,
            'name': info['name']
        })
    
    return jsonify({
        'success': True,
        'total_types': len(FILE_TYPES),
        'types_by_category': types_by_category,
        'all_extensions': list(ALLOWED_EXTENSIONS)
    })


@parser_bp.route('/api/parser/upload', methods=['POST'])
async def upload_and_extract():
    """Upload files → Extract data → Delete files → Return structured data."""
    if 'files[]' not in request.files and 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No files provided'}), 400
    
    skill_id = request.form.get('skill_id')
    if not skill_id:
        return jsonify({'success': False, 'error': 'skill_id required'}), 400
    
    skill_context = get_skill_context(skill_id)
    
    # Collect files
    files_data = []
    uploaded_files = request.files.getlist('files[]') or [request.files.get('file')]
    
    for file in uploaded_files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
            
            if ext not in ALLOWED_EXTENSIONS:
                continue
            
            files_data.append((filename, file.content_type, file.read()))
    
    if not files_data:
        return jsonify({'success': False, 'error': 'No valid files'}), 400
    
    # Process files
    manager = DataExtractionManager()
    
    try:
        results = await manager.process_files(
            files=files_data,
            skill_id=skill_id,
            skill_context=skill_context,
            auto_delete=True
        )
        
        # Store in database
        if results['extracted_data']:
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                for item in results['extracted_data']:
                    cursor.execute("""
                        INSERT INTO fb_extracted_data 
                        (id, skill_id, content_type, user_input, assistant_response, raw_content,
                         source_filename, source_type, category, tags, importance_score, confidence,
                         tokens, is_approved, is_archived, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                    """, (
                        item['id'], item['skill_id'], item['content_type'],
                        item['user_input'], item['assistant_response'], item['raw_content'],
                        item['source_filename'], item['source_type'], item['category'],
                        item['tags'], item['importance_score'], item['confidence'],
                        item['tokens'], False, False, '{}'
                    ))
                
                conn.commit()
                results['items_stored'] = cursor.rowcount
                cursor.close()
                conn.close()
            except Exception as e:
                results['errors'].append(f"DB error: {str(e)}")
        
    finally:
        await manager.close()
    
    # Create job record
    job_id = hashlib.md5(f"{skill_id}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
    extraction_jobs[job_id] = {
        'skill_id': skill_id,
        'status': 'complete',
        'results': results,
        'created_at': datetime.now().isoformat()
    }
    
    return jsonify({
        'success': True,
        'job_id': job_id,
        'files_processed': results['files_processed'],
        'files_failed': results['files_failed'],
        'items_extracted': results['items_extracted'],
        'items_stored': results.get('items_stored', 0),
        'errors': results['errors'],
        'data': results['extracted_data']
    })


@parser_bp.route('/api/parser/data/<skill_id>', methods=['GET'])
def get_extracted_data(skill_id):
    """Get all extracted data for a skill with filtering/sorting."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build query
        query = "SELECT * FROM fb_extracted_data WHERE skill_id = %s"
        params = [skill_id]
        
        # Filters
        if request.args.get('category'):
            query += " AND category = %s"
            params.append(request.args.get('category'))
        
        if request.args.get('tag'):
            query += " AND %s = ANY(tags)"
            params.append(request.args.get('tag'))
        
        if request.args.get('min_importance'):
            query += " AND importance_score >= %s"
            params.append(float(request.args.get('min_importance')))
        
        if request.args.get('search'):
            query += " AND (user_input ILIKE %s OR assistant_response ILIKE %s)"
            search = f"%{request.args.get('search')}%"
            params.extend([search, search])
        
        if request.args.get('approved') == 'true':
            query += " AND is_approved = true"
        elif request.args.get('approved') == 'false':
            query += " AND is_approved = false"
        
        if request.args.get('archived') == 'true':
            query += " AND is_archived = true"
        else:
            query += " AND is_archived = false"
        
        # Sorting
        sort_field = request.args.get('sort', 'importance_score')
        sort_order = 'DESC' if request.args.get('order', 'desc') == 'desc' else 'ASC'
        
        # Validate sort field
        valid_sorts = ['importance_score', 'created_at', 'tokens', 'category']
        if sort_field not in valid_sorts:
            sort_field = 'importance_score'
        
        query += f" ORDER BY {sort_field} {sort_order}"
        
        # Pagination
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        offset = (page - 1) * per_page
        query += f" LIMIT {per_page} OFFSET {offset}"
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        # Get total count
        count_query = "SELECT COUNT(*) FROM fb_extracted_data WHERE skill_id = %s AND is_archived = false"
        cursor.execute(count_query, [skill_id])
        total = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': (total + per_page - 1) // per_page,
            'data': rows
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@parser_bp.route('/api/parser/data/<skill_id>/bulk', methods=['POST'])
def bulk_update_data(skill_id):
    """Bulk update extracted data items."""
    data = request.json
    action = data.get('action')
    item_ids = data.get('item_ids', [])
    value = data.get('value')
    
    if not item_ids:
        return jsonify({'success': False, 'error': 'No items specified'}), 400
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if action == 'approve':
            cursor.execute(
                "UPDATE fb_extracted_data SET is_approved = true WHERE id = ANY(%s) AND skill_id = %s",
                [item_ids, skill_id]
            )
        
        elif action == 'archive':
            cursor.execute(
                "UPDATE fb_extracted_data SET is_archived = true WHERE id = ANY(%s) AND skill_id = %s",
                [item_ids, skill_id]
            )
        
        elif action == 'delete':
            cursor.execute(
                "DELETE FROM fb_extracted_data WHERE id = ANY(%s) AND skill_id = %s",
                [item_ids, skill_id]
            )
        
        elif action == 'tag' and value:
            cursor.execute(
                "UPDATE fb_extracted_data SET tags = array_append(tags, %s) WHERE id = ANY(%s) AND skill_id = %s AND NOT %s = ANY(tags)",
                [value, item_ids, skill_id, value]
            )
        
        elif action == 'categorize' and value:
            cursor.execute(
                "UPDATE fb_extracted_data SET category = %s WHERE id = ANY(%s) AND skill_id = %s",
                [value, item_ids, skill_id]
            )
        
        elif action == 'move_to_training':
            # Move approved items to training examples
            cursor.execute("""
                INSERT INTO fb_training_examples (skill_id, user_input, assistant_response, category, quality_score, tokens, source)
                SELECT skill_id, user_input, assistant_response, category, 
                       GREATEST(1, LEAST(5, ROUND(importance_score / 20))), tokens, 
                       'extracted:' || source_filename
                FROM fb_extracted_data
                WHERE id = ANY(%s) AND skill_id = %s
            """, [item_ids, skill_id])
            
            # Mark as used
            cursor.execute(
                "UPDATE fb_extracted_data SET is_archived = true WHERE id = ANY(%s)",
                [item_ids]
            )
        
        conn.commit()
        affected = cursor.rowcount
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'affected': affected})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@parser_bp.route('/api/parser/data/<skill_id>/stats', methods=['GET'])
def get_data_stats(skill_id):
    """Get statistics about extracted data."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE is_approved) as approved,
                COUNT(*) FILTER (WHERE is_archived) as archived,
                AVG(importance_score) as avg_importance,
                SUM(tokens) as total_tokens,
                COUNT(DISTINCT category) as categories,
                COUNT(DISTINCT source_filename) as sources
            FROM fb_extracted_data
            WHERE skill_id = %s AND is_archived = false
        """, [skill_id])
        
        row = cursor.fetchone()
        
        # Category breakdown
        cursor.execute("""
            SELECT category, COUNT(*) as count
            FROM fb_extracted_data
            WHERE skill_id = %s AND is_archived = false
            GROUP BY category
        """, [skill_id])
        categories = {r[0]: r[1] for r in cursor.fetchall()}
        
        # Tag breakdown
        cursor.execute("""
            SELECT UNNEST(tags) as tag, COUNT(*) as count
            FROM fb_extracted_data
            WHERE skill_id = %s AND is_archived = false
            GROUP BY tag
            ORDER BY count DESC
            LIMIT 20
        """, [skill_id])
        tags = {r[0]: r[1] for r in cursor.fetchall()}
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'stats': {
                'total': row[0] or 0,
                'approved': row[1] or 0,
                'archived': row[2] or 0,
                'avg_importance': round(row[3] or 0, 1),
                'total_tokens': row[4] or 0,
                'categories_count': row[5] or 0,
                'sources_count': row[6] or 0
            },
            'categories': categories,
            'tags': tags
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
