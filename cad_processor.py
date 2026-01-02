# cad_processor.py - Optimized DXF to JSON converter for AI Training
"""
Converts DXF files into structured JSON for different AI training needs:
- SPATIAL: Geometry, bounding boxes, layers (for "The Eyes")
- QUANTITY: Linear feet, block counts, areas (for "The Estimator")
- SPECS: Text notes, dimensions, leaders (for "The Detailer")
- FULL: Complete data dump

Optimizations (per AI Auditor review):
1. Parse file ONCE, generate MANY outputs
2. Normalize all units to FEET for consistency
3. Robust error handling for corrupted geometry
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, BinaryIO, Tuple

try:
    import ezdxf
    from ezdxf import bbox
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False


class CADProcessor:
    """Process DXF files into AI-readable JSON formats (Optimized)."""

    # Unit conversion factors to FEET
    UNITS_TO_FEET = {
        0: 1.0,        # Unitless - assume feet
        1: 1/12.0,     # Inches to Feet
        2: 1.0,        # Feet to Feet
        3: 1/5280.0,   # Miles to Feet
        4: 0.00328084, # Millimeters to Feet
        5: 0.0328084,  # Centimeters to Feet
        6: 3.28084,    # Meters to Feet
    }

    UNITS_NAMES = {
        0: 'Unitless',
        1: 'Inches',
        2: 'Feet',
        3: 'Miles',
        4: 'Millimeters',
        5: 'Centimeters',
        6: 'Meters'
    }

    def __init__(self):
        if not EZDXF_AVAILABLE:
            raise ImportError("ezdxf library required. Install with: pip install ezdxf")

    def _get_conversion_factor(self, doc) -> float:
        """Returns factor to convert drawing units to FEET."""
        units = doc.header.get('$INSUNITS', 0)
        return self.UNITS_TO_FEET.get(units, 1.0)

    def _get_units_name(self, doc) -> str:
        """Returns human-readable units name."""
        units = doc.header.get('$INSUNITS', 0)
        return self.UNITS_NAMES.get(units, 'Unknown')

    def _parse_once(self, file_path: str = None, file_obj = None) -> Tuple:
        """Parse file ONCE - critical for efficiency."""
        if file_obj:
            import tempfile
            import os

            # Handle Flask FileStorage or similar objects
            filename = getattr(file_obj, 'filename', getattr(file_obj, 'name', 'uploaded.dxf'))
            if hasattr(filename, 'split'):
                filename = Path(filename).name if '/' in filename or '\\' in filename else filename

            # Get content from file object
            if hasattr(file_obj, 'stream'):
                # Flask FileStorage
                file_obj.stream.seek(0)
                content = file_obj.stream.read()
            elif hasattr(file_obj, 'read'):
                if hasattr(file_obj, 'seek'):
                    file_obj.seek(0)
                content = file_obj.read()
            else:
                raise ValueError("Unable to read file object")

            # Ensure bytes
            if isinstance(content, str):
                content = content.encode('utf-8')

            # Write to temp file - ezdxf.readfile works most reliably
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.dxf', delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                doc = ezdxf.readfile(tmp_path)
            finally:
                os.unlink(tmp_path)  # Clean up temp file
        elif file_path:
            doc = ezdxf.readfile(file_path)
            filename = Path(file_path).name
        else:
            raise ValueError("Must provide file_path or file_obj")

        return doc, doc.modelspace(), filename

    def get_preview(self, file_path: str = None, file_obj: BinaryIO = None) -> Dict:
        """
        Quick preview of DXF contents BEFORE full processing.
        Shows user what's in the file so they can choose output types.
        """
        try:
            doc, msp, filename = self._parse_once(file_path, file_obj)
        except Exception as e:
            return {"error": f"Failed to read DXF: {str(e)}"}

        # Count entities by type
        entity_counts = defaultdict(int)
        layers = set()
        text_samples = []
        block_names = set()

        for e in msp:
            entity_counts[e.dxftype()] += 1
            layers.add(e.dxf.layer)

            # Sample some text for preview
            if e.dxftype() == 'TEXT' and len(text_samples) < 5:
                text_samples.append(e.dxf.text[:50])
            elif e.dxftype() == 'MTEXT' and len(text_samples) < 5:
                text_samples.append(e.text[:50])

            # Collect block names
            if e.dxftype() == 'INSERT':
                block_names.add(e.dxf.name)

        return {
            'success': True,
            'filename': filename,
            'units': self._get_units_name(doc),
            'units_code': doc.header.get('$INSUNITS', 0),
            'conversion_to_feet': self._get_conversion_factor(doc),
            'layers': sorted(list(layers)),
            'layer_count': len(layers),
            'entity_counts': dict(entity_counts),
            'total_entities': sum(entity_counts.values()),
            'block_names': sorted(list(block_names)),
            'text_samples': text_samples,
            'recommendations': self._get_recommendations(entity_counts)
        }

    def _get_recommendations(self, entity_counts: Dict) -> List[str]:
        """Suggest which output types would be useful based on content."""
        recs = []

        has_geometry = entity_counts.get('LINE', 0) + entity_counts.get('LWPOLYLINE', 0) > 0
        has_text = entity_counts.get('TEXT', 0) + entity_counts.get('MTEXT', 0) > 0
        has_dims = entity_counts.get('DIMENSION', 0) > 0
        has_blocks = entity_counts.get('INSERT', 0) > 0

        if has_geometry:
            recs.append("SPATIAL - Good for layer analysis and spatial relationships")
            recs.append("QUANTITY - Good for linear footage takeoffs")

        if has_text or has_dims:
            recs.append("SPECS - Good for extracting notes and dimensions")

        if has_blocks:
            recs.append("QUANTITY - Will count blocks (drains, scuppers, etc.)")

        if not recs:
            recs.append("FULL - Recommended for complete data extraction")

        return recs

    def process(self, file_path: str = None, file_obj: BinaryIO = None,
                output_types: List[str] = None) -> Dict[str, Dict]:
        """
        Master processing function. Reads ONCE, exports MANY.

        Args:
            file_path: Path to DXF file
            file_obj: File-like object (for uploads)
            output_types: List of formats: ['spatial', 'quantity', 'specs', 'full']

        Returns:
            Dict with format names as keys and JSON data as values
        """
        if output_types is None:
            output_types = ['full']

        # 1. PARSE ONCE (Efficiency Fix from Auditor)
        try:
            doc, msp, filename = self._parse_once(file_path, file_obj)
        except Exception as e:
            return {"error": f"Failed to parse DXF: {str(e)}"}

        # 2. CALCULATE CONVERSION (Unit Normalization from Auditor)
        to_feet = self._get_conversion_factor(doc)
        meta = {
            "filename": filename,
            "units": self._get_units_name(doc),
            "units_code": doc.header.get('$INSUNITS', 0),
            "conversion_to_feet": to_feet,
            "layers": [l.dxf.name for l in doc.layers]
        }

        results = {}

        # 3. GENERATE REQUESTED FORMATS (pass msp, don't re-read)
        for output_type in output_types:
            otype = output_type.lower().strip()

            if otype == 'spatial':
                results['spatial'] = self._gen_spatial(msp, meta)
            elif otype == 'quantity':
                results['quantity'] = self._gen_quantity(msp, meta, to_feet)
            elif otype == 'specs':
                results['specs'] = self._gen_specs(msp, meta, to_feet)
            elif otype == 'full':
                results['full'] = self._gen_full(msp, meta, to_feet)

        return results

    def _gen_spatial(self, msp, meta: Dict) -> Dict:
        """
        LENS 1: THE EYES - Spatial Analysis
        Bounding boxes, layers, closed/open status for geometric reasoning.
        """
        data = {
            'format': 'SPATIAL',
            'purpose': 'Geometry analysis for spatial AI (The Eyes)',
            'meta': meta,
            'geometry': []
        }

        layers_found = set()

        for e in msp:
            layers_found.add(e.dxf.layer)

            if e.dxftype() in ['LWPOLYLINE', 'POLYLINE', 'LINE', 'CIRCLE', 'ARC']:
                try:
                    # Wrap in try/except - corrupted geometry won't crash (Auditor fix)
                    bounds = bbox.extents([e])
                    if bounds.has_data:
                        data['geometry'].append({
                            'type': e.dxftype(),
                            'layer': e.dxf.layer,
                            'bounds': {
                                'min_x': round(bounds.extmin.x, 4),
                                'min_y': round(bounds.extmin.y, 4),
                                'max_x': round(bounds.extmax.x, 4),
                                'max_y': round(bounds.extmax.y, 4)
                            },
                            'closed': getattr(e, 'closed', False)
                        })
                except Exception:
                    continue  # Skip bad geometry, don't crash

        data['layers_with_geometry'] = sorted(list(layers_found))
        data['geometry_count'] = len(data['geometry'])
        return data

    def _gen_quantity(self, msp, meta: Dict, to_feet: float) -> Dict:
        """
        LENS 2: THE ESTIMATOR - Quantity Takeoff
        Pre-calculated linear feet (normalized to FEET) and block counts.
        """
        linear_by_layer = defaultdict(float)
        block_counts = defaultdict(int)
        entity_counts = defaultdict(lambda: defaultdict(int))

        for e in msp:
            layer = e.dxf.layer
            entity_counts[layer][e.dxftype()] += 1

            # Lines - simple distance calculation
            if e.dxftype() == 'LINE':
                try:
                    dist = e.dxf.start.distance(e.dxf.end)
                    linear_by_layer[layer] += (dist * to_feet)
                except Exception:
                    pass

            # Polylines - sum segment lengths
            elif e.dxftype() == 'LWPOLYLINE':
                try:
                    points = list(e.get_points())
                    length = 0
                    for i in range(len(points) - 1):
                        p1, p2 = points[i], points[i+1]
                        length += ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) ** 0.5

                    # Add closing segment if closed
                    if e.closed and len(points) > 1:
                        p1, p2 = points[-1], points[0]
                        length += ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) ** 0.5

                    linear_by_layer[layer] += (length * to_feet)
                except Exception:
                    pass

            # Blocks (drains, scuppers, equipment)
            elif e.dxftype() == 'INSERT':
                block_counts[e.dxf.name] += 1

        return {
            'format': 'QUANTITY',
            'purpose': 'Quantity takeoff for estimation AI (The Estimator)',
            'meta': meta,
            'linear_feet_by_layer': {k: round(v, 2) for k, v in sorted(linear_by_layer.items())},
            'total_linear_feet': round(sum(linear_by_layer.values()), 2),
            'block_counts': dict(sorted(block_counts.items())),
            'entity_counts_by_layer': {k: dict(v) for k, v in sorted(entity_counts.items())}
        }

    def _gen_specs(self, msp, meta: Dict, to_feet: float) -> Dict:
        """
        LENS 3: THE DETAILER - Specifications
        Text notes, dimensions, leaders for spec compliance checking.
        """
        data = {
            'format': 'SPECS',
            'purpose': 'Specification extraction for detail AI (The Detailer)',
            'meta': meta,
            'notes': [],
            'dimensions': [],
            'leaders': []
        }

        for e in msp:
            # TEXT entities
            if e.dxftype() == 'TEXT':
                try:
                    data['notes'].append({
                        'layer': e.dxf.layer,
                        'text': e.dxf.text.strip(),
                        'location': [round(e.dxf.insert.x, 2), round(e.dxf.insert.y, 2)],
                        'height': round(e.dxf.height, 2) if hasattr(e.dxf, 'height') else None
                    })
                except Exception:
                    pass

            # MTEXT entities (multi-line text)
            elif e.dxftype() == 'MTEXT':
                try:
                    data['notes'].append({
                        'layer': e.dxf.layer,
                        'text': e.text.strip(),
                        'location': [round(e.dxf.insert.x, 2), round(e.dxf.insert.y, 2)],
                        'height': round(e.dxf.char_height, 2) if hasattr(e.dxf, 'char_height') else None
                    })
                except Exception:
                    pass

            # DIMENSION entities
            elif e.dxftype() == 'DIMENSION':
                try:
                    raw_value = getattr(e.dxf, 'actual_measurement', 0)
                    data['dimensions'].append({
                        'layer': e.dxf.layer,
                        'value_raw': round(raw_value, 4),
                        'value_feet': round(raw_value * to_feet, 4),
                        'override_text': getattr(e.dxf, 'text', '') or None
                    })
                except Exception:
                    pass

            # LEADER entities
            elif e.dxftype() == 'LEADER':
                try:
                    data['leaders'].append({
                        'layer': e.dxf.layer,
                        'annotation': getattr(e.dxf, 'annotation', None)
                    })
                except Exception:
                    pass

        data['note_count'] = len(data['notes'])
        data['dimension_count'] = len(data['dimensions'])
        return data

    def _gen_full(self, msp, meta: Dict, to_feet: float) -> Dict:
        """
        FULL: Complete Data Dump
        All entities with full geometry data for comprehensive analysis.
        """
        data = {
            'format': 'FULL',
            'purpose': 'Complete DXF data extraction',
            'meta': meta,
            'entities': []
        }

        for e in msp:
            entity = {
                'type': e.dxftype(),
                'layer': e.dxf.layer,
                'color': getattr(e.dxf, 'color', None)
            }

            try:
                # Add type-specific data
                if e.dxftype() == 'LINE':
                    entity['start'] = [round(e.dxf.start.x, 4), round(e.dxf.start.y, 4)]
                    entity['end'] = [round(e.dxf.end.x, 4), round(e.dxf.end.y, 4)]
                    entity['length_raw'] = round(e.dxf.start.distance(e.dxf.end), 4)
                    entity['length_feet'] = round(e.dxf.start.distance(e.dxf.end) * to_feet, 4)

                elif e.dxftype() == 'LWPOLYLINE':
                    points = [[round(p[0], 4), round(p[1], 4)] for p in e.get_points()]
                    entity['points'] = points
                    entity['point_count'] = len(points)
                    entity['closed'] = e.closed

                elif e.dxftype() == 'CIRCLE':
                    entity['center'] = [round(e.dxf.center.x, 4), round(e.dxf.center.y, 4)]
                    entity['radius'] = round(e.dxf.radius, 4)
                    entity['radius_feet'] = round(e.dxf.radius * to_feet, 4)

                elif e.dxftype() == 'ARC':
                    entity['center'] = [round(e.dxf.center.x, 4), round(e.dxf.center.y, 4)]
                    entity['radius'] = round(e.dxf.radius, 4)
                    entity['start_angle'] = round(e.dxf.start_angle, 2)
                    entity['end_angle'] = round(e.dxf.end_angle, 2)

                elif e.dxftype() in ['TEXT', 'MTEXT']:
                    entity['content'] = e.dxf.text if e.dxftype() == 'TEXT' else e.text
                    entity['location'] = [round(e.dxf.insert.x, 4), round(e.dxf.insert.y, 4)]

                elif e.dxftype() == 'INSERT':
                    entity['block_name'] = e.dxf.name
                    entity['location'] = [round(e.dxf.insert.x, 4), round(e.dxf.insert.y, 4)]
                    entity['scale'] = [
                        round(getattr(e.dxf, 'xscale', 1), 4),
                        round(getattr(e.dxf, 'yscale', 1), 4)
                    ]
                    entity['rotation'] = round(getattr(e.dxf, 'rotation', 0), 2)

                elif e.dxftype() == 'DIMENSION':
                    raw_val = getattr(e.dxf, 'actual_measurement', 0)
                    entity['value_raw'] = round(raw_val, 4)
                    entity['value_feet'] = round(raw_val * to_feet, 4)

            except Exception:
                entity['parse_error'] = True

            data['entities'].append(entity)

        data['entity_count'] = len(data['entities'])
        return data


# Convenience functions for direct use
def get_dxf_preview(file_path: str = None, file_obj: BinaryIO = None) -> Dict:
    """Quick preview of DXF file contents."""
    processor = CADProcessor()
    return processor.get_preview(file_path, file_obj)


def process_dxf(file_path: str = None, file_obj: BinaryIO = None,
                output_types: List[str] = None) -> Dict[str, Dict]:
    """Process DXF file to JSON."""
    processor = CADProcessor()
    return processor.process(file_path, file_obj, output_types)
