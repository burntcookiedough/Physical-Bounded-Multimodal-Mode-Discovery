import json
import os
from sklearn.metrics.pairwise import cosine_similarity

class ConflictPatternLibrary:
    """
    Stores and retrieves modality conflict signatures.
    Each entry is a (conflict_vector, outcome_label, dataset_source) tuple.
    Library grows with each validated run.
    """
    
    def __init__(self, library_path='conflict_pattern_library.json'):
        self.library_path = library_path
        self.entries = self._load()
    
    def add_pattern(self, conflict_vector, outcome_label, dataset_source, risk_score):
        """
        conflict_vector: dict with keys = modality names, values = disagreement magnitude
        Example: {'vibration_de': 0.87, 'vibration_fe': 0.91, 'current': 0.12, 'temperature': 0.09}
        """
        entry = {
            'conflict_vector': conflict_vector,
            'vector_normalized': self._normalize(conflict_vector),
            'outcome_label': outcome_label,    # from PIAML auto-labeling
            'dataset_source': dataset_source,
            'risk_score': risk_score,
            'confirmed': False                  # becomes True after human review (optional)
        }
        self.entries.append(entry)
        self._save()
    
    def query(self, new_conflict_vector, top_k=3):
        """
        Given a new conflict pattern, find the k most similar historical patterns.
        Returns predicted label + confidence based on cosine similarity.
        """
        if not self.entries:
            return None
        
        new_vec = self._normalize(new_conflict_vector)
        # All known modalities across library + new vector
        all_mods = set(new_vec.keys())
        for e in self.entries:
            all_mods.update(e['vector_normalized'].keys())
            
        modalities = sorted(list(all_mods))
        
        new_array = [new_vec.get(m, 0.0) for m in modalities]
        stored_arrays = []
        for entry in self.entries:
            stored_arrays.append([entry['vector_normalized'].get(m, 0.0) for m in modalities])
        
        similarities = cosine_similarity([new_array], stored_arrays)[0]
        # Get top k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        matches = []
        for idx in top_indices:
            matches.append({
                'outcome_label': self.entries[idx]['outcome_label'],
                'similarity': float(similarities[idx]),
                'dataset_source': self.entries[idx]['dataset_source'],
                'risk_score': self.entries[idx]['risk_score']
            })
        
        # Weighted vote for predicted label
        label_scores = {}
        for match in matches:
            label = match['outcome_label']
            label_scores[label] = label_scores.get(label, 0.0) + match['similarity']
        
        predicted_label = max(label_scores, key=label_scores.get)
        total_score = sum(label_scores.values())
        confidence = label_scores[predicted_label] / total_score if total_score > 0 else 0.0
        
        return {
            'predicted_outcome': predicted_label,
            'confidence': confidence,
            'top_matches': matches,
            'library_size': len(self.entries)
        }
    
    def _normalize(self, conflict_vector):
        total = sum(conflict_vector.values()) + 1e-10
        return {k: v / total for k, v in conflict_vector.items()}
    
    def _save(self):
        with open(self.library_path, 'w') as f:
            json.dump(self.entries, f, indent=2)
    
    def _load(self):
        if not os.path.exists(self.library_path):
            return []
        try:
            with open(self.library_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
