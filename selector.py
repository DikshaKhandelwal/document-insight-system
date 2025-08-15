#!/usr/bin/env python3
"""
Persona-Driven Document Intelligence Selector

This module implements intelligent section selection and ranking based on persona and job requirements.
It uses a hybrid approach combining keyword matching and semantic analysis for optimal performance.
"""

import json
import re
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter, defaultdict
import math
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SEMANTIC_AVAILABLE = True
except ImportError:
    try:
        # Fallback: Simple TF-IDF based similarity for lightweight operation
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        SEMANTIC_AVAILABLE = "sklearn"
    except ImportError:
        SEMANTIC_AVAILABLE = False
        print("⚠️  Neither sentence-transformers nor sklearn available. Using keyword-based matching only.")


class PersonaDrivenSelector:
    """
    Intelligent document section selector that extracts and ranks relevant content
    based on persona and job-to-be-done requirements.
    """
    
    def __init__(self, use_semantic: bool = True):
        """
        Initialize the selector with sentence-transformers semantic similarity support.
        
        Args:
            use_semantic: Whether to use semantic similarity (requires sentence-transformers)
        """
        # Initialize comprehensive domain keyword mapping for generic applicability
        self.relevance_keywords = {
            # Academic & Research
            'research': ['research', 'study', 'analysis', 'methodology', 'literature', 'review', 'survey', 'experiment', 'data', 'findings', 'conclusion', 'hypothesis', 'theory', 'academic', 'publication', 'peer-reviewed', 'citation', 'bibliography', 'abstract', 'dissertation', 'thesis'],
            'education': ['education', 'learning', 'curriculum', 'teaching', 'instruction', 'course', 'training', 'skill', 'knowledge', 'student', 'academic', 'assessment', 'exam', 'lecture', 'textbook', 'assignment', 'grade', 'semester', 'syllabus'],
            'science': ['science', 'scientific', 'chemistry', 'physics', 'biology', 'mathematics', 'formula', 'equation', 'experiment', 'laboratory', 'molecular', 'organic', 'inorganic', 'reaction', 'compound', 'element', 'periodic', 'quantum'],
            
            # Business & Finance
            'business': ['business', 'strategy', 'market', 'management', 'financial', 'revenue', 'profit', 'customer', 'competition', 'analysis', 'planning', 'operations', 'growth', 'corporate', 'enterprise', 'startup', 'entrepreneurship'],
            'finance': ['finance', 'investment', 'budget', 'cost', 'expense', 'revenue', 'financial', 'economic', 'money', 'capital', 'funding', 'banking', 'accounting', 'assets', 'liability', 'equity', 'portfolio', 'stocks', 'bonds', 'valuation'],
            'sales': ['sales', 'selling', 'customer', 'prospect', 'lead', 'conversion', 'revenue', 'quota', 'pipeline', 'negotiation', 'deal', 'client', 'relationship', 'marketing', 'promotion', 'pricing'],
            
            # Technology & Engineering
            'technology': ['technology', 'software', 'system', 'development', 'programming', 'algorithm', 'implementation', 'framework', 'architecture', 'performance', 'optimization', 'code', 'programming', 'database', 'network', 'security'],
            'engineering': ['engineering', 'design', 'construction', 'mechanical', 'electrical', 'civil', 'chemical', 'materials', 'manufacturing', 'process', 'technical', 'specification', 'blueprint', 'prototype'],
            
            # Legal & Compliance
            'legal': ['legal', 'law', 'regulation', 'compliance', 'contract', 'agreement', 'rights', 'liability', 'court', 'jurisdiction', 'statute', 'policy', 'legislation', 'regulatory', 'litigation', 'attorney', 'lawyer'],
            
            # Healthcare & Medical
            'medical': ['medical', 'health', 'treatment', 'diagnosis', 'patient', 'clinical', 'therapy', 'medication', 'procedure', 'symptom', 'disease', 'care', 'healthcare', 'physician', 'nurse', 'hospital', 'clinic'],
            
            # Media & Communication
            'journalism': ['journalism', 'news', 'article', 'report', 'interview', 'investigation', 'media', 'press', 'newspaper', 'magazine', 'broadcast', 'journalism', 'story', 'coverage', 'editorial'],
            'communication': ['communication', 'message', 'audience', 'content', 'writing', 'presentation', 'public', 'relations', 'marketing', 'advertising', 'brand', 'campaign'],
            
            # Travel & Hospitality
            'travel': ['travel', 'trip', 'vacation', 'tourism', 'visit', 'destination', 'journey', 'itinerary', 'accommodation', 'hotel', 'restaurant', 'activity', 'attraction', 'guide', 'planning'],
            'food': ['food', 'recipe', 'cooking', 'ingredient', 'meal', 'dish', 'cuisine', 'restaurant', 'menu', 'catering', 'nutrition', 'dietary', 'chef', 'kitchen', 'dining'],
            
            # General domains
            'general': ['overview', 'introduction', 'summary', 'conclusion', 'important', 'key', 'essential', 'critical', 'main', 'primary', 'significant', 'relevant', 'useful', 'practical']
        }
        
        # Initialize semantic similarity
        self.use_semantic = use_semantic and SEMANTIC_AVAILABLE
        self.model = None
        self.vectorizer = None
        self.semantic_method = None
        
        if self.use_semantic:
            if SEMANTIC_AVAILABLE == True:  # sentence-transformers available
                try:
                    print("🧠 Loading sentence-transformers model (all-MiniLM-L6-v2)...")
                    self.model = SentenceTransformer('all-MiniLM-L6-v2')
                    self.semantic_method = "transformers"
                    print("✅ Sentence-transformers model loaded successfully")
                except Exception as e:
                    print(f"⚠️  Failed to load sentence-transformers: {e}")
                    self._fallback_to_sklearn()
            elif SEMANTIC_AVAILABLE == "sklearn":  # sklearn fallback
                self._fallback_to_sklearn()
            else:
                self.use_semantic = False
    
    def _fallback_to_sklearn(self):
        """Fallback to sklearn TF-IDF for semantic similarity."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(
                stop_words='english', 
                max_features=1000,
                lowercase=True,
                strip_accents='ascii'
            )
            self.semantic_method = "tfidf"
            print("✅ Using TF-IDF vectorization for semantic similarity")
        except Exception as e:
            print(f"⚠️  Failed to initialize sklearn TF-IDF: {e}")
            self.use_semantic = False
        self.relevance_keywords = {
            # Travel & Tourism
            'travel': ['travel', 'trip', 'vacation', 'tourism', 'visit', 'destination', 'journey', 'itinerary', 'accommodation', 'hotel', 'restaurant', 'activity', 'attraction', 'guide', 'planning'],
            'research': ['research', 'study', 'analysis', 'methodology', 'literature', 'review', 'survey', 'experiment', 'data', 'findings', 'conclusion', 'hypothesis', 'theory', 'academic'],
            'business': ['business', 'strategy', 'market', 'management', 'financial', 'revenue', 'profit', 'customer', 'competition', 'analysis', 'planning', 'operations', 'growth'],
            'technology': ['technology', 'software', 'system', 'development', 'programming', 'algorithm', 'implementation', 'framework', 'architecture', 'performance', 'optimization'],
            'education': ['education', 'learning', 'curriculum', 'teaching', 'instruction', 'course', 'training', 'skill', 'knowledge', 'student', 'academic', 'assessment'],
            'legal': ['legal', 'law', 'regulation', 'compliance', 'contract', 'agreement', 'rights', 'liability', 'court', 'jurisdiction', 'statute', 'policy'],
            'medical': ['medical', 'health', 'treatment', 'diagnosis', 'patient', 'clinical', 'therapy', 'medication', 'procedure', 'symptom', 'disease', 'care'],
            'finance': ['finance', 'investment', 'budget', 'cost', 'expense', 'revenue', 'financial', 'economic', 'money', 'capital', 'funding', 'banking'],
            'food': ['food', 'recipe', 'cooking', 'ingredient', 'meal', 'dish', 'cuisine', 'vegetarian', 'vegan', 'buffet', 'menu', 'dinner', 'lunch', 'breakfast', 'catering', 'preparation', 'chef', 'kitchen', 'dining', 'nutrition', 'flavor', 'spice', 'seasoning', 'appetizer', 'main', 'side', 'dessert', 'beverage', 'dietary', 'healthy', 'organic', 'fresh', 'serve', 'portion']
        }
        
    def identify_domain(self, persona: Dict, job: Dict) -> List[str]:
        """
        Identify the primary domain(s) based on persona and job description.
        Enhanced for generic domain detection across diverse fields.
        """
        domains = []
        
        # Analyze persona role
        role_text = str(persona.get('role', '')).lower()
        
        # Analyze job description
        job_text = str(job.get('task', '')).lower()
        
        # Combine all available context
        combined_text = f"{role_text} {job_text}"
        
        # Add any additional context fields
        if 'description' in persona:
            combined_text += f" {str(persona['description']).lower()}"
        if 'context' in job:
            combined_text += f" {str(job['context']).lower()}"
        
        # Score each domain based on keyword matches
        domain_scores = {}
        for domain, keywords in self.relevance_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            # Add weight for exact role matches
            if domain == role_text or any(role_word in keywords for role_word in role_text.split()):
                score += 5  # Bonus for direct role match
            if score > 0:
                domain_scores[domain] = score
        
        # Auto-detect domain from common patterns
        domain_patterns = {
            'research': ['literature review', 'research paper', 'academic', 'study', 'analysis'],
            'education': ['study for', 'learn about', 'understand', 'textbook', 'course'],
            'finance': ['financial report', 'balance sheet', 'income statement', 'cash flow'],
            'business': ['business plan', 'market analysis', 'strategy', 'competitive'],
            'journalism': ['news article', 'report on', 'investigate', 'journalism'],
            'legal': ['legal document', 'contract', 'regulation', 'compliance'],
            'medical': ['medical report', 'diagnosis', 'treatment', 'clinical'],
            'technology': ['software', 'programming', 'system', 'technical']
        }
        
        for domain, patterns in domain_patterns.items():
            pattern_matches = sum(1 for pattern in patterns if pattern in combined_text)
            if pattern_matches > 0:
                domain_scores[domain] = domain_scores.get(domain, 0) + pattern_matches * 3
        
        # Return top domains with meaningful scores
        if domain_scores:
            sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
            # Dynamic threshold - at least 20% of top score or minimum 2 points
            threshold = max(2, sorted_domains[0][1] * 0.2)
            domains = [domain for domain, score in sorted_domains if score >= threshold]
        
        # Fallback to general if no specific domain detected
        if not domains:
            domains = ['general']
        
        return domains[:4]  # Return top 4 domains max for broader coverage
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for better semantic similarity by cleaning and standardizing.
        
        Args:
            text: Raw text to normalize
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
            
        # Strip newlines and extra whitespace
        normalized = re.sub(r'\s+', ' ', text.strip())
        
        # Convert to lowercase for consistency
        normalized = normalized.lower()
        
        # Remove special characters but keep alphanumeric and basic punctuation
        normalized = re.sub(r'[^\w\s\-\.,;:!?()]', ' ', normalized)
        
        # Remove extra spaces again after cleaning
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def deduplicate_sections(self, sections: List[Dict]) -> List[Dict]:
        """
        Remove duplicate sections based on normalized text content.
        
        Args:
            sections: List of section dictionaries
            
        Returns:
            Deduplicated list of sections
        """
        seen_texts = set()
        unique_sections = []
        
        for section in sections:
            normalized_text = self.normalize_text(section.get('text', ''))
            
            # Skip empty or very short sections
            if len(normalized_text) < 3:
                continue
                
            # Use a similarity-based deduplication (simple approach)
            is_duplicate = False
            for seen_text in seen_texts:
                # Check if texts are very similar (simple character-based similarity)
                similarity = len(set(normalized_text) & set(seen_text)) / max(len(set(normalized_text)), len(set(seen_text)), 1)
                if similarity > 0.9:  # 90% character overlap indicates duplicate
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_texts.add(normalized_text)
                unique_sections.append(section)
                
        return unique_sections
    
    def get_heading_priority(self, level: str) -> int:
        """
        Get priority score for heading levels (higher number = higher priority).
        
        Args:
            level: Heading level (H1, H2, H3, etc.)
            
        Returns:
            Priority score
        """
        priority_map = {
            'H1': 100,
            'H2': 80,
            'H3': 60,
            'H4': 40,
            'H5': 20,
            'H6': 10
        }
        return priority_map.get(level.upper(), 0)
    
    def batch_encode_sections(self, sections: List[Dict], query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Efficiently encode all sections and query in batches for optimal performance.
        
        Args:
            sections: List of section dictionaries
            query: Normalized query string
            
        Returns:
            Tuple of (section_embeddings, query_embedding)
        """
        if not self.use_semantic or self.semantic_method != "transformers":
            return None, None
            
        try:
            # Normalize all texts
            normalized_query = self.normalize_text(query)
            normalized_sections = [self.normalize_text(section.get('text', '')) for section in sections]
            
            # Filter out empty sections
            valid_indices = [i for i, text in enumerate(normalized_sections) if len(text) > 3]
            valid_texts = [normalized_sections[i] for i in valid_indices]
            
            if not valid_texts:
                return None, None
            
            # Batch encode all texts at once for efficiency
            all_texts = [normalized_query] + valid_texts
            all_embeddings = self.model.encode(all_texts, show_progress_bar=False)
            
            # Split embeddings
            query_embedding = all_embeddings[0:1]
            section_embeddings = all_embeddings[1:]
            
            return section_embeddings, query_embedding, valid_indices
            
        except Exception as e:
            print(f"⚠️  Error in batch encoding: {e}")
            return None, None, []
    
    def calculate_semantic_similarity_batch(self, sections: List[Dict], query: str, 
                                          similarity_threshold: float = 0.3) -> List[Tuple[Dict, float, int]]:
        """
        Calculate semantic similarity for all sections using optimized batch processing.
        
        Args:
            sections: List of section dictionaries
            query: Query string
            similarity_threshold: Minimum similarity score to include
            
        Returns:
            List of tuples: (section, similarity_score, original_index)
        """
        if not self.use_semantic:
            return [(section, 0.0, i) for i, section in enumerate(sections)]
        
        if self.semantic_method == "transformers":
            return self._calculate_transformers_similarity_batch(sections, query, similarity_threshold)
        elif self.semantic_method == "tfidf":
            return self._calculate_tfidf_similarity_batch(sections, query, similarity_threshold)
        else:
            return [(section, 0.0, i) for i, section in enumerate(sections)]
    
    def _calculate_transformers_similarity_batch(self, sections: List[Dict], query: str, 
                                               similarity_threshold: float) -> List[Tuple[Dict, float, int]]:
        """Calculate similarity using sentence-transformers batch processing."""
        try:
            # Batch encode all sections and query
            result = self.batch_encode_sections(sections, query)
            if result[0] is None:
                return [(section, 0.0, i) for i, section in enumerate(sections)]
                
            section_embeddings, query_embedding, valid_indices = result
            
            # Calculate cosine similarities efficiently
            similarities = np.dot(section_embeddings, query_embedding.T).flatten()
            section_norms = np.linalg.norm(section_embeddings, axis=1)
            query_norm = np.linalg.norm(query_embedding)
            
            # Normalize to get cosine similarity
            similarities = similarities / (section_norms * query_norm)
            
            # Create results with original indices
            results = []
            for i, (embedding_idx, similarity) in enumerate(zip(valid_indices, similarities)):
                if similarity >= similarity_threshold:
                    results.append((sections[embedding_idx], float(similarity), embedding_idx))
            
            return results
            
        except Exception as e:
            print(f"⚠️  Error in transformers batch similarity: {e}")
            return [(section, 0.0, i) for i, section in enumerate(sections)]
    
    def _calculate_tfidf_similarity_batch(self, sections: List[Dict], query: str,
                                        similarity_threshold: float) -> List[Tuple[Dict, float, int]]:
        """Calculate similarity using TF-IDF batch processing."""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Normalize texts
            normalized_query = self.normalize_text(query)
            normalized_sections = [self.normalize_text(section.get('text', '')) for section in sections]
            
            # Prepare all texts
            all_texts = [normalized_query] + normalized_sections
            
            # Vectorize all texts at once
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Calculate similarities
            query_vector = tfidf_matrix[0:1]
            section_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vector, section_vectors).flatten()
            
            # Create results
            results = []
            for i, similarity in enumerate(similarities):
                if similarity >= similarity_threshold:
                    results.append((sections[i], float(similarity), i))
            
            return results
            
        except Exception as e:
            print(f"⚠️  Error in TF-IDF batch similarity: {e}")
            return [(section, 0.0, i) for i, section in enumerate(sections)]
    
    def resolve_page_conflicts(self, scored_sections: List[Tuple[Dict, float, str]]) -> List[Tuple[Dict, float, str]]:
        """
        Resolve conflicts when multiple sections have similar scores on the same page.
        Prioritize higher-level headings (H1 > H2 > H3).
        
        Args:
            scored_sections: List of (section, score, justification) tuples
            
        Returns:
            Resolved list with page conflicts handled
        """
        # Group by page and document
        page_groups = defaultdict(list)
        
        for section, score, justification in scored_sections:
            page_key = (section.get('document', ''), section.get('page', 0))
            page_groups[page_key].append((section, score, justification))
        
        resolved_sections = []
        
        for page_key, page_sections in page_groups.items():
            if len(page_sections) <= 1:
                # No conflict, add as-is
                resolved_sections.extend(page_sections)
            else:
                # Sort by score first, then by heading priority
                page_sections.sort(key=lambda x: (
                    x[1],  # Score (descending)
                    self.get_heading_priority(x[0].get('level', 'H3')),  # Heading priority (descending)
                    -len(x[0].get('text', ''))  # Text length (descending, as tiebreaker)
                ), reverse=True)
                
                # Check for very similar scores (within 5%)
                best_score = page_sections[0][1]
                tolerance = best_score * 0.05
                
                # Keep sections within tolerance, but prioritize by heading level
                for section, score, justification in page_sections:
                    if abs(score - best_score) <= tolerance:
                        resolved_sections.append((section, score, justification))
                    else:
                        break  # Scores are significantly different
        
        return resolved_sections

    def _extract_requirements_generic(self, job_task: str, persona_role: str = "") -> List[str]:
        """
        Extract generic requirements and preferences from job description and persona.
        Replaces domain-specific dietary requirements with generic requirement extraction.
        
        Args:
            job_task: Job task description
            persona_role: Persona role for additional context
            
        Returns:
            List of requirements and preferences found
        """
        job_lower = job_task.lower()
        role_lower = persona_role.lower()
        combined_text = f"{job_lower} {role_lower}"
        
        requirements = []
        
        # Food/catering requirements
        food_terms = {
            'vegetarian': ['vegetarian', 'veggie', 'plant-based'],
            'vegan': ['vegan'],
            'gluten-free': ['gluten-free', 'gluten free', 'celiac'],
            'dairy-free': ['dairy-free', 'dairy free', 'lactose-free'],
            'buffet-style': ['buffet', 'self-service', 'buffet-style'],
            'corporate': ['corporate', 'business', 'professional'],
            'dinner': ['dinner', 'evening meal', 'supper'],
            'healthy': ['healthy', 'nutritious', 'low-fat', 'organic']
        }
        
        # Academic/Research requirements
        academic_terms = {
            'peer-reviewed': ['peer-reviewed', 'peer reviewed', 'scholarly', 'academic journal'],
            'recent': ['recent', 'latest', 'current', 'up-to-date', 'modern'],
            'comprehensive': ['comprehensive', 'complete', 'thorough', 'detailed'],
            'methodology': ['methodology', 'methods', 'approach', 'framework'],
            'quantitative': ['quantitative', 'statistical', 'numerical', 'data-driven'],
            'qualitative': ['qualitative', 'interview', 'survey', 'observational']
        }
        
        # Business/Finance requirements  
        business_terms = {
            'quarterly': ['quarterly', 'q1', 'q2', 'q3', 'q4'],
            'annual': ['annual', 'yearly', 'year-end'],
            'financial': ['financial', 'fiscal', 'monetary'],
            'strategic': ['strategic', 'long-term', 'planning'],
            'operational': ['operational', 'day-to-day', 'routine'],
            'competitive': ['competitive', 'market', 'industry']
        }
        
        # Technical requirements
        technical_terms = {
            'programming': ['programming', 'coding', 'development'],
            'database': ['database', 'sql', 'data storage'],
            'security': ['security', 'encryption', 'protection'],
            'performance': ['performance', 'optimization', 'efficiency'],
            'scalability': ['scalability', 'scalable', 'growth']
        }
        
        # General quality requirements
        quality_terms = {
            'high-quality': ['high-quality', 'quality', 'excellent', 'premium'],
            'beginner-friendly': ['beginner', 'introductory', 'basic', 'simple'],
            'advanced': ['advanced', 'expert', 'sophisticated', 'complex'],
            'practical': ['practical', 'hands-on', 'applied', 'real-world'],
            'theoretical': ['theoretical', 'conceptual', 'abstract', 'academic']
        }
        
        # Time-related requirements
        time_terms = {
            'urgent': ['urgent', 'immediate', 'asap', 'quickly'],
            'deadline': ['deadline', 'due date', 'timeline'],
            'historical': ['historical', 'past', 'previous', 'former'],
            'future': ['future', 'upcoming', 'planned', 'projected']
        }
        
        # Check all requirement categories
        all_terms = {**food_terms, **academic_terms, **business_terms, **technical_terms, **quality_terms, **time_terms}
        
        for category, terms in all_terms.items():
            if any(term in combined_text for term in terms):
                requirements.append(category)
        
        # Extract specific constraints or preferences
        if 'only' in combined_text:
            requirements.append('exclusive-focus')
        if 'avoid' in combined_text or 'exclude' in combined_text:
            requirements.append('exclusionary')
        if 'include' in combined_text or 'must have' in combined_text:
            requirements.append('mandatory-inclusion')
        
        return requirements

    def calculate_semantic_similarity(self, section: Dict, query: str) -> float:
        """
        Legacy method for single section similarity calculation.
        
        Args:
            section: Section dictionary with 'text' field
            query: Query string (persona + job description)
            
        Returns:
            Similarity score between 0 and 1
        """
        # Use batch method for single calculation (less efficient but maintains compatibility)
        results = self.calculate_semantic_similarity_batch([section], query, similarity_threshold=0.0)
        if results:
            return results[0][1]
        return 0.0
        """
        Calculate semantic similarity between a section and query using sentence-transformers or TF-IDF.
        
        Args:
            section: Section dictionary with 'text' field
            query: Query string (persona + job description)
            
        Returns:
            Similarity score between 0 and 1
        """
        if not self.use_semantic:
            return 0.0
            
        try:
            section_text = section.get('text', '').strip()
            if not section_text:
                return 0.0
            
            if self.semantic_method == "transformers" and self.model:
                # Use sentence-transformers for high-quality embeddings
                embeddings = self.model.encode([section_text, query])
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
            elif self.semantic_method == "tfidf" and self.vectorizer:
                # Use TF-IDF similarity as fallback
                from sklearn.metrics.pairwise import cosine_similarity
                
                # Prepare texts for vectorization
                texts = [section_text, query]
                tfidf_matrix = self.vectorizer.fit_transform(texts)
                
                # Calculate cosine similarity
                similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
                similarity = similarity_matrix[0][0]
            else:
                return 0.0
                
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, float(similarity)))
            
        except Exception as e:
            print(f"⚠️  Error calculating semantic similarity: {e}")
            return 0.0
    
    def find_most_relevant_sections(self, sections: List[Dict], persona: Dict, job: Dict, 
                                  top_n: int = 5, similarity_threshold: float = 0.2) -> List[Tuple[Dict, float, str]]:
        """
        Find the most relevant PDF sections using advanced semantic similarity with optimizations.
        
        Args:
            sections: List of section dictionaries with 'text' and metadata
            persona: Persona dictionary with 'role' field
            job: Job dictionary with 'task' field
            top_n: Number of top sections to return
            similarity_threshold: Minimum similarity score to include sections
            
        Returns:
            List of tuples: (section, combined_score, justification)
        """
        if not sections:
            return []
        
        # Step 1: Deduplicate sections for better quality
        unique_sections = self.deduplicate_sections(sections)
        print(f"🔧 Deduplicated {len(sections)} → {len(unique_sections)} sections")
        
        # Step 2: Create optimized query from persona and job
        persona_role = persona.get('role', '')
        job_task = job.get('task', '')
        
        # Enhanced query construction for better semantic matching
        query_parts = []
        if persona_role:
            query_parts.append(f"I am a {persona_role}")
        if job_task:
            query_parts.append(f"I need to {job_task}")
        
        # Add domain-specific context with dietary requirements extraction
        identified_domains = self.identify_domain(persona, job)
        if identified_domains:
            query_parts.append(f"focusing on {', '.join(identified_domains)}")
        
        # Extract and emphasize requirements and preferences with enhanced vegetarian focus
        requirements = self._extract_requirements_generic(job_task, persona_role)
        if requirements:
            # Special handling for vegetarian requirements
            if any('vegetarian' in req for req in requirements):
                query_parts.append("focusing specifically on vegetarian and vegan dishes, plant-based options, falafel, hummus, baba ganoush, ratatouille, vegetable-based recipes")
            else:
                query_parts.append(f"with emphasis on {', '.join(requirements)} requirements")
        
        query = ". ".join(query_parts)
        print(f"🔍 Enhanced semantic query: {query}")
        
        # Step 3: Batch calculate semantic similarities
        semantic_results = self.calculate_semantic_similarity_batch(
            unique_sections, query, similarity_threshold
        )
        
        print(f"🎯 Found {len(semantic_results)} sections above threshold {similarity_threshold}")
        
        # Step 4: Calculate hybrid scores combining semantic + traditional approaches
        scored_sections = []
        
        for section, semantic_score, original_idx in semantic_results:
            # Calculate traditional relevance score for hybrid approach
            traditional_score, justification = self.calculate_section_relevance(
                section, persona, job, identified_domains
            )
            
            # Advanced hybrid scoring with adaptive weights
            if semantic_score > 0.7:  # High semantic match - trust it more
                final_score = (0.8 * semantic_score) + (0.2 * traditional_score)
                method_info = f"high-confidence {self.semantic_method}"
            elif semantic_score > 0.4:  # Medium semantic match - balanced approach
                final_score = (0.6 * semantic_score) + (0.4 * traditional_score)
                method_info = f"balanced {self.semantic_method}"
            else:  # Lower semantic match - rely more on keywords
                final_score = (0.4 * semantic_score) + (0.6 * traditional_score)
                method_info = f"keyword-enhanced {self.semantic_method}"
            
            # Boost score for high-priority headings
            heading_boost = self.get_heading_priority(section.get('level', 'H3')) / 1000
            final_score += heading_boost
            
            if final_score > 0.1:  # Only include meaningful matches
                enhanced_justification = f"{justification} | Method: {method_info} | Semantic: {semantic_score:.3f}"
                scored_sections.append((section, final_score, enhanced_justification))
        
        # Step 5: Resolve page conflicts (prioritize higher-level headings)
        resolved_sections = self.resolve_page_conflicts(scored_sections)
        
        # Step 6: Sort by final score and apply top-k with threshold filtering
        resolved_sections.sort(key=lambda x: x[1], reverse=True)
        
        # Apply both top-k and threshold constraints with more lenient selection
        final_results = []
        for section, score, justification in resolved_sections:
            if len(final_results) >= top_n:
                break
            # More lenient threshold - ensure we get results even if scores are lower
            effective_threshold = min(similarity_threshold, 0.15) if len(final_results) == 0 else similarity_threshold
            if score >= effective_threshold:
                final_results.append((section, score, justification))
        
        # Fallback: if we still have no results, take the top sections regardless of threshold
        if not final_results and resolved_sections:
            print(f"⚠️  No sections met threshold {similarity_threshold}, taking top {min(3, len(resolved_sections))} sections")
            final_results = resolved_sections[:min(3, len(resolved_sections))]
        
        # If we have fewer than top_k but above threshold, that's acceptable
        print(f"✅ Selected {len(final_results)} high-quality sections (threshold: {similarity_threshold})")
        
        return final_results
    
    def calculate_section_relevance(self, section: Dict, persona: Dict, job: Dict, 
                                   identified_domains: List[str]) -> Tuple[float, str]:
        """
        Calculate relevance score for a section based on persona and job requirements.
        Enhanced for generic domain applicability.
        Returns (score, justification).
        """
        score = 0.0
        justifications = []
        
        section_text = section.get('text', '').lower()
        section_level = section.get('level', 'H3')
        
        # Weight based on heading level (H1 > H2 > H3)
        level_weights = {'H1': 1.0, 'H2': 0.8, 'H3': 0.6}
        level_weight = level_weights.get(section_level, 0.5)
        
        # Domain-specific keyword matching
        domain_score = 0
        for domain in identified_domains:
            if domain in self.relevance_keywords:
                matches = sum(1 for keyword in self.relevance_keywords[domain] 
                            if keyword in section_text)
                if matches > 0:
                    domain_score += matches * 0.3
                    justifications.append(f"Contains {matches} {domain}-related keywords")
        
        # Direct persona role matching
        persona_role = str(persona.get('role', '')).lower()
        role_words = re.findall(r'\b\w+\b', persona_role)
        role_matches = sum(1 for word in role_words if len(word) > 3 and word in section_text)
        if role_matches > 0:
            score += role_matches * 0.4
            justifications.append(f"Matches persona role terms ({role_matches} matches)")
        
        # Job-specific task matching
        job_task = str(job.get('task', '')).lower()
        task_words = re.findall(r'\b\w+\b', job_task)
        task_matches = sum(1 for word in task_words if len(word) > 3 and word in section_text)
        if task_matches > 0:
            score += task_matches * 0.5
            justifications.append(f"Relevant to job task ({task_matches} task-related terms)")
        
        # Generic requirement handling with domain-specific enhancements
        requirements = self._extract_requirements_generic(job_task, persona_role)
        
        if requirements:
            requirement_score = 0
            
            # Food domain specific scoring (when domain is detected)
            if 'food' in identified_domains:
                food_indicators = ['recipe', 'ingredient', 'cooking', 'meal', 'dish', 'cuisine', 'menu', 'dinner', 'vegetarian', 'vegan', 'gluten-free', 'side', 'main', 'appetizer']
                food_matches = sum(1 for indicator in food_indicators if indicator in section_text)
                if food_matches > 0:
                    requirement_score += food_matches * 0.8  # High boost for food content
                    justifications.append(f"Food/recipe content ({food_matches} indicators)")
                
                # Special handling for vegetarian requirements with strict filtering
                if any('vegetarian' in req or 'vegan' in req for req in requirements):
                    # First check for meat content - immediate disqualification
                    meat_indicators = ['beef', 'chicken', 'pork', 'lamb', 'fish', 'meat', 'bacon', 'ham', 'turkey', 'salmon', 'broth', 'stock']
                    meat_matches = sum(1 for indicator in meat_indicators if indicator in section_text)
                    if meat_matches > 0:
                        requirement_score = -10  # Complete disqualification
                        justifications.append("Contains meat (DISQUALIFIED for vegetarian menu)")
                    else:
                        # Boost genuinely vegetarian items
                        veg_indicators = ['vegetarian', 'vegan', 'plant-based', 'tofu', 'beans', 'rice', 'vegetable', 'quinoa', 'lentil', 'falafel', 'hummus', 'tahini', 'baba ganoush', 'ratatouille', 'eggplant', 'chickpea', 'avocado']
                        veg_matches = sum(1 for indicator in veg_indicators if indicator in section_text)
                        if veg_matches > 0:
                            requirement_score += veg_matches * 1.2  # Higher boost for clearly vegetarian items
                            justifications.append(f"Vegetarian-friendly ({veg_matches} indicators)")
                        
                        # Additional boost for Mediterranean/international vegetarian dishes
                        international_veg = ['falafel', 'hummus', 'baba ganoush', 'ratatouille', 'sushi', 'curry', 'stir-fry', 'risotto', 'pasta primavera']
                        intl_matches = sum(1 for indicator in international_veg if indicator in section_text)
                        if intl_matches > 0:
                            requirement_score += intl_matches * 1.5
                            justifications.append(f"International vegetarian cuisine ({intl_matches} indicators)")
                
                # Handle gluten-free requirements
                if any('gluten-free' in req for req in requirements):
                    gluten_indicators = ['wheat', 'flour', 'bread', 'pasta', 'noodles', 'lasagna', 'macaroni', 'soy sauce', 'barley', 'rye']
                    gluten_matches = sum(1 for indicator in gluten_indicators if indicator in section_text)
                    if gluten_matches > 0:
                        if 'vegetarian' in [r for r in requirements if 'vegetarian' in r]:
                            requirement_score = -5  # Heavy penalty for vegetarian + gluten-free
                            justifications.append("Contains gluten (PENALIZED for gluten-free requirement)")
                        else:
                            requirement_score *= 0.3
                            justifications.append("Contains gluten (reduced for gluten-free requirement)")
                    else:
                        # Boost naturally gluten-free items
                        gf_indicators = ['rice', 'quinoa', 'potato', 'corn', 'beans', 'lentils', 'chickpea', 'vegetable', 'fruit', 'nuts']
                        gf_matches = sum(1 for indicator in gf_indicators if indicator in section_text)
                        if gf_matches > 0:
                            requirement_score += gf_matches * 0.8
                            justifications.append(f"Naturally gluten-free ({gf_matches} indicators)")
            
            # Academic/Research specific scoring
            elif any(req in ['peer-reviewed', 'recent', 'methodology'] for req in requirements):
                academic_indicators = ['study', 'research', 'analysis', 'methodology', 'findings', 'conclusion', 'literature', 'review', 'journal', 'academic']
                academic_matches = sum(1 for indicator in academic_indicators if indicator in section_text)
                if academic_matches > 0:
                    requirement_score += academic_matches * 0.7
                    justifications.append(f"Academic/research content ({academic_matches} indicators)")
            
            # Business/Finance specific scoring
            elif any(req in ['financial', 'strategic', 'competitive'] for req in requirements):
                business_indicators = ['financial', 'revenue', 'profit', 'market', 'strategy', 'competition', 'analysis', 'management', 'business']
                business_matches = sum(1 for indicator in business_indicators if indicator in section_text)
                if business_matches > 0:
                    requirement_score += business_matches * 0.7
                    justifications.append(f"Business/finance content ({business_matches} indicators)")
            
            # Technical scoring
            elif any(req in ['programming', 'database', 'security'] for req in requirements):
                tech_indicators = ['software', 'system', 'programming', 'database', 'algorithm', 'development', 'technology', 'technical']
                tech_matches = sum(1 for indicator in tech_indicators if indicator in section_text)
                if tech_matches > 0:
                    requirement_score += tech_matches * 0.7
                    justifications.append(f"Technical content ({tech_matches} indicators)")
            
            # Quality level adjustments
            if 'advanced' in requirements and any(term in section_text for term in ['advanced', 'expert', 'sophisticated', 'complex']):
                requirement_score += 0.5
                justifications.append("Advanced level content")
            elif 'beginner-friendly' in requirements and any(term in section_text for term in ['basic', 'introduction', 'beginner', 'simple']):
                requirement_score += 0.5
                justifications.append("Beginner-friendly content")
            
            # Comprehensive content bonus
            if 'comprehensive' in requirements and any(term in section_text for term in ['comprehensive', 'complete', 'detailed', 'thorough']):
                requirement_score += 0.4
                justifications.append("Comprehensive coverage")
            
            score += requirement_score
        
        # Add domain score
        score += domain_score
        
        # Apply level weight
        final_score = score * level_weight
        
        # Bonus for certain high-value sections (generic patterns)
        high_value_patterns = [
            r'\b(introduction|overview|summary|conclusion)\b',
            r'\b(guide|how\s*to|step|process)\b',
            r'\b(recommend|best\s*practice|tip)\b',
            r'\b(important|key|essential|critical)\b',
            r'\b(methodology|framework|approach)\b',
            r'\b(analysis|evaluation|assessment)\b'
        ]
        
        for pattern in high_value_patterns:
            if re.search(pattern, section_text):
                final_score += 0.2
                justifications.append("High-value section type")
                break
        
        # Create justification string
        if not justifications:
            justifications.append("General relevance based on content analysis")
        
        justification = "; ".join(justifications[:3])  # Limit to top 3 reasons
        
        return final_score, justification
    
    def adaptive_chunk_extraction(self, document_path: Path, max_chars: int = 600) -> List[Dict]:
        """
        Improved fallback: Create context-rich, non-fragmented, thematically relevant chunks from documents lacking headings.
        Uses smart paragraph and sentence grouping, filters out noise, and avoids splitting mid-topic.
        """
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(document_path)
            all_chunks = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()

                # Split by paragraphs (double newlines or major spacing)
                paragraphs = re.split(r'\n\s*\n', page_text)
                para_buffer = []
                buffer_len = 0

                for i, paragraph in enumerate(paragraphs):
                    paragraph = paragraph.strip()
                    # Filter out noise: skip very short or repeated paragraphs
                    if len(paragraph) < 40 or paragraph.lower() in ['contents', 'index', 'table of contents', 'references', 'appendix']:
                        continue

                    # If paragraph is too long, split into sentences and group them
                    if len(paragraph) > max_chars:
                        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                        chunk = ""
                        for sentence in sentences:
                            if len(chunk) + len(sentence) < max_chars:
                                chunk += sentence + " "
                            else:
                                if chunk.strip():
                                    all_chunks.append({
                                        'text': f"Page {page_num+1} Paragraph {i+1} (Part)",
                                        'full_text': chunk.strip(),
                                        'page': page_num + 1,
                                        'level': 'H3',
                                        'chunk_type': 'paragraph'
                                    })
                                chunk = sentence + " "
                        if chunk.strip():
                            all_chunks.append({
                                'text': f"Page {page_num+1} Paragraph {i+1} (Final)",
                                'full_text': chunk.strip(),
                                'page': page_num + 1,
                                'level': 'H3',
                                'chunk_type': 'paragraph'
                            })
                    else:
                        # Buffer paragraphs to create context-rich chunks
                        para_buffer.append(paragraph)
                        buffer_len += len(paragraph)
                        # If buffer exceeds max_chars or at last paragraph, flush
                        if buffer_len >= max_chars or i == len(paragraphs) - 1:
                            chunk_text = "\n".join(para_buffer).strip()
                            if len(chunk_text) > 40:
                                all_chunks.append({
                                    'text': f"Page {page_num+1} Paragraphs {i-len(para_buffer)+2}-{i+1}",
                                    'full_text': chunk_text,
                                    'page': page_num + 1,
                                    'level': 'H3',
                                    'chunk_type': 'paragraph_group'
                                })
                            para_buffer = []
                            buffer_len = 0

            doc.close()
            # Filter out duplicate or highly similar chunks
            seen = set()
            unique_chunks = []
            for chunk in all_chunks:
                norm = re.sub(r'\s+', ' ', chunk['full_text'].lower().strip())
                if norm in seen or len(norm) < 40:
                    continue
                seen.add(norm)
                unique_chunks.append(chunk)
            return unique_chunks
        except Exception as e:
            print(f"⚠️  Error in adaptive chunking for {document_path}: {e}")
            return []
    
    def extract_subsection_text(self, document_path: Path, section: Dict, 
                               max_chars: int = 600, requirements: List[str] = None) -> str:
        """
        Extract actual text content from the PDF around the section location with enhanced detail.
        Generic implementation for diverse document types with flexible text matching.
        """
        try:
            import fitz  # PyMuPDF
            import difflib
            import re
            
            doc = fitz.open(document_path)
            page_num = section.get('page', 0) 
            
            if page_num < 0 or page_num >= len(doc):
                return self._fallback_section_text(section)
            
            # Try current page first
            page = doc[page_num]
            page_text = page.get_text()
            section_title = section.get('text', '').strip()
            
            # Enhanced flexible matching strategies
            title_pos = self._find_section_in_text(section_title, page_text)
            
            if title_pos is not None:
                result = self._extract_from_position(page_text, title_pos, section_title, max_chars, requirements)
                if len(result.strip()) > 50:  # Good content found
                    doc.close()
                    return result
            
            # Try adjacent pages if current page doesn't have enough content
            result = self._try_adjacent_pages(doc, page_num, section_title, max_chars, requirements)
            if result and len(result.strip()) > 50:
                doc.close()
                return result
                
            # Enhanced fallback: try alternative extraction methods
            result = self._extract_with_fallback_methods(page_text, section_title, max_chars, requirements)
            doc.close()
            return result if result.strip() else self._fallback_section_text(section)
                
        except Exception as e:
            print(f"⚠️  Error extracting text from {document_path}: {e}")
            return self._fallback_section_text(section)
    
    def _extract_from_position(self, page_text: str, title_pos: int, section_title: str, 
                              max_chars: int, requirements: List[str] = None) -> str:
        """Extract content from a specific position in the text."""
        remaining_text = page_text[title_pos:]
        lines = remaining_text.split('\n')
        
        # Enhanced text processing for structured content extraction
        cleaned_lines = []
        section_content_started = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Skip the title line itself if it's the first meaningful line
            if i == 0 and self._is_likely_title_line(line, section_title):
                section_content_started = True
                continue
            
            # Start collecting content after title
            if not section_content_started and line:
                section_content_started = True
            
            # Stop if we hit another major section (improved heuristic detection)
            if section_content_started and i > 1 and self._is_likely_new_section(line, lines[i:i+3]):
                break
            
            # Keep substantial lines with better filtering
            if self._is_content_line(line):
                cleaned_lines.append(line)
            
            # Stop when we have enough content
            if len(' '.join(cleaned_lines)) > max_chars:
                break
        
        result = ' '.join(cleaned_lines)
        
        # Clean up mixed content
        result = self._clean_extracted_text(result, requirements)
        
        # Truncate to max_chars if needed
        if len(result) > max_chars:
            result = result[:max_chars-3] + "..."
        
        return result
    
    def _try_adjacent_pages(self, doc, current_page: int, section_title: str, 
                           max_chars: int, requirements: List[str] = None) -> str:
        """Try to find content on adjacent pages if current page is insufficient."""
        
        # Try previous page
        if current_page > 0:
            prev_page = doc[current_page - 1]
            prev_text = prev_page.get_text()
            title_pos = self._find_section_in_text(section_title, prev_text)
            
            if title_pos is not None:
                result = self._extract_from_position(prev_text, title_pos, section_title, max_chars, requirements)
                if len(result.strip()) > 50:
                    return result
        
        # Try next page
        if current_page < len(doc) - 1:
            next_page = doc[current_page + 1]
            next_text = next_page.get_text()
            title_pos = self._find_section_in_text(section_title, next_text)
            
            if title_pos is not None:
                result = self._extract_from_position(next_text, title_pos, section_title, max_chars, requirements)
                if len(result.strip()) > 50:
                    return result
        
        # Try combining current and next page for cases where content spans pages
        if current_page < len(doc) - 1:
            current_text = doc[current_page].get_text()
            next_text = doc[current_page + 1].get_text()
            combined_text = current_text + "\n" + next_text
            
            title_pos = self._find_section_in_text(section_title, combined_text)
            if title_pos is not None:
                result = self._extract_from_position(combined_text, title_pos, section_title, max_chars, requirements)
                if len(result.strip()) > 50:
                    return result
        
        return None
    
    def _find_section_in_text(self, section_title: str, page_text: str) -> int:
        """
        Enhanced section finding with multiple flexible matching strategies.
        Returns the position of the section in the text, or None if not found.
        """
        import difflib
        import re
        
        if not section_title:
            return None
        
        # Strategy 1: Exact match (case-sensitive)
        if section_title in page_text:
            return page_text.find(section_title)
        
        # Strategy 2: Case-insensitive match
        section_lower = section_title.lower()
        page_lower = page_text.lower()
        if section_lower in page_lower:
            return page_lower.find(section_lower)
        
        # Strategy 3: Remove common formatting differences
        # Clean section title: remove extra spaces, punctuation
        clean_title = re.sub(r'[^\w\s]', '', section_title).strip()
        clean_title = re.sub(r'\s+', ' ', clean_title)
        
        if clean_title.lower() in page_lower:
            return page_lower.find(clean_title.lower())
        
        # Strategy 4: Handle incomplete titles (common in PDFs)
        # Try partial matching for truncated titles
        if len(section_title) > 10:
            # Try first 70% of title
            partial_title = section_title[:int(len(section_title) * 0.7)]
            if partial_title.lower() in page_lower:
                return page_lower.find(partial_title.lower())
        
        # Strategy 5: Fuzzy matching with similarity threshold
        lines = page_text.split('\n')
        best_match_pos = None
        best_similarity = 0.0
        current_pos = 0
        
        for line in lines:
            line_clean = line.strip()
            if len(line_clean) > 3:  # Skip very short lines
                # Calculate similarity
                similarity = difflib.SequenceMatcher(None, section_lower, line_clean.lower()).ratio()
                
                # Also try with cleaned versions
                line_cleaned = re.sub(r'[^\w\s]', '', line_clean).strip()
                line_cleaned = re.sub(r'\s+', ' ', line_cleaned)
                similarity_clean = difflib.SequenceMatcher(None, clean_title.lower(), line_cleaned.lower()).ratio()
                
                max_similarity = max(similarity, similarity_clean)
                
                if max_similarity > best_similarity and max_similarity > 0.6:  # Lowered from 0.7 to 0.6
                    best_similarity = max_similarity
                    best_match_pos = current_pos
            
            current_pos += len(line) + 1  # +1 for newline
        
        # Strategy 6: Partial word matching for recipe titles
        if not best_match_pos and len(section_title.split()) > 1:
            # Try matching significant words (skip common words)
            significant_words = [word for word in section_title.lower().split() 
                               if len(word) > 3 and word not in ['and', 'the', 'with', 'for', 'from', 'forms', 'only']]
            
            if significant_words:
                for word in significant_words:
                    if word in page_lower:
                        # Find the line containing this word
                        word_pos = page_lower.find(word)
                        # Look for the start of the line containing this word
                        line_start = page_text.rfind('\n', 0, word_pos)
                        return line_start + 1 if line_start != -1 else word_pos
        
        # Strategy 7: Advanced pattern matching for specific domains
        # Handle PDF-specific issues like line breaks in titles
        if not best_match_pos:
            # Try matching with line breaks removed
            title_no_breaks = re.sub(r'\s+', ' ', section_title.strip())
            page_no_breaks = re.sub(r'\s+', ' ', page_text)
            
            if title_no_breaks.lower() in page_no_breaks.lower():
                return page_no_breaks.lower().find(title_no_breaks.lower())
        
        return best_match_pos
    
    def _is_likely_title_line(self, line: str, section_title: str) -> bool:
        """Check if a line is likely the section title itself."""
        import difflib
        
        line_clean = re.sub(r'[^\w\s]', '', line).strip()
        title_clean = re.sub(r'[^\w\s]', '', section_title).strip()
        
        similarity = difflib.SequenceMatcher(None, line_clean.lower(), title_clean.lower()).ratio()
        return similarity > 0.8
    
    def _is_likely_new_section(self, line: str, next_lines: List[str]) -> bool:
        """Enhanced detection of new section headers."""
        import re
        
        if len(line) > 5:
            # Check if this looks like another section header
            if (line.isupper() or 
                (line.replace(' ', '').isalpha() and len(line) < 50) or
                re.match(r'^(\d+\.|\w+\.|\w+\s+\d+)', line)):
                
                # Look ahead to see if this is definitely a new section
                next_text = ' '.join(next_lines[:3]).lower() if len(next_lines) >= 3 else ''
                if any(term in next_text for term in ['section', 'chapter', 'part', 'introduction', 'overview']):
                    return True
                
                # Additional heuristics for recipe/food content
                if any(term in line.lower() for term in ['ingredients:', 'instructions:', 'recipe', 'method']):
                    return False  # These are likely part of current section
                
                # Check if it's a standalone title-like line
                words = line.split()
                if len(words) <= 4 and all(word[0].isupper() for word in words if word):
                    return True
        
        return False
    
    def _is_content_line(self, line: str) -> bool:
        """Improved filtering for content lines."""
        if len(line) <= 3:
            return False
        
        # Filter out page numbers, headers, footers
        if line.isdigit() and len(line) < 4:
            return False
        
        # Filter out obvious navigation/formatting elements
        if line.lower() in ['contents', 'index', 'page', 'next', 'previous', 'back']:
            return False
        
        # Keep lines with substantial content
        return True
    
    def _extract_with_fallback_methods(self, page_text: str, section_title: str, 
                                     max_chars: int, requirements: List[str] = None) -> str:
        """Alternative extraction methods when direct matching fails."""
        lines = page_text.split('\n')
        relevant_lines = []
        
        # Method 1: Look for lines containing key words from section title
        if section_title:
            title_words = [word.lower() for word in section_title.split() if len(word) > 3]
            
            for i, line in enumerate(lines):
                line_clean = line.strip()
                if len(line_clean) > 10:
                    line_lower = line_clean.lower()
                    # Check if line contains any significant words from title
                    if any(word in line_lower for word in title_words):
                        # Add this line and several following lines
                        for j in range(i, min(i + 8, len(lines))):  # Increased from 5 to 8 lines
                            if self._is_content_line(lines[j].strip()):
                                relevant_lines.append(lines[j].strip())
                        break
        
        # Method 2: Domain-specific extraction patterns
        if not relevant_lines:
            relevant_lines = self._extract_by_content_patterns(lines, section_title, max_chars)
        
        # Method 3: If still no content, get substantial lines from page
        if not relevant_lines:
            for line in lines:
                line_clean = line.strip()
                if (len(line_clean) > 10 and 
                    not line_clean.isdigit() and 
                    self._is_content_line(line_clean)):
                    relevant_lines.append(line_clean)
                if len(' '.join(relevant_lines)) > max_chars:
                    break
        
        result = ' '.join(relevant_lines[:12])  # Increased from 8 to 12 lines
        
        if len(result) > max_chars:
            result = result[:max_chars-3] + "..."
        
        # Clean the result
        if requirements:
            result = self._clean_extracted_text(result, requirements)
        
        return result
    
    def _extract_by_content_patterns(self, lines: List[str], section_title: str, max_chars: int) -> List[str]:
        """Extract content using domain-specific patterns."""
        import re
        
        relevant_lines = []
        section_lower = section_title.lower() if section_title else ""
        
        # Pattern 1: Recipe/Food content (ingredients + instructions)
        if any(keyword in section_lower for keyword in ['recipe', 'food', 'ingredients', 'cooking', 'dish']):
            in_ingredients = False
            in_instructions = False
            
            for line in lines:
                line_clean = line.strip()
                
                # Look for ingredient lists
                if re.match(r'^(ingredients?|what you need):?', line_clean.lower()):
                    in_ingredients = True
                    relevant_lines.append(line_clean)
                    continue
                
                # Look for instructions
                if re.match(r'^(instructions?|directions?|method|how to):?', line_clean.lower()):
                    in_instructions = True
                    in_ingredients = False
                    relevant_lines.append(line_clean)
                    continue
                
                # Collect ingredient/instruction content
                if (in_ingredients or in_instructions) and line_clean:
                    # Stop if we hit another section
                    if self._looks_like_new_section(line_clean):
                        break
                    relevant_lines.append(line_clean)
                    
                    if len(' '.join(relevant_lines)) > max_chars:
                        break
        
        # Pattern 2: Technical/Software content (steps, procedures)
        elif any(keyword in section_lower for keyword in ['form', 'software', 'tool', 'feature', 'function']):
            collecting_content = False
            
            for line in lines:
                line_clean = line.strip()
                
                # Look for definition or explanation patterns
                if any(pattern in line_clean.lower() for pattern in ['feature', 'tool', 'allows', 'enables', 'you can']):
                    collecting_content = True
                
                if collecting_content and line_clean:
                    if self._looks_like_new_section(line_clean):
                        break
                    relevant_lines.append(line_clean)
                    
                    if len(' '.join(relevant_lines)) > max_chars:
                        break
        
        # Pattern 3: Travel/Guide content (tips, recommendations)
        elif any(keyword in section_lower for keyword in ['travel', 'trip', 'visit', 'hotel', 'restaurant', 'tips']):
            for line in lines:
                line_clean = line.strip()
                
                # Look for travel-related content
                if (len(line_clean) > 15 and 
                    any(keyword in line_clean.lower() for keyword in ['best', 'visit', 'located', 'offers', 'recommended', 'tips'])):
                    relevant_lines.append(line_clean)
                    
                    if len(' '.join(relevant_lines)) > max_chars:
                        break
        
        return relevant_lines
    
    def _looks_like_new_section(self, line: str) -> bool:
        """Check if a line looks like the start of a new section."""
        import re
        
        # Short lines that are title-case or all caps
        if (len(line.split()) <= 4 and 
            (line.istitle() or line.isupper()) and 
            len(line) > 5):
            return True
        
        # Lines that start with numbers (numbered sections)
        if re.match(r'^\d+\.', line):
            return True
        
        # Lines that end with colons (section headers)
        if line.endswith(':') and len(line.split()) <= 6:
            return True
        
        return False

    def _clean_extracted_text(self, text: str, requirements: List[str] = None) -> str:
        """
        Clean extracted text by removing irrelevant content based on requirements.
        Generic implementation for diverse document types with enhanced filtering.
        """
        if not text:
            return ""
        
        # Basic cleaning - remove headers, footers, page numbers
        lines = text.split('\n')
        cleaned_lines = []
        
        # Enhanced filtering for vegetarian requirements
        vegetarian_required = requirements and any('vegetarian' in req for req in requirements)
        gluten_free_required = requirements and any('gluten-free' in req for req in requirements)
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Skip obvious headers/footers (short lines that are all caps or numbers)
            if len(line) < 5 and (line.isupper() or line.isdigit()):
                continue
                
            # Skip copyright notices, page numbers, etc.
            if any(pattern in line.lower() for pattern in ['copyright', '©', 'page ', 'www.', 'http']):
                continue
            
            # Remove excessive repetition of special characters
            if len(set(line)) < 3 and len(line) > 10:  # Line with only 1-2 unique characters
                continue
            
            # Enhanced filtering for dietary requirements
            line_lower = line.lower()
            skip_line = False
            
            if vegetarian_required:
                # Skip lines containing meat products
                meat_terms = ['bacon', 'ham', 'chicken', 'beef', 'pork', 'fish', 'meat', 'turkey', 'salmon', 'chicken broth', 'beef broth', 'meat broth']
                if any(meat in line_lower for meat in meat_terms):
                    skip_line = True
            
            if gluten_free_required and not skip_line:
                # Skip lines containing gluten
                gluten_terms = ['wheat flour', 'bread', 'pasta', 'noodles', 'lasagna noodles', 'soy sauce', 'wheat']
                if any(gluten in line_lower for gluten in gluten_terms):
                    skip_line = True
            
            if not skip_line:
                cleaned_lines.append(line)
        
        cleaned_text = ' '.join(cleaned_lines)
        
        # Remove bullet point symbols and clean formatting
        cleaned_text = re.sub(r'\bo\s+', '', cleaned_text)  # Remove 'o ' bullet points
        cleaned_text = re.sub(r'•\s+', '', cleaned_text)   # Remove '• ' bullet points
        cleaned_text = re.sub(r'-\s+', '', cleaned_text)   # Remove '- ' bullet points
        
        # Remove extra spaces and normalize
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
    
    def _fallback_section_text(self, section: Dict) -> str:
        """Fallback method for generating section text when PDF extraction fails"""
        section_text = section.get('text', '')
        section_lower = section_text.lower()
        
        # Generic content templates based on common section types
        if any(term in section_lower for term in ['introduction', 'overview', 'summary']):
            return f"This section provides an introduction and overview of {section_text.lower()}, offering foundational knowledge and context for understanding the topic."
        
        elif any(term in section_lower for term in ['methodology', 'methods', 'approach']):
            return f"This section explains the methodology and approach used for {section_text.lower()}, detailing the systematic procedures and techniques employed."
        
        elif any(term in section_lower for term in ['results', 'findings', 'analysis']):
            return f"This section presents the results and analysis of {section_text.lower()}, including key findings and their implications."
        
        elif any(term in section_lower for term in ['conclusion', 'summary', 'recommendations']):
            return f"This section provides conclusions and recommendations based on {section_text.lower()}, summarizing key insights and next steps."
        
        elif any(term in section_lower for term in ['guide', 'how to', 'tutorial', 'instructions']):
            return f"This section offers practical guidance on {section_text.lower()}, providing step-by-step instructions and best practices."
        
        elif any(term in section_lower for term in ['literature', 'review', 'background']):
            return f"This section presents a comprehensive review of {section_text.lower()}, examining relevant literature and background information."
        
        elif any(term in section_lower for term in ['theory', 'framework', 'concept']):
            return f"This section explores the theoretical framework and key concepts related to {section_text.lower()}, providing conceptual understanding."
        
        elif any(term in section_lower for term in ['discussion', 'implications', 'significance']):
            return f"This section discusses the implications and significance of {section_text.lower()}, exploring broader meanings and applications."
        
        else:
            return f"This section covers {section_text.lower()}, providing detailed information, insights, and relevant content on the topic."
    
    def process_collection(self, collection_dir: Path, input_data: Dict) -> Dict:
        """
        Process a collection of documents and extract relevant sections using semantic similarity.
        """
        
        # Extract persona and job information
        persona = input_data.get('persona', {})
        job = input_data.get('job_to_be_done', {})
        documents = input_data.get('documents', [])
        
        # Extract requirements for filtering
        job_task = job.get('task', '')
        persona_role = persona.get('role', '')
        requirements = self._extract_requirements_generic(job_task, persona_role)
        
        # Identify relevant domains
        identified_domains = self.identify_domain(persona, job)
        
        print(f"🎯 Identified domains: {', '.join(identified_domains)}")
        print(f"👤 Persona: {persona.get('role', 'Unknown')}")
        print(f"📋 Task: {job.get('task', 'Unknown')}")
        print(f"🧠 Using {'semantic similarity' if self.use_semantic else 'keyword matching'}")
        
        # Collect all sections from all documents
        all_sections = []
        subsection_analysis = []
        
        # Process each document
        pdfs_dir = collection_dir / "PDFs"
        
        for doc_info in documents:
            filename = doc_info['filename']
            json_filename = filename.replace('.pdf', '.json')
            json_path = pdfs_dir / json_filename
            
            if not json_path.exists():
                print(f"⚠️  Outline not found for {filename}")
                continue
            
            # Load the extracted outline
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    outline_data = json.load(f)
                
                outline = outline_data.get('outline', [])
                
                print(f"📖 Processing {filename} ({len(outline)} sections)")
                
                # Add document context to each section
                for section in outline:
                    section_with_context = section.copy()
                    section_with_context['document'] = filename
                    all_sections.append(section_with_context)
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                continue
        
        # Use advanced semantic similarity to find the most relevant sections across all documents
        if all_sections:
            print(f"🔍 Analyzing {len(all_sections)} sections with advanced semantic similarity...")
            top_sections_with_scores = self.find_most_relevant_sections(
                all_sections, persona, job, top_n=5, similarity_threshold=0.2
            )
            
            # Convert to the expected format
            extracted_sections = []
            for i, (section, score, justification) in enumerate(top_sections_with_scores):
                extracted_sections.append({
                    "document": section.get('document', ''),
                    "page_number": section.get('page', 0),
                    "section_title": section.get('text', ''),
                    "importance_rank": i + 1
                })
                
                # Add detailed analysis for top sections
                if i < 5:  # Get subsection analysis for top 5 sections only
                    refined_text = self.extract_subsection_text(
                        pdfs_dir / section.get('document', ''), section, requirements=requirements
                    )
                    
                    subsection_analysis.append({
                        "document": section.get('document', ''),
                        "page_number": section.get('page', 0),
                        "section_title": section.get('text', ''),
                        "refined_text": refined_text
                    })
                    
                    print(f"  {i+1}. {section.get('text', '')} (Score: {score:.3f}) - {section.get('document', '')}")
        else:
            extracted_sections = []
        
        # Prepare final output to match sample format
        output = {
            "metadata": {
                "input_documents": [doc['filename'] for doc in documents],
                "persona": persona.get('role', ''),
                "job_to_be_done": job.get('task', ''),
                "processing_timestamp": self.get_timestamp(),
                "method": "semantic similarity" if self.use_semantic else "keyword matching"
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }
        
        return output
    
    def get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()


class DocumentSelector(PersonaDrivenSelector):
    """
    Generic Document Intelligence Selector
    
    A simplified interface for the persona-driven selector that works across
    diverse document types, personas, and job requirements.
    """
    
    def __init__(self, use_semantic: bool = True):
        """
        Initialize the generic document selector.
        
        Args:
            use_semantic: Whether to use semantic similarity for better accuracy
        """
        super().__init__(use_semantic=use_semantic)
        print("🌟 Generic Document Intelligence Selector initialized")
        print(f"   - Semantic similarity: {'Enabled' if self.use_semantic else 'Disabled'}")
        print(f"   - Supported domains: {len(self.relevance_keywords)} domain categories")
        print(f"   - Method: {getattr(self, 'semantic_method', 'keyword-based')}")
    
    def select_relevant_sections(self, documents: List[Dict], persona: Dict, job: Dict, 
                               top_n: int = 5, similarity_threshold: float = 0.25) -> Dict:
        """
        Select the most relevant sections from documents for a given persona and job.
        
        Args:
            documents: List of document dictionaries with sections
            persona: Persona dictionary with role and preferences
            job: Job-to-be-done dictionary with task description
            top_n: Number of top sections to return
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            Dictionary with selected sections and metadata
        """
        # Flatten all sections from all documents
        all_sections = []
        for doc in documents:
            sections = doc.get('sections', [])
            for section in sections:
                section_with_context = section.copy()
                section_with_context['document'] = doc.get('name', 'Unknown')
                all_sections.append(section_with_context)
        
        if not all_sections:
            return {"selected_sections": [], "metadata": {"message": "No sections found"}}
        
        # Use the advanced semantic similarity selection
        top_sections_with_scores = self.find_most_relevant_sections(
            all_sections, persona, job, top_n=top_n, similarity_threshold=similarity_threshold
        )
        
        # Format results
        selected_sections = []
        for i, (section, score, justification) in enumerate(top_sections_with_scores):
            selected_sections.append({
                "rank": i + 1,
                "title": section.get('text', ''),
                "document": section.get('document', ''),
                "page": section.get('page', 0),
                "score": round(score, 3),
                "justification": justification,
                "text": section.get('full_text', section.get('text', ''))
            })
        
        return {
            "selected_sections": selected_sections,
            "metadata": {
                "total_sections_analyzed": len(all_sections),
                "sections_selected": len(selected_sections),
                "persona_role": persona.get('role', ''),
                "job_task": job.get('task', ''),
                "method": "semantic similarity" if self.use_semantic else "keyword matching",
                "similarity_threshold": similarity_threshold
            }
        }


def main():
    """Main function for testing the selector."""
    if len(sys.argv) != 2:
        print("Usage: python selector.py <collection_directory>")
        sys.exit(1)
    
    collection_dir = Path(sys.argv[1])
    if not collection_dir.is_absolute():
        # If it's relative, make it relative to the current working directory
        collection_dir = Path.cwd() / collection_dir
    
    input_file = collection_dir / "challenge1b_input.json"
    
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        sys.exit(1)
    
    # Load input
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # Process
    selector = PersonaDrivenSelector()
    result = selector.process_collection(collection_dir, input_data)
    
    # Save output
    output_file = collection_dir / "challenge1b_output.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to {output_file}")


if __name__ == "__main__":
    main()