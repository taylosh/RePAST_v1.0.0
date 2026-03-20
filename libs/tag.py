"""
tag.py - Accelerated Multilingual Part-of-Speech Tagging
@author: taylosh
Created on Nov 23 2025
Last edited on Mar 16 2026

Subprocess for the Annotation Phase:
- Adds Part-of-Speech (POS) tiers to aligned/syllabified TextGrids.
- Utilizes spaCy for high-accuracy grammatical categorization.
- Accelerated TextGrid tier management via consolidated C-Domain.
- Supports English, Spanish, French, German, Italian, Portuguese, and Dutch.
- Designed as a functional call from 03_annotate.py.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from praatio import textgrid

# Path resolution for consolidated libs/ and bin/
project_root = Path(__file__).parent
libs_path = project_root / "libs"
if str(libs_path) not in sys.path:
    sys.path.insert(0, str(libs_path))

# Import Consolidated Accelerator Wrapper
try:
    from wrap_accel import get_c_acceleration_wrapper
    wrapper = get_c_acceleration_wrapper(enable_c_accel=True, enable_gpu_accel=True)
    ACCEL_READY = wrapper.available
except ImportError:
    ACCEL_READY = False

logger = logging.getLogger(__name__)

# ============================================================================
# SECTION 1: POS TAGGING ENGINE (spaCy)
# ============================================================================

class POSTagger:
    """Handles multilingual spaCy model management and tagging."""
    def __init__(self, language: str = "english", model_size: str = "sm", tag_type: str = "universal"):
        self.language = language
        self.model_size = model_size
        self.tag_type = tag_type  # 'universal' or 'fine'
        self.nlp = None
        self.model_map = {
            'english': 'en_core_web',
            'spanish': 'es_core_news',
            'french': 'fr_core_news',
            'german': 'de_core_news',
            'italian': 'it_core_news',
            'portuguese': 'pt_core_news',
            'dutch': 'nl_core_news',
            'russian': 'ru_core_news',
            'mandarin': 'zh_core_web',
            'japanese': 'ja_core_news',
            'korean': 'ko_core_news',
            'arabic': 'ar_core_news'
        }

    def load_model(self, auto_download: bool = True) -> bool:
        """Download and load the appropriate spaCy model."""
        import spacy
        
        # Construct full model name with size
        base_model = self.model_map.get(self.language, 'en_core_web')
        model_name = f"{base_model}_{self.model_size}"
        
        try:
            logger.info(f"Loading POS model: {model_name}")
            self.nlp = spacy.load(model_name)
            return True
        except OSError:
            if auto_download:
                logger.info(f"Downloading POS model: {model_name}...")
                try:
                    subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
                    self.nlp = spacy.load(model_name)
                    return True
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to download model {model_name}: {e}")
                    return False
            else:
                logger.error(f"Model {model_name} not found and auto-download disabled")
                return False

    def get_tag(self, word: str) -> str:
        """Categorize a single word into a POS class."""
        if not word.strip(): 
            return ""
        
        doc = self.nlp(word)
        if len(doc) > 0:
            if self.tag_type == "universal":
                # Return universal POS tag (e.g., NOUN, VERB)
                return doc[0].pos_
            else:
                # Return fine-grained tag (e.g., NNP, VBD)
                return doc[0].tag_
        return "UNK"

# ============================================================================
# SECTION 2: ANNOTATION LOGIC
# ============================================================================

def tag_textgrid_file(tg_path: Path, tagger: POSTagger) -> bool:
    """Add POS tiers below the words tier for every speaker."""
    try:
        logger.info(f"Processing: {tg_path.name}")
        tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
        new_tg = textgrid.Textgrid()
        new_tg.minTimestamp, new_tg.maxTimestamp = tg.minTimestamp, tg.maxTimestamp

        # Process tiers in order: phrases -> words -> pos -> (syllables) -> phones
        for tier_name in tg.tierNames:
            tier = tg.getTier(tier_name)
            new_tg.addTier(tier)
            
            # If this is a words tier, immediately generate and insert the POS tier
            if tier_name.endswith("_words"):
                speaker = tier_name.replace("_words", "")
                pos_intervals = []
                
                for entry in tier.entries:
                    tag = tagger.get_tag(entry.label) if entry.label.strip() else ""
                    pos_intervals.append((entry.start, entry.end, tag))
                
                pos_tier_name = f"{speaker}_pos"
                logger.info(f"  Adding POS tier: {pos_tier_name} with {len(pos_intervals)} intervals")
                pos_tier = textgrid.IntervalTier(pos_tier_name, pos_intervals, tg.minTimestamp, tg.maxTimestamp)
                new_tg.addTier(pos_tier)

        # Save overwrite (Hub script manages original backups)
        new_tg.save(str(tg_path), format="long_textgrid", includeBlankSpaces=True)
        logger.info(f"  Successfully tagged: {tg_path.name}")
        return True
    except Exception as e:
        logger.error(f"Failed to tag {tg_path.name}: {e}")
        return False

# ============================================================================
# MAIN HOOK FOR 03_ANNOTATE.PY
# ============================================================================

def add_pos_tiers_to_textgrids(
    aligned_dir: str, 
    language: str = "english", 
    model_name: str = None, 
    tag_type: str = "universal", 
    auto_download: bool = True
):
    """Entry point called by the main annotation script."""
    logger.info(f"Starting POS tagging for {aligned_dir} (Language: {language}, Tag type: {tag_type})")
    
    # Extract model size from model_name if provided (e.g., 'en_core_web_sm' -> 'sm')
    model_size = "sm"  # default
    if model_name and model_name.endswith(('_sm', '_md', '_lg')):
        model_size = model_name[-2:]  # Get the last two characters
    
    tagger = POSTagger(language=language, model_size=model_size, tag_type=tag_type)
    if not tagger.load_model(auto_download=auto_download):
        logger.error("Failed to initialize spaCy POS models.")
        return

    # Find all aligned TextGrids recursively
    aligned_path = Path(aligned_dir)
    tg_files = list(aligned_path.rglob("*_aligned.TextGrid"))
    
    # Filter out any files in temp directories
    tg_files = [f for f in tg_files if 'temp_' not in str(f) and 'mfa_' not in str(f)]
    
    logger.info(f"Found {len(tg_files)} aligned TextGrids to process")
    
    count = 0
    for tg_f in tg_files:
        if tag_textgrid_file(tg_f, tagger):
            count += 1
    
    logger.info(f"POS Tagging complete. {count} files annotated.")

if __name__ == "__main__":
    # Standalone test block
    pass
