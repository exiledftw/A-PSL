import pandas as pd
import re

class AvatarNLPRouter:
    def __init__(self, animation_csv_path, variants_csv_path):
        # Load the CSVs
        self.animations_df = pd.read_csv(animation_csv_path, encoding='utf-8')
        self.variants_df = pd.read_csv(variants_csv_path, encoding='utf-8')
        
        # Build dictionaries for fast lookup
        # 1. Map slang/synonyms to canonical roots
        self.variant_map = dict(zip(
            self.variants_df['variant'].astype(str).str.lower(), 
            self.variants_df['canonical'].astype(str).str.lower()
        ))
        
        # 2. Map Roman Urdu and English directly to the .bvh animation filename
        self.animation_map = {}
        for _, row in self.animations_df.iterrows():
            anim_file = str(row['animation_clip']) + ".bvh"
            if pd.notna(row['urdu_roman']):
                self.animation_map[str(row['urdu_roman']).lower()] = anim_file
            if pd.notna(row['english']):
                self.animation_map[str(row['english']).lower()] = anim_file

    def translate_to_avatar(self, sentence):
        print(f"\n[Doctor (Whisper AI) Speaks]: \"{sentence}\"")
        
        # Clean and tokenize
        words = re.findall(r'\b\w+\b', sentence.lower())
        
        animation_sequence = []
        
        for word in words:
            # 1. Check if word is a synonym/variant, convert to canonical
            canonical = self.variant_map.get(word, word)
            
            # 2. Look up the canonical word to get the animation file
            motion_file = self.animation_map.get(canonical)
            
            if motion_file:
                animation_sequence.append(motion_file)
                print(f" -> Found '{word}'  --> Mapping to Avatar File: [{motion_file}]")
            else:
                # Ignore stop words (I, and, the, is, am)
                pass
                
        print(f"\n[FINAL AVATAR EXECUTION QUEUE]: {animation_sequence}\n")
        print("-" * 50)
        return animation_sequence

if __name__ == "__main__":
    try:
        router = AvatarNLPRouter(
            animation_csv_path=r"E:\datasets\animation_clips.csv",
            variants_csv_path=r"E:\datasets\word_variants.csv"
        )
        
        # Test 1: Mixed English and Roman Urdu
        router.translate_to_avatar("Salam, I need some madad and water please")
        
        # Test 2: Using synonyms (e.g. 'greet' instead of 'salam', 'paani' instead of 'water')
        router.translate_to_avatar("Greet the patient and give them paani")
        
    except Exception as e:
        print("Error loading dataset:", e)
