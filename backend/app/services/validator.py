"""
Indonesian License Plate Validator and Normalizer
"""
import re
from typing import Tuple, Optional


class PlateValidator:
    """
    Validator for Indonesian license plate format.
    
    Indonesian plate format: [Region Code] [Numbers] [Series Letters]
    Examples: B 1234 ABC, D 1 A, AB 12 CD, H 1234 AA
    
    Region codes are typically 1-2 letters indicating the province/region.
    Numbers can be 1-4 digits.
    Series letters are typically 1-3 letters.
    """
    
    # Indonesian region codes (common ones)
    REGION_CODES = {
        # DKI Jakarta
        'B': 'Jakarta',
        # West Java
        'D': 'Bandung', 'E': 'Cirebon', 'F': 'Bogor', 'T': 'Karawang', 'Z': 'Garut',
        # Central Java  
        'G': 'Pekalongan', 'H': 'Semarang', 'K': 'Pati', 'R': 'Banyumas', 'AA': 'Kedu',
        'AD': 'Surakarta', 'AB': 'Yogyakarta',
        # East Java
        'L': 'Surabaya', 'M': 'Madura', 'N': 'Malang', 'P': 'Jember', 'S': 'Bojonegoro',
        'W': 'Sidoarjo', 'AE': 'Madiun', 'AG': 'Kediri',
        # Sumatra
        'BA': 'Sumbar', 'BB': 'Tapanuli', 'BD': 'Bengkulu', 'BE': 'Lampung',
        'BG': 'Sumsel', 'BH': 'Jambi', 'BK': 'Sumut', 'BL': 'Aceh', 'BM': 'Riau',
        'BN': 'Babel', 'BP': 'Kepri',
        # Kalimantan
        'DA': 'Kalsel', 'KB': 'Kalbar', 'KH': 'Kalteng', 'KT': 'Kaltim', 'KU': 'Kaltara',
        # Sulawesi
        'DB': 'Manado', 'DC': 'Gorontalo', 'DD': 'Sulsel', 'DL': 'Sitaro',
        'DM': 'Gorontalo', 'DN': 'Sulteng', 'DT': 'Sultra', 'DW': 'Sulbar',
        # Eastern Indonesia
        'DE': 'Maluku', 'DG': 'Malut', 'DK': 'Bali', 'DR': 'NTB', 'DH': 'NTT',
        'PA': 'Papua', 'PB': 'Papua Barat',
        # Special
        'RI': 'Pemerintah',
    }
    
    # Character substitution map for normalization
    CHAR_TO_DIGIT = {
        'O': '0', 'o': '0',
        'I': '1', 'i': '1', 'l': '1', 'L': '1',
        'Z': '2', 'z': '2',
        'S': '5', 's': '5',
        'B': '8', 'b': '8',
    }
    
    DIGIT_TO_CHAR = {
        '0': 'O',
        '1': 'I',
        '2': 'Z',
        '5': 'S',
        '8': 'B',
    }
    
    # Regex pattern for Indonesian plate
    # Format: 1-2 letters, 1-4 digits, 1-3 letters
    PLATE_PATTERN = re.compile(
        r'^([A-Z]{1,2})\s*(\d{1,4})\s*([A-Z]{1,3})$',
        re.IGNORECASE
    )
    
    # Pattern to detect expiry date (MM-YY, MM/YY, MM.YY, MMYY, or with spaces)
    EXPIRY_DATE_PATTERNS = [
        re.compile(r'(0[1-9]|1[0-2])[-/.\s](\d{2})'),  # MM-YY, MM/YY, MM.YY, MM YY
        re.compile(r'^(0[1-9]|1[0-2])(\d{2})$'),  # MMYY
        re.compile(r'\b\d{2}[-/.\s]\d{2}\b'),  # Any XX-YY pattern
    ]
    
    @classmethod
    def remove_expiry_date(cls, text: str) -> str:
        """
        Remove expiry date patterns from the text.
        
        Indonesian motorcycle plates have expiry dates in MM-YY or MM/YY format.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Text with expiry date removed
        """
        if not text:
            return ""
        
        result = text
        
        # Remove common expiry date patterns
        for pattern in cls.EXPIRY_DATE_PATTERNS:
            result = pattern.sub('', result)
        
        # Also remove standalone 4-digit numbers that look like MMYY
        # But be careful not to remove valid plate numbers
        words = result.split()
        filtered_words = []
        
        for word in words:
            # Skip if it's a pure MMYY pattern at the end
            cleaned_word = re.sub(r'[^A-Za-z0-9]', '', word)
            if len(cleaned_word) == 4 and cleaned_word.isdigit():
                month = int(cleaned_word[:2])
                year = int(cleaned_word[2:])
                # If it looks like a valid expiry date (month 01-12, year 20-30)
                if 1 <= month <= 12 and 20 <= year <= 35:
                    continue  # Skip this word
            filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    @classmethod
    def normalize_text(cls, text: str) -> str:
        """
        Normalize OCR text by removing unwanted characters and standardizing format.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Normalized text (uppercase, cleaned)
        """
        if not text:
            return ""
        
        # First remove expiry dates
        text = cls.remove_expiry_date(text)
        
        # Remove all non-alphanumeric characters except spaces
        cleaned = re.sub(r'[^A-Za-z0-9\s]', '', text)
        
        # Collapse multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Convert to uppercase
        cleaned = cleaned.upper()
        
        return cleaned
    
    @classmethod
    def normalize_plate_number(cls, text: str) -> str:
        """
        Normalize a detected plate number to standard Indonesian format.
        
        This applies character substitution based on position context:
        - In the region code (beginning): digits become letters
        - In the number section (middle): letters become digits
        - In the series section (end): digits become letters
        
        Args:
            text: Cleaned/normalized OCR text
            
        Returns:
            Plate number in standard format (e.g., "B 1234 ABC")
        """
        if not text:
            return ""
        
        # First, normalize the text (includes expiry date removal)
        normalized = cls.normalize_text(text)
        
        # Remove all spaces to process
        chars = normalized.replace(' ', '')
        
        if len(chars) < 3:
            return normalized
        
        # Try to identify the three parts
        # Find the transition from letters to digits (end of region code)
        region_end = 0
        for i, c in enumerate(chars):
            if c.isdigit():
                region_end = i
                break
        else:
            # No digits found, might be all letters - return as is
            return normalized
        
        # If first character might be a misread digit (like 8 for B)
        # and we have no region code, try to fix it
        if region_end == 0:
            first_char = chars[0]
            if first_char in cls.DIGIT_TO_CHAR:
                chars = cls.DIGIT_TO_CHAR[first_char] + chars[1:]
                region_end = 1
        
        # Find the transition from digits to letters (end of numbers)
        number_end = region_end
        for i in range(region_end, len(chars)):
            if chars[i].isalpha():
                number_end = i
                break
        else:
            number_end = len(chars)
        
        # Extract parts
        region_part = chars[:region_end] if region_end > 0 else ""
        number_part = chars[region_end:number_end]
        series_part = chars[number_end:] if number_end < len(chars) else ""
        
        # Normalize region code (should be letters)
        normalized_region = ""
        for c in region_part:
            if c.isdigit() and c in cls.DIGIT_TO_CHAR:
                normalized_region += cls.DIGIT_TO_CHAR[c]
            else:
                normalized_region += c.upper()
        
        # Normalize number part (should be digits)
        normalized_number = ""
        for c in number_part:
            if c.isalpha() and c.upper() in cls.CHAR_TO_DIGIT:
                normalized_number += cls.CHAR_TO_DIGIT[c.upper()]
            elif c.isdigit():
                normalized_number += c
        
        # Normalize series part (should be letters)
        normalized_series = ""
        for c in series_part:
            if c.isdigit() and c in cls.DIGIT_TO_CHAR:
                normalized_series += cls.DIGIT_TO_CHAR[c]
            else:
                normalized_series += c.upper()
        
        # Format with spaces
        parts = [p for p in [normalized_region, normalized_number, normalized_series] if p]
        return " ".join(parts)
    
    @classmethod
    def validate(cls, plate_number: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if a plate number follows Indonesian format.
        
        Args:
            plate_number: Plate number to validate
            
        Returns:
            Tuple of (is_valid, region_name or None)
        """
        if not plate_number:
            return False, None
        
        # Normalize first
        normalized = cls.normalize_plate_number(plate_number)
        
        # Match against pattern
        match = cls.PLATE_PATTERN.match(normalized.replace(' ', ''))
        
        if not match:
            # Try with normalized (spaced) version
            match = cls.PLATE_PATTERN.match(normalized.replace(' ', ''))
        
        if not match:
            return False, None
        
        region_code = match.group(1).upper()
        
        # Check if region code is valid
        region_name = cls.REGION_CODES.get(region_code)
        
        # Even if region is not in our list, it could still be valid
        # (we may not have all region codes)
        return True, region_name
    
    @classmethod
    def process_ocr_result(cls, raw_text: str) -> Tuple[str, bool, Optional[str]]:
        """
        Process raw OCR result: normalize and validate.
        
        Args:
            raw_text: Raw OCR output
            
        Returns:
            Tuple of (normalized_plate, is_valid, region_name)
        """
        # First clean the raw text by removing expiry dates
        cleaned_text = cls.remove_expiry_date(raw_text)
        
        # Then normalize
        normalized = cls.normalize_plate_number(cleaned_text)
        
        # Validate
        is_valid, region = cls.validate(normalized)
        
        return normalized, is_valid, region
