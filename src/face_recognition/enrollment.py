"""
Enrollment Manager Module

Manages enrolled face profiles for recognition.
Stores face signatures (not images) for privacy.

Data Storage:
- Profiles saved as JSON files
- Each profile contains: name, signature, metadata
- No face images stored - only geometric features
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np


class EnrollmentManager:
    """
    Manages face enrollment database.

    Storage format (JSON):
    {
        "profiles": {
            "user_id": {
                "name": "John Doe",
                "signature": [0.12, 0.34, ...],  // 32 features
                "created_at": "2024-01-01T12:00:00",
                "updated_at": "2024-01-01T12:00:00",
                "sample_count": 3,
                "metadata": {}
            }
        }
    }
    """

    def __init__(self, data_dir: str = None):
        """
        Initialize enrollment manager.

        Args:
            data_dir: Directory to store enrolled profiles.
                     If None, uses 'data' folder in module directory.
        """
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        else:
            data_dir = Path(data_dir)

        self.data_dir = data_dir
        self.data_file = self.data_dir / "enrolled_profiles.json"
        self.profiles = self._load_profiles()

    def _load_profiles(self) -> Dict:
        """Load profiles from JSON file."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # Corrupted file, start fresh
                return {"profiles": {}}
        return {"profiles": {}}

    def _save_profiles(self):
        """Save profiles to JSON file."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.data_file, 'w') as f:
            json.dump(self.profiles, f, indent=2)

    def generate_user_id(self, name: str) -> str:
        """Generate a user ID from name."""
        # Create simple ID from name
        base_id = name.lower().replace(" ", "_")
        # Check for duplicates
        if base_id in self.profiles["profiles"]:
            counter = 1
            while f"{base_id}_{counter}" in self.profiles["profiles"]:
                counter += 1
            return f"{base_id}_{counter}"
        return base_id

    def enroll(self, name: str, signature: np.ndarray,
               user_id: str = None,
               metadata: Dict = None) -> Tuple[str, bool]:
        """
        Enroll a new face or update existing.

        Args:
            name: Display name for the profile
            signature: Face signature (numpy array from FeatureExtractor)
            user_id: Optional custom user ID (auto-generated if None)
            metadata: Optional additional metadata

        Returns:
            (user_id, is_new) - The user ID and whether this is a new enrollment
        """
        is_new = True

        if user_id is None:
            user_id = self.generate_user_id(name)
        elif user_id in self.profiles["profiles"]:
            is_new = False

        # Convert numpy array to list for JSON storage
        signature_list = signature.tolist() if isinstance(signature, np.ndarray) else list(signature)

        now = datetime.now().isoformat()

        if is_new:
            # New enrollment
            self.profiles["profiles"][user_id] = {
                "name": name,
                "signature": signature_list,
                "created_at": now,
                "updated_at": now,
                "sample_count": 1,
                "metadata": metadata or {}
            }
        else:
            # Update existing - average with previous signature
            existing = self.profiles["profiles"][user_id]
            old_signature = np.array(existing["signature"])
            old_count = existing["sample_count"]

            # Running average
            new_count = old_count + 1
            avg_signature = (old_signature * old_count + signature) / new_count

            self.profiles["profiles"][user_id].update({
                "signature": avg_signature.tolist(),
                "updated_at": now,
                "sample_count": new_count,
                "metadata": metadata or existing.get("metadata", {})
            })

        self._save_profiles()
        return user_id, is_new

    def enroll_multiple(self, name: str, signatures: List[np.ndarray],
                       user_id: str = None,
                       metadata: Dict = None) -> Tuple[str, bool]:
        """
        Enroll with multiple samples (better accuracy).

        Args:
            name: Display name
            signatures: List of signature arrays
            user_id: Optional custom user ID
            metadata: Optional metadata

        Returns:
            (user_id, is_new)
        """
        if not signatures:
            raise ValueError("At least one signature required")

        # Average all signatures
        avg_signature = np.mean(signatures, axis=0)
        return self.enroll(name, avg_signature, user_id, metadata)

    def remove(self, user_id: str) -> bool:
        """
        Remove an enrolled profile.

        Args:
            user_id: ID of profile to remove

        Returns:
            True if removed, False if not found
        """
        if user_id in self.profiles["profiles"]:
            del self.profiles["profiles"][user_id]
            self._save_profiles()
            return True
        return False

    def get_profile(self, user_id: str) -> Optional[Dict]:
        """
        Get a profile by user ID.

        Returns:
            Profile dict or None if not found
        """
        return self.profiles["profiles"].get(user_id)

    def get_signature(self, user_id: str) -> Optional[np.ndarray]:
        """
        Get signature for a user.

        Returns:
            numpy array or None if not found
        """
        profile = self.get_profile(user_id)
        if profile:
            return np.array(profile["signature"])
        return None

    def get_all_signatures(self) -> Dict[str, Tuple[str, np.ndarray]]:
        """
        Get all enrolled signatures.

        Returns:
            Dict mapping user_id -> (name, signature_array)
        """
        result = {}
        for user_id, profile in self.profiles["profiles"].items():
            result[user_id] = (profile["name"], np.array(profile["signature"]))
        return result

    def list_profiles(self) -> List[Dict]:
        """
        List all enrolled profiles (without signatures).

        Returns:
            List of profile info dicts
        """
        profiles = []
        for user_id, profile in self.profiles["profiles"].items():
            profiles.append({
                "user_id": user_id,
                "name": profile["name"],
                "created_at": profile["created_at"],
                "sample_count": profile["sample_count"]
            })
        return profiles

    def get_count(self) -> int:
        """Get number of enrolled profiles."""
        return len(self.profiles["profiles"])

    def clear(self):
        """Remove all enrolled profiles."""
        self.profiles = {"profiles": {}}
        self._save_profiles()

    def export_profile(self, user_id: str, filepath: str) -> bool:
        """
        Export a single profile to a file.

        Args:
            user_id: Profile to export
            filepath: Destination file path

        Returns:
            True if successful
        """
        profile = self.get_profile(user_id)
        if profile:
            with open(filepath, 'w') as f:
                json.dump({
                    "user_id": user_id,
                    **profile
                }, f, indent=2)
            return True
        return False

    def import_profile(self, filepath: str, overwrite: bool = False) -> Optional[str]:
        """
        Import a profile from file.

        Args:
            filepath: Source file path
            overwrite: If True, overwrite existing profile with same user_id

        Returns:
            Imported user_id or None if failed
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            user_id = data.get("user_id")
            if not user_id:
                return None

            if user_id in self.profiles["profiles"] and not overwrite:
                return None

            # Clean up the data
            del data["user_id"]
            self.profiles["profiles"][user_id] = data
            self._save_profiles()
            return user_id

        except (json.JSONDecodeError, IOError, KeyError):
            return None


# CLI utility
if __name__ == "__main__":
    print("EnrollmentManager Module")
    print("=" * 50)

    manager = EnrollmentManager()

    print(f"Data file: {manager.data_file}")
    print(f"Enrolled profiles: {manager.get_count()}")

    print("\nCommands:")
    print("  manager.enroll(name, signature)  - Add face")
    print("  manager.remove(user_id)          - Remove face")
    print("  manager.list_profiles()         - List all")
    print("  manager.get_signature(user_id)  - Get signature")

    print("\n" + "=" * 50)
    print("Example:")
    print("  import numpy as np")
    print("  manager = EnrollmentManager()")
    print("  signature = np.random.rand(32)  # Fake signature")
    print("  manager.enroll('John', signature)")
    print("  print(manager.list_profiles())")
