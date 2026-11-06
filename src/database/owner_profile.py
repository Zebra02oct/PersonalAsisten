"""
Owner Profile Manager

Manages the single owner profile for face recognition.
Handles enrollment, verification, and profile updates.
"""

import json
from datetime import datetime
from typing import Optional, Tuple, List
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy.orm import Session
from .connection import DatabaseConfig
from .models import OwnerProfile, LoginHistory


class OwnerProfileManager:
    """
    Manages the owner profile for face recognition.

    This is a singleton-style manager for single-user system.
    """

    def __init__(self):
        """Initialize the profile manager."""
        self.session: Optional[Session] = None
        self._ensure_profile_exists()

    def _get_session(self) -> Session:
        """Get database session."""
        return DatabaseConfig.get_session_factory()()

    def _ensure_profile_exists(self):
        """Ensure there's at least one profile record."""
        with self._get_session() as session:
            profile = session.query(OwnerProfile).first()
            if not profile:
                # Create default empty profile
                profile = OwnerProfile(
                    name="Owner",
                    is_enrolled=False,
                    sample_count=0
                )
                session.add(profile)
                session.commit()

    def get_profile(self) -> Optional[OwnerProfile]:
        """Get the owner profile."""
        with self._get_session() as session:
            return session.query(OwnerProfile).first()

    def is_enrolled(self) -> bool:
        """Check if owner is enrolled."""
        profile = self.get_profile()
        return profile.is_enrolled if profile else False

    def enroll(self, name: str, date_of_birth: str,
               signature: List[float],
               eye_shape_left: str = None,
               eye_shape_right: str = None,
               sample_count: int = 1) -> Tuple[bool, str]:
        """
        Enroll or update owner profile.

        Args:
            name: Owner's name
            date_of_birth: Date of birth in YYYY-MM-DD format
            signature: Face signature list (32 values)
            eye_shape_left: Left eye shape classification
            eye_shape_right: Right eye shape classification
            sample_count: Number of samples enrolled

        Returns:
            (success, message)
        """
        try:
            # Parse date
            dob = datetime.strptime(date_of_birth, "%Y-%m-%d").date()

            with self._get_session() as session:
                profile = session.query(OwnerProfile).first()

                if not profile:
                    profile = OwnerProfile()
                    session.add(profile)

                # Update profile
                profile.name = name
                profile.date_of_birth = dob
                profile.signature = json.dumps(signature)
                profile.eye_shape_left = eye_shape_left
                profile.eye_shape_right = eye_shape_right
                profile.sample_count = sample_count
                profile.is_enrolled = True
                profile.enrolled_at = datetime.utcnow()
                profile.updated_at = datetime.utcnow()

                session.commit()

                return True, f"Enrollment successful! Welcome, {name}."

        except ValueError as e:
            return False, f"Invalid date format. Use YYYY-MM-DD. Error: {e}"
        except Exception as e:
            return False, f"Enrollment failed: {e}"

    def update_signature(self, signature: List[float],
                       eye_shape_left: str = None,
                       eye_shape_right: str = None):
        """Update signature with new samples (running average)."""
        with self._get_session() as session:
            profile = session.query(OwnerProfile).first()

            if profile and profile.signature:
                # Load existing signature
                existing = json.loads(profile.signature)
                sample_count = profile.sample_count

                # Running average
                new_signature = [
                    (existing[i] * sample_count + signature[i]) / (sample_count + 1)
                    for i in range(len(existing))
                ]

                profile.signature = json.dumps(new_signature)
                profile.sample_count = sample_count + 1
            else:
                profile.signature = json.dumps(signature)
                profile.sample_count = 1

            if eye_shape_left:
                profile.eye_shape_left = eye_shape_left
            if eye_shape_right:
                profile.eye_shape_right = eye_shape_right

            profile.updated_at = datetime.utcnow()
            session.commit()

    def get_signature(self) -> Optional[List[float]]:
        """Get stored signature."""
        profile = self.get_profile()
        if profile and profile.signature:
            return json.loads(profile.signature)
        return None

    def get_user_info(self) -> Optional[dict]:
        """Get owner info for display."""
        profile = self.get_profile()
        if profile:
            return {
                'name': profile.name,
                'date_of_birth': str(profile.date_of_birth) if profile.date_of_birth else None,
                'age': profile.age,
                'eye_shape_left': profile.eye_shape_left,
                'eye_shape_right': profile.eye_shape_right,
                'sample_count': profile.sample_count,
                'is_enrolled': profile.is_enrolled,
                'enrolled_at': str(profile.enrolled_at) if profile.enrolled_at else None,
                'last_login': str(profile.last_login) if profile.last_login else None,
            }
        return None

    def record_login(self, success: bool, confidence: float = None,
                   message: str = None):
        """Record a login attempt."""
        with self._get_session() as session:
            history = LoginHistory(
                success=success,
                confidence=confidence,
                message=message
            )
            session.add(history)

            # Update last login
            profile = session.query(OwnerProfile).first()
            if profile:
                profile.last_login = datetime.utcnow()

            session.commit()

    def get_login_history(self, limit: int = 10) -> List[LoginHistory]:
        """Get recent login history."""
        with self._get_session() as session:
            return session.query(LoginHistory)\
                .order_by(LoginHistory.timestamp.desc())\
                .limit(limit)\
                .all()

    def reset_enrollment(self):
        """Reset enrollment - clear signature but keep profile."""
        with self._get_session() as session:
            profile = session.query(OwnerProfile).first()
            if profile:
                profile.name = "Owner"
                profile.date_of_birth = None
                profile.signature = None
                profile.eye_shape_left = None
                profile.eye_shape_right = None
                profile.sample_count = 0
                profile.is_enrolled = False
                profile.enrolled_at = None
                profile.updated_at = datetime.utcnow()
                session.commit()

    def delete_profile(self):
        """Delete owner profile completely."""
        with self._get_session() as session:
            session.query(LoginHistory).delete()
            session.query(OwnerProfile).delete()
            session.commit()


# Singleton instance
_manager = None


def get_manager() -> OwnerProfileManager:
    """Get singleton manager instance."""
    global _manager
    if _manager is None:
        _manager = OwnerProfileManager()
    return _manager
