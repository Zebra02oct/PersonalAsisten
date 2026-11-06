"""
Owner Profile Database Model

SQLAlchemy model for single-user face recognition system.
Stores biometric signature and profile information.
"""

from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Boolean
from sqlalchemy.orm import relationship
from .connection import Base


class OwnerProfile(Base):
    """
    Single owner profile for face recognition login.

    This is a simplified model for personal use - only one user.
    """
    __tablename__ = 'owner_profile'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Profile info
    name = Column(String(255), nullable=False)
    date_of_birth = Column(Date, nullable=True)  # YYYY-MM-DD format
    is_enrolled = Column(Boolean, default=False)

    # Face signature (32 geometric features stored as JSON)
    signature = Column(String(1000), nullable=True)  # JSON string of list

    # Eye shape classification
    eye_shape_left = Column(String(50), nullable=True)
    eye_shape_right = Column(String(50), nullable=True)

    # Enrollment info
    sample_count = Column(Integer, default=0)

    # Timestamps
    enrolled_at = Column(DateTime, nullable=True)
    last_login = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def age(self):
        """Calculate age from date of birth."""
        if self.date_of_birth:
            today = date.today()
            return today.year - self.date_of_birth.year - (
                (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
            )
        return None

    def __repr__(self):
        status = "ENROLLED" if self.is_enrolled else "NOT ENROLLED"
        return f"<OwnerProfile(name='{self.name}', status={status})>"


class LoginHistory(Base):
    """Track login attempts for security monitoring."""
    __tablename__ = 'login_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    success = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=True)
    method = Column(String(50), default='face_recognition')  # face_recognition, fallback, etc
    message = Column(String(255), nullable=True)

    def __repr__(self):
        status = "SUCCESS" if self.success else "FAILED"
        return f"<LoginHistory({status}, confidence={self.confidence})>"
