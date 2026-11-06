"""
Personal Assistant - Main Launcher

Simple launcher for face login and face enrollment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.login_system.face_login import FaceLoginSystem


def main():
    """Main launcher."""
    print("=" * 60)
    print("  PERSONAL ASSISTANT - FACE LOGIN SYSTEM")
    print("=" * 60)
    print()
    print("  Options:")
    print("  [1] Login with face recognition")
    print("  [2] Enroll new face")
    print("  [3] Reset enrollment")
    print("  [4] Exit")
    print()

    choice = input("  Select option (1-4): ").strip()

    login_system = FaceLoginSystem()

    if choice == '1':
        # Check if enrolled
        if login_system.check_is_enrolled():
            success, user_info, confidence = login_system.verify()
            if success:
                print("\n  ✓ Access granted! Launching dashboard...")
                # TODO: Launch dashboard
            else:
                print("\n  ✗ Access denied!")
        else:
            print("\n  ⚠ No face enrolled. Please enroll first.")
            enroll_choice = input("  Enroll now? (Y/N): ").strip().lower()
            if enroll_choice == 'y':
                login_system.enroll()

    elif choice == '2':
        login_system.enroll()

    elif choice == '3':
        confirm = input("  Reset all enrollment data? Type 'yes': ").strip().lower()
        if confirm == 'yes':
            login_system.profile_manager.reset_enrollment()
            print("  ✓ Enrollment reset complete.")

    elif choice == '4':
        print("\n  Goodbye!")
        sys.exit(0)

    else:
        print("\n  Invalid option!")


if __name__ == "__main__":
    main()
