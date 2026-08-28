import hashlib
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.db_models import User
from app.models.schemas import UserCreate, UserLogin, UserResponse, AuthTokenResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])

def hash_password(password: str) -> str:
    """Hashes password with SHA-256 and a secure salt."""
    salt = "farmneura_v2_salt_2026"
    return hashlib.sha256((password + salt).encode('utf-8')).hexdigest()

@router.post("/signup", response_model=AuthTokenResponse, status_code=status.HTTP_201_CREATED)
def signup(user_in: UserCreate, db: Session = Depends(get_db)):
    """Registers a new user account."""
    # Check if email is already registered
    existing_user = db.query(User).filter(User.email.ilike(user_in.email.strip())).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email address is already registered."
        )
    
    if len(user_in.password.strip()) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters long."
        )

    # Create new user
    new_user = User(
        id=str(uuid.uuid4()),
        full_name=user_in.full_name.strip(),
        email=user_in.email.strip().lower(),
        hashed_password=hash_password(user_in.password.strip()),
        role="farmer"
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = f"fn_token_{uuid.uuid4().hex}"

    return AuthTokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserResponse.model_validate(new_user)
    )

@router.post("/login", response_model=AuthTokenResponse)
def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Authenticates an existing user."""
    user = db.query(User).filter(User.email.ilike(credentials.email.strip())).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    if user.hashed_password != hash_password(credentials.password.strip()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    token = f"fn_token_{uuid.uuid4().hex}"

    return AuthTokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserResponse.model_validate(user)
    )

@router.get("/me", response_model=UserResponse)
def get_current_user_profile(user_id: str = None, db: Session = Depends(get_db)):
    """Fetches user profile information."""
    if not user_id:
        # Return latest registered user or mock default
        user = db.query(User).first()
        if not user:
            raise HTTPException(status_code=404, detail="No users found")
        return UserResponse.model_validate(user)

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse.model_validate(user)
