from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, JSON, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

# 1. 사용자 (의사)
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    
    patients = relationship("Patient", back_populates="doctor")

# 2. 환자
class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id")) # 담당 의사
    name = Column(String, index=True)
    birthDate = Column(String) # YYYY-MM-DD
    gender = Column(String)     # M/F
    medical_history = Column(Text, nullable=True) # 기저질환 등
    created_at = Column(DateTime, default=datetime.now)
    
    doctor = relationship("User", back_populates="patients")
    sessions = relationship("ChatSession", back_populates="patient")

# 3. 채팅 세션 (사이드바 목록)
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    title = Column(String) # 예: "24년 1차 흉부 촬영"
    created_at = Column(DateTime, default=datetime.now)
    
    patient = relationship("Patient", back_populates="sessions")
    messages = relationship("Message", back_populates="session")

# 4. 메시지 (대화 내용)
class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    role = Column(String) # 'user' (의사) vs 'assistant' (AI)
    content = Column(Text) # 채팅 텍스트
    image_path = Column(String, nullable=True) # 업로드된 이미지 경로
    created_at = Column(DateTime, default=datetime.now)
    
    session = relationship("ChatSession", back_populates="messages")
    diagnosis = relationship("DiagnosisResult", back_populates="message", uselist=False)

# 5. 진단 결과 (AI 분석 데이터)
class DiagnosisResult(Base):
    __tablename__ = "diagnosis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey("messages.id")) # 어떤 메시지의 결과인지
    
    # 모델의 원본 예측 결과 (JSON으로 저장하면 유연함)
    # 예: {"Pneumonia": 0.88, "Edema": 0.12}
    model_raw_json = Column(JSON) 
    
    # GPT가 해석한 텍스트
    gpt_interpretation = Column(Text)
    
    # 의사의 피드백 (정답 여부) -> 추후 모델 개선용
    doctor_feedback = Column(Boolean, nullable=True) 
    
    message = relationship("Message", back_populates="diagnosis")