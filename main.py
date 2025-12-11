from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
import shutil
import os
import uuid
import json

# 1. 파일 이름 변경 반영 (models -> DBTable)
import DBTable
import database

# --- 초기 설정 ---
DBTable.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

os.makedirs("static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- [초기 세팅] 서버 켜질 때 테스트용 의사(User) 자동 생성 ---
@app.on_event("startup")
def startup_event():
    db = database.SessionLocal()
    user = db.query(DBTable.User).first()
    if not user:
        # 테스트용 의사 계정 생성 (ID: 1)
        test_user = DBTable.User(
            email="doctor@test.com",
            full_name="Dr. Kim",
            hashed_password="dummy_password"
        )
        db.add(test_user)
        db.commit()
    db.close()

# --- Pydantic 스키마 ---
class PatientCreate(BaseModel):
    name: str
    birthDate: str
    gender: str

# --- API 구현 ---

# 1. 환자 등록 (User ID 연결 추가)
@app.post("/patients/register")
def register_patient(patient: PatientCreate, db: Session = Depends(get_db)):
    existing = db.query(DBTable.Patient).filter(
        DBTable.Patient.name == patient.name,
        DBTable.Patient.birthDate == patient.birthDate
    ).first()
    
    if existing:
        return {"status": "found", "message": "기존 환자 기록을 불러옵니다.", "data": existing}
    
    new_patient = DBTable.Patient(
        name=patient.name,
        birthDate=patient.birthDate,
        gender=patient.gender,
        user_id=1  # ⚠️ 현재 로그인 기능이 없으므로 1번 의사로 고정
    )
    
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)
    
    return {"status": "created", "message": "새 환자가 등록되었습니다.", "data": new_patient}

# 2. 환자 목록 조회
@app.get("/patients")
def get_patients(db: Session = Depends(get_db)):
    return db.query(DBTable.Patient).order_by(DBTable.Patient.created_at.desc()).all()

# [수정됨] 환자 상세 정보 조회 (404 에러 방지 및 데이터 포맷팅)
@app.get("/patients/{patient_id}")
def get_patient(patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(DBTable.Patient).filter(DBTable.Patient.id == patient_id).first()
    
    if not patient:
        # 환자가 없으면 404 에러를 명확하게 반환
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # 프론트엔드 포맷에 맞춰 딕셔너리로 반환
    return {
        "id": patient.id,
        "name": patient.name,
        "birthDate": patient.birthDate,
        "gender": patient.gender,
        # created_at이 있을 경우 날짜 문자열로 변환
        "lastVisit": patient.created_at.strftime("%Y-%m-%d") if patient.created_at else "",
    }

# 3. 특정 환자의 메시지 가져오기 (데이터 직렬화 강화)
@app.get("/patients/{patient_id}/messages")
def get_messages(patient_id: int, db: Session = Depends(get_db)):
    # 해당 환자의 모든 세션 조회
    sessions = db.query(DBTable.ChatSession).filter(DBTable.ChatSession.patient_id == patient_id).all()
    
    all_messages = []
    for session in sessions:
        for msg in session.messages:
            # ORM 객체를 직접 리턴하면 에러가 날 수 있으므로 딕셔너리로 변환
            all_messages.append({
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "image_path": msg.image_path,
                "created_at": msg.created_at  # FastAPI가 자동으로 ISO format으로 변환해줌
            })
    
    # 시간순 정렬 (오래된 것 -> 최신 것)
    all_messages.sort(key=lambda x: x["created_at"])
    return all_messages












# 4. X-ray 분석 및 채팅 저장
@app.post("/analyze")
async def analyze_xray(
    patient_id: int = Form(...),
    text: str = Form(""),
    file: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    # A. 세션 관리 로직
    last_session = db.query(DBTable.ChatSession)\
        .filter(DBTable.ChatSession.patient_id == patient_id)\
        .order_by(DBTable.ChatSession.created_at.desc())\
        .first()
    
    if not last_session:
        last_session = DBTable.ChatSession(
            patient_id=patient_id,
            title=f"첫 방문 진료 ({uuid.uuid4().hex[:8]})"
        )
        db.add(last_session)
        db.commit()
        db.refresh(last_session)
    
    current_session_id = last_session.id

    # B. 이미지 저장
    image_url = None
    if file:
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = f"static/uploads/{filename}"
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        image_url = f"http://127.0.0.1:8000/{filepath}"

    # C. [User Message] DB 저장
    user_msg = DBTable.Message(
        session_id=current_session_id,
        role="user",
        content=text,
        image_path=image_url
    )
    db.add(user_msg)
    db.commit()

    # D. AI 분석 (가상)
    ai_response_text = ""
    diagnosis_data = None

    if file:
        ai_response_text = "AI 분석 결과: 폐렴(Pneumonia) 소견이 관찰됩니다.\n신뢰도는 88%이며, 우측 하부 폐엽에 음영이 증가해 있습니다."
        diagnosis_data = {
            "disease": "Pneumonia",
            "probability": 0.88,
            "location": "Right Lower Lobe"
        }
    else:
        ai_response_text = "네, 추가적인 증상이나 궁금한 점을 말씀해 주세요."

    # E. [Assistant Message] DB 저장
    ai_msg = DBTable.Message(
        session_id=current_session_id,
        role="assistant",
        content=ai_response_text
    )
    db.add(ai_msg)
    db.commit()
    db.refresh(ai_msg)

    # F. [Diagnosis Result] 별도 저장
    if diagnosis_data:
        diagnosis_entry = DBTable.DiagnosisResult(
            message_id=ai_msg.id,
            model_raw_json=diagnosis_data,
            gpt_interpretation=ai_response_text,
            doctor_feedback=None
        )
        db.add(diagnosis_entry)
        db.commit()

    return ai_msg