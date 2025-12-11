"""
radiomics_llm_report.py

역할:
- Radiomics + 머신러닝 결과(JSON)와 TB/Radiomics 지식베이스를 입력으로 받아
- GPT 모델이 해석한 의사용 요약 / 환자용 설명 / 권고사항 / 디스클레이머를 JSON으로 반환하는 모듈

사용 예시:
from radiomics_llm_report import generate_radiomics_report

report = generate_radiomics_report(case_json)
print(report)
"""

import os
import json
from typing import Dict, Any
from openai import OpenAI
from pathlib import Path

# ============================
# 0. OpenAI 클라이언트
# ============================

def _create_client() -> OpenAI:
    """
    OPENAI_API_KEY 환경변수가 없으면 바로 에러를 던져서
    왜 안 되는지 한 번에 알 수 있게 함.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다.\n"
            "PowerShell 예시:  setx OPENAI_API_KEY \"실제_API_키\"\n"
            "새 터미널을 연 뒤에 다시 실행해주세요."
        )
    return OpenAI(api_key=api_key)

client = _create_client()

# ============================
# 1. TB & Radiomics 지식베이스
# ============================

def load_kb_text() -> str:
    """
    같은 폴더에 있는 tb_1.txt ~ tb_8.txt를 순서대로 읽어서
    하나의 긴 텍스트로 합친다.
    """
    base_dir = Path(__file__).parent
    parts = []
    for i in range(1, 9):  # tb_1.txt ~ tb_8.txt
        path = base_dir / f"tb_{i}.txt"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                parts.append(f.read())

    kb_text = "\n\n".join(parts).strip()

    # 혹시 파일이 하나도 없을 때를 대비한 안전장치
    if not kb_text:
        kb_text = (
            "[주의] TB 지식베이스 파일(tb_1.txt ~ tb_8.txt)을 찾지 못했습니다. "
            "이 메시지는 디버깅용이며, 실제 서비스 시에는 해당 파일들을 반드시 준비해야 합니다."
        )
    return kb_text

KB_TEXT = load_kb_text()

# ============================
# 1-b. Radiomics 모델 학습 / 성능 컨텍스트
# ============================

# → 항상 LLM에 같이 들어가야 하는 전역 정보
MODEL_INFO = {
    "dataset": {
        "file": "outputs/radiomics/features_with_synthetic_normals.csv",
        "samples": 430,
        "features": 102,
        "tb": 330,
        "normal": 100,
        "note": "Normal 100 samples are synthetic; analysis only, not clinical."
    },
    "logreg": {
        "roc_auc": 1.0,
        "report": {
            "0": {
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
                "support": 20.0
            },
            "1": {
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
                "support": 66.0
            },
            "accuracy": 1.0,
            "macro avg": {
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
                "support": 86.0
            },
            "weighted avg": {
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
                "support": 86.0
            }
        }
    },
    "random_forest": {
        "roc_auc": 1.0,
        "report": {
            "0": {
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
                "support": 20.0
            },
            "1": {
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
                "support": 66.0
            },
            "accuracy": 1.0,
            "macro avg": {
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
                "support": 86.0
            },
            "weighted avg": {
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
                "support": 86.0
            }
        },
        "feature_importance_top10": [
            {"feature": "original_glcm_Imc1", "importance": 0.22146515262089497},
            {"feature": "original_firstorder_Maximum", "importance": 0.09920766439252111},
            {"feature": "original_firstorder_Range", "importance": 0.09046836293554351},
            {"feature": "original_glcm_Imc2", "importance": 0.09046461237081396},
            {"feature": "original_glcm_Idmn", "importance": 0.08295208563768676},
            {"feature": "original_glcm_Idn", "importance": 0.0776288087458565},
            {"feature": "original_glszm_ZoneEntropy", "importance": 0.04313120124481508},
            {"feature": "original_glcm_MCC", "importance": 0.036601263782084933},
            {"feature": "original_firstorder_Minimum", "importance": 0.025533780704809675},
            {"feature": "original_glszm_SmallAreaEmphasis", "importance": 0.017861450200795617}
        ],
        "group_importance": {
            "firstorder": 0.2396907092672334,
            "glcm": 0.5722353732746164,
            "glrlm": 0.042304072164474696,
            "glszm": 0.07950720366207635,
            "gldm": 0.030843772974466065,
            "ngtdm": 0.017670140370472022,
            "shape2D": 0.017748728286661283
        }
    },
    "disclaimer": [
        "Synthetic normals used; external validation required.",
        "Not for clinical decision-making."
    ]
}

# ============================
# 2. 시스템 프롬프트
# ============================

SYSTEM_PROMPT = """
You are an AI assistant that supports clinicians by interpreting machine-learning based
radiomics predictions from chest X-ray images, with a focus on tuberculosis (TB) and lung nodules.

Your role:
- Use the provided medical knowledge base AND the radiomics JSON together.
- Summarize likely diagnoses and risk level based on model outputs and clinical context.
- Explain how confident the model seems (low / moderate / high), in qualitative terms.
- Suggest reasonable next steps (e.g., CT, PET-CT, follow-up interval, referral).

Important constraints:
- You are a decision support tool, NOT a doctor.
- Never make definitive diagnoses or treatment decisions.
- Always include a disclaimer that this is a research prototype and must be reviewed by a physician.
- If the input is ambiguous or probabilities are similar, clearly say that uncertainty is high.
- Be conservative and safety-oriented in your recommendations.
"""

# ============================
# 3. 유저 프롬프트 생성 함수
# ============================

def build_user_prompt(case_json: Dict[str, Any], kb_text: str = KB_TEXT) -> str:
    """
    Radiomics + ML 결과(JSON)와 TB/Radiomics 지식베이스 텍스트를 받아
    LLM에 줄 user 프롬프트 문자열을 구성한다.
    """

    return f"""
다음은 Radiomics 기반 머신러닝 모델이 흉부 X-ray에서 추출한 결과(JSON)와,
결핵(TB) 및 폐결절, Radiomics 관련 의학 지식베이스입니다.

[MEDICAL_KNOWLEDGE_BASE]
{kb_text}
[END_OF_KNOWLEDGE_BASE]

[MODEL_TRAINING_AND_PERFORMANCE]
아래 JSON은 현재 Radiomics + 머신러닝 모델이 어떤 데이터셋으로 학습되었고,
어떤 지표(ROC-AUC, classification report), 어떤 feature group을 중요하게 보고 있는지에 대한 요약 정보입니다.
특히 normal 100개 샘플이 synthetic 이라는 점과,
이 모델이 아직 연구/분석용이며 임상 의사결정에 직접 사용되면 안 된다는 제한점을 반영해야 합니다.

{json.dumps(MODEL_INFO, ensure_ascii=False, indent=2)}
[END_OF_MODEL_TRAINING_AND_PERFORMANCE]

아래 JSON은 한 명의 환자 케이스에 대한 정보입니다.
- patient: 나이, 성별, 흡연력, 주요 임상 증상 등
- lesion_info: 병변 위치, 크기(mm), 개수, 형태(morphology) 등
- ml_results: Radiomics + 머신러닝 기반 예측 결과 (주요 라벨, 확률, 위험도 점수 등)
- radiomics_features: 원본 이미지에서 추출된 Radiomics feature 벡터 (optional)

[CASE_JSON]
{json.dumps(case_json, ensure_ascii=False, indent=2)}
[END_OF_CASE_JSON]

위의 지식베이스, 모델 정보, 케이스 JSON을 바탕으로, 아래 요구사항을 모두 한국어로 작성하세요.
반드시 지정된 JSON 형식으로만 응답해야 합니다. 추가 설명 문장은 넣지 마세요.

요구사항:
1. doctor_summary:
   - 임상의(의사)를 대상으로 하는 짧은 요약입니다.
   - 가장 가능성이 높은 진단 또는 감별진단을 서술하고,
     Radiomics 확률, TB 여부, 결절 크기/위치, 위험인자(흡연력 등)를 근거로
     전체적인 위험도를 설명하세요.
   - Fleischner guideline 또는 TB guideline과 연결되는 내용이 있으면 언급하세요.

2. patient_friendly:
   - 환자 또는 보호자가 이해하기 쉬운 설명입니다.
   - 의학 용어는 가능한 풀어서 설명하고, 현재 결과가 의미하는 바와
   - 왜 추가 검사가 필요할 수 있는지 차분하게 설명하세요.
   - 과도한 공포를 유발하지 않되, 필요한 경각심은 유지하세요.

3. risk_level:
   - overall clinical risk level을 "low", "moderate", "high" 중 하나로 선택하세요.
   - Radiomics 예측 확률, 위험인자, 결절 크기/모양 등을 종합적으로 고려합니다.

4. recommendations:
   - 2~4개의 구체적인 권고사항을 리스트로 제시합니다.
   - 예: "6개월 후 저선량 CT 추적 검사 권장", "호흡기내과 전문의 진료 의뢰" 등
   - 가이드라인과 지식베이스를 참고하여, 현실적인 검사/추적 전략을 제안하세요.

5. disclaimer:
   - 이 결과는 Radiomics 기반 AI 모델과 제한된 지식베이스를 바탕으로 한
     연구용/보조용 도구에 불과하며, 최종 진단과 치료 결정은 반드시
     담당 의사(전문의)가 내려야 한다는 점을 명시하세요.
   - 또한, Radiomics 모델은 synthetic normal 데이터를 포함한 연구용 모델이므로,
     임상 현장에서는 외부 검증을 거친 뒤 참고 자료로만 활용해야 합니다.

응답 형식(JSON):

{{
  "doctor_summary": "string",
  "patient_friendly": "string",
  "risk_level": "low | moderate | high",
  "recommendations": ["string", "string", "..."],
  "disclaimer": "string"
}}
"""

# ============================
# 4. 메인 함수: JSON -> LLM -> 리포트
# ============================

def generate_radiomics_report(
    case_json: Dict[str, Any],
    model: str = "gpt-4.1-mini",
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> Dict[str, Any]:
    """
    Radiomics ML 결과(JSON)를 입력으로 받아,
    TB/Radiomics 지식베이스를 함께 참고한 LLM 리포트를 JSON으로 반환한다.
    """

    if not isinstance(case_json, dict):
        raise TypeError("case_json은 dict 형태여야 합니다.")

    user_prompt = build_user_prompt(case_json, KB_TEXT)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content.strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # 디버깅을 위해 raw 출력 포함해서 에러 발생
        raise ValueError("LLM output is not valid JSON:\n" + content)

    # 최소 필드 검증 및 기본값 설정
    result = {
        "doctor_summary": parsed.get("doctor_summary", "").strip(),
        "patient_friendly": parsed.get("patient_friendly", "").strip(),
        "risk_level": parsed.get("risk_level", "moderate").strip(),
        "recommendations": parsed.get("recommendations", []),
        "disclaimer": parsed.get("disclaimer", "").strip(),
    }

    # recommendations가 문자열 하나로 올 수도 있어서 보정
    if isinstance(result["recommendations"], str):
        result["recommendations"] = [result["recommendations"]]

    return result

# ============================
# 5. 단독 실행 테스트용
# ============================

if __name__ == "__main__":
    # 데모용 케이스 JSON 예시
    demo_case = {
        "patient": {
            "id": "case_001",
            "age": 67,
            "sex": "M",
            "smoking_history": "30 pack-years",
            "major_clinical_info": [
                "chronic cough",
                "weight loss for 2 months"
            ]
        },
        "lesion_info": {
            "location": "right upper lobe",
            "size_mm": 22.4,
            "num_lesions": 1,
            "morphology": [
                "spiculated margin",
                "solid nodule"
            ]
        },
        "ml_results": {
            "primary_label": "active_tb_suspected",
            "primary_probability": 0.78,
            "tb_feature_positive": True,
            "top_candidates": [
                {"label": "active_tb_suspected", "probability": 0.78},
                {"label": "benign_nodule", "probability": 0.12},
                {"label": "inflammatory_lesion", "probability": 0.10}
            ],
            "risk_score": 0.80
        },
        # Radiomics feature 벡터가 있으면 이렇게 붙이면 됨 (예시)
        # "radiomics_features": {...}
    }

    print(">> Radiomics + TB LLM report demo 실행 중...\n")
    report = generate_radiomics_report(demo_case)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n>> 완료.")
