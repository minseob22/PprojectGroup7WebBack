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

# ============================
# 0. OpenAI 클라이언트
# ============================

# 환경변수 OPENAI_API_KEY 에 키를 넣어두고 사용
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ============================
# 1. TB & Radiomics 지식베이스
# ============================

KB_TEXT = """
[Basic TB Knowledge – WHO]
- 결핵(Tuberculosis, TB)은 결핵균(Mycobacterium tuberculosis)에 의해 발생하는 공기 매개 감염 질환이다.
- WHO 자료에 따르면 결핵은 여전히 전 세계 주요 사망 원인 중 하나지만, 예방과 치료가 가능한 질환이다.
- 표준 치료는 대개 4~6개월 이상 항결핵제 복용이 필요하며, 중간에 자의로 중단할 경우 내성 결핵(MDR-TB) 위험이 증가한다.
- 위험군: HIV 감염자, 당뇨병 환자, 영양실조 상태, 흡연자 등은 결핵 발병 위험이 높다.

[Latent vs Active TB – CDC]
- 잠복 결핵(Latent TB): 몸 안에 결핵균이 있지만 비활성 상태로, 증상이 없고 전염성도 없다.
- 활동성 결핵(Active TB): 기침(3주 이상), 발열, 체중 감소, 식욕 부진, 피 섞인 가래(객혈) 등의 증상이 나타나며 전염성이 있다.
- 전파 경로: 활동성 환자가 기침, 재채기를 할 때 공기 중으로 퍼진 미세한 비말을 통해 감염된다.
- 일반적으로 옷, 식기, 화장실 공유만으로는 결핵이 전염되지는 않는다.

[Chest X-ray Findings in TB – Radiology Assistant 등]
- 활동성 TB(Active): 상폐야(upper lobe) 중심의 침윤, 공동(cavity), tree-in-bud 패턴(작은 결절이 나뭇가지 모양으로 퍼진 양상) 등이 대표적인 소견이다.
- 비활동성 TB(Inactive): 석회화된 결절(calcified nodule), 섬유화(fibrosis), 흉막 비후(pleural thickening) 등이 보일 수 있다.
- 특히 tree-in-bud 패턴은 활동성 결핵을 강하게 시사하는 소견으로 알려져 있다.

[Fleischner Society Pulmonary Nodule Guidelines (요약)]
- 고형(solid) 폐결절 크기에 따른 추적 권고(저위험군 기준 예시):
  - 6mm 미만: 추가 추적 CT가 필요하지 않을 수 있다.
  - 6~8mm: 대개 6~12개월 후 CT 추적 검사를 고려한다.
  - 8mm 이상: 3개월 이내 CT, PET-CT 또는 조직 검사를 포함한 적극적인 평가를 권고한다.
- 부분고형(subsolid) 결절은 성장 속도가 느리고 악성 가능성이 있어, 더 긴 기간(최대 5년)까지 장기 추적이 필요할 수 있다.
- 최종 결정은 환자의 위험인자(흡연력, 나이, 기저질환)를 함께 고려하여 전문의가 판단한다.

[Radiomics in TB & Lung Disease]
- Radiomics는 CT나 X-ray 영상에서 사람 눈에 보이지 않는 미세한 텍스처, 모양, 강도 분포 등의 수치를 추출하여 질병의 특성을 정량화하는 기법이다.
- 몇몇 연구에서는 chest X-ray 기반 Radiomics 점수(RadScore 등)가 결핵 치료 반응 모니터링에 유용하며,
  치료가 잘 될수록 특정 텍스처 값이 감소하는 경향을 보인다고 보고한다.
- 다른 연구에서는 결절 모양, tree-in-bud 패턴과 관련된 Radiomics feature를 이용해 다제내성 결핵(MDR-TB)의 가능성을 예측하려는 시도도 있다.
- 이러한 Radiomics 분석은 육안 판독만으로는 구분하기 어려운 패턴을 반영하므로, 임상적 판단을 보조하는 참고 자료로 활용될 수 있다.

[주의 사항]
- 위 지식은 대규모 가이드라인 및 연구를 요약한 것으로, 개별 환자에게 그대로 적용하기보다는
  위험도 평가와 추가 검사 필요성을 판단하는 참고 정보로 사용해야 한다.
- 최종 진단과 치료 결정은 반드시 흉부 영상의학과 및 호흡기내과 전문의를 포함한 의료진이 내려야 한다.
"""

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

아래 JSON은 한 명의 환자 케이스에 대한 정보입니다.
- patient: 나이, 성별, 흡연력, 주요 임상 증상 등
- lesion_info: 병변 위치, 크기(mm), 개수, 형태(morphology) 등
- ml_results: Radiomics + 머신러닝 기반 예측 결과 (주요 라벨, 확률, 위험도 점수 등)

[CASE_JSON]
{json.dumps(case_json, ensure_ascii=False, indent=2)}
[END_OF_CASE_JSON]

위의 지식베이스와 JSON을 바탕으로, 아래 요구사항을 모두 한국어로 작성하세요.
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
     왜 추가 검사가 필요할 수 있는지 차분하게 설명하세요.
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

응답 형식(JSON):

{
  "doctor_summary": "string",
  "patient_friendly": "string",
  "risk_level": "low | moderate | high",
  "recommendations": ["string", "string", "..."],
  "disclaimer": "string"
}
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
        }
    }

    print(">> Radiomics + TB LLM report demo 실행 중...\n")
    report = generate_radiomics_report(demo_case)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n>> 완료.")
