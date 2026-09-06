from forensic_viz import generate_pressure_heatmap, generate_stroke_diff_overlay, img_to_base64
from database import get_db, VerificationLog
from sqlalchemy.orm import Session

@app.get("/powerbi/telemetry")
def get_powerbi_dataset(db: Session = Depends(get_db)):
    """
    Outputs a flattened, schema-optimized dataset directly readable 
    by Power BI Desktop 'Get Data -> Web' connector.
    """
    records = db.query(VerificationLog).all()
    rows = []
    for r in records:
        rows.append({
            "Log_ID": r.id,
            "Timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Date": r.timestamp.strftime("%Y-%m-%d"),
            "Hour": r.timestamp.hour,
            "Verdict": r.verdict,
            "Is_Genuine": 1 if "GENUINE" in r.verdict else 0,
            "Is_Forged": 1 if "FORGED" in r.verdict else 0,
            "Overall_Score": float(r.overall_score),
            "Metric_Confidence": float(r.metric_confidence),
            "Keypoint_Confidence": float(r.keypoint_confidence),
            "CNN_Similarity": float(getattr(r, "cnn_similarity", 0.0) or 0.0),
            "Ref_Corners": int(r.ref_corners or 0),
            "Test_Corners": int(r.test_corners or 0),
            "Corner_Divergence": abs(int(r.ref_corners or 0) - int(r.test_corners or 0)),
            "Pass_Threshold": 65.0
        })
    return rows