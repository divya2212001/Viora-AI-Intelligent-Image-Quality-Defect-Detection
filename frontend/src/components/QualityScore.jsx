import {
  getQualityClass,
  getScoreDescription,
} from "../utils/helpers";

function QualityScore({ score, label }) {
  const numericScore = Number(score ?? 0);

  return (
    <div className="quality-score-card">

      <div className="score-left">

        <div className="section-label">
          OVERALL QUALITY
        </div>

        <div className="score-number">
          {numericScore}
          <span>/100</span>
        </div>

        <p className="score-description">
          {getScoreDescription(numericScore)}
        </p>

      </div>

      <div className="score-right">

        <div
          className={`quality-badge ${getQualityClass(label)}`}
        >
          {label || "UNKNOWN"}
        </div>

        <div className="score-meter">

          <div
            className="score-meter-fill"
            style={{
              width: `${Math.min(
                100,
                Math.max(0, numericScore)
              )}%`,
            }}
          />

        </div>

      </div>

    </div>
  );
}

export default QualityScore;